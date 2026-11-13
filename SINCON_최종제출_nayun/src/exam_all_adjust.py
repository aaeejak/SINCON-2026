"""시험일 일찍 하교 보정을 검바위 밖 다른 역으로 넓힌 것 (10~14시).

근거 (학습 데이터만 사용):
- 검바위(3208) 12시 급증으로 찾은 시험 기간 5블록(2025-04·07·10·12, 2026-04)에 다른 역들도 10~14시 승차가 오른다
  (예: 부평삼거리 13시, 인천가좌 11시, 독정 10~12시, 아시아드경기장 13시). 블록 간 역·시간별 상승의 상관 0.3~0.7.
- 상승 = log1p(블록 평일 평균) − log1p(블록 전후 14일 평일 중앙값), 역·시간별. 블록·시간대 공통 이동(전 역 중앙값)은 빼서
  모델이 날짜 효과로 이미 잡는 부분과 겹치지 않게 한다. 평소 승차 30명 미만인 역·시간, 검바위(이미 exam_adjust·geombawi_adjust로 보정)는 제외.
- 한 블록을 빼고 나머지 4블록 평균으로 그 블록을 맞히는 검증(LOBO): 축소 계수 k=0.466 (임계 0.1이면 0.548).
  오차(MSE) 25-04 0.0100→0.0085, 25-07 0.0072→0.0064, 26-04 0.0107→0.0091, 25-10 0.0047→0.0049, 25-12 0.0034→0.0039.
적용: 예측 구간 시험 추정일(exam_adjust.DAYS, 364일 이동 추정)의 10~14시에 log1p 예측 += k × 5블록 평균 상승.
사용법: python src/exam_all_adjust.py 입력.csv 출력.csv [k=0.466] [thr=0] [months=7,10]
"""
import sys

import numpy as np
import pandas as pd

from exam_adjust import DAYS

LIFT = "data/derived/exam_day_lift.csv"   # 만드는 코드: src/exam_all_adjust.py build


def build():
    tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
    tr["off"] = tr.date.map(cal.is_offday)
    B = {"25-04": pd.date_range("2025-04-28", "2025-05-01"),
         "25-07": pd.to_datetime(["2025-07-03", "2025-07-04", "2025-07-07", "2025-07-08", "2025-07-09"]),
         "25-10": pd.to_datetime(["2025-10-16", "2025-10-17", "2025-10-20", "2025-10-21"]),
         "25-12": pd.to_datetime(["2025-12-17", "2025-12-18", "2025-12-19", "2025-12-22", "2025-12-23"]),
         "26-04": pd.date_range("2026-04-27", "2026-04-30")}
    allex = set(d for v in B.values() for d in v)
    res, bm = {}, {}
    for k, days in B.items():
        lo, hi = days.min() - pd.Timedelta(days=14), days.max() + pd.Timedelta(days=14)
        w = tr[(tr.date.between(lo, hi)) & (tr.off == 0)]
        base = w[~w.date.isin(allex)].groupby(["station_id", "hour"]).boardings.median()
        ex = w[w.date.isin(days)].groupby(["station_id", "hour"]).boardings.mean()
        res[k] = np.log1p(ex) - np.log1p(base)
        bm[k] = base
    t = pd.DataFrame(res)
    t = t[pd.DataFrame(bm).min(axis=1) >= 30].drop(index=3208, level=0)
    t.to_csv(LIFT)
    print("저장", LIFT, len(t))


def adjust(sub, test, k=0.466, thr=0.0, months=(7, 10)):
    t = pd.read_csv(LIFT).set_index(["station_id", "hour"])
    t = t - t.groupby(level=1).transform("median")
    lift = t.mean(axis=1)
    lift = lift[lift.index.get_level_values(1).isin(range(10, 15))]
    lift = lift.where(lift.abs() >= thr, 0.0).rename("lift").reset_index()
    days = [d for d in pd.to_datetime(DAYS) if d.month in months]
    kk = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    kk["date"] = pd.to_datetime(kk["date"])
    kk = kk.merge(lift, on=["station_id", "hour"], how="left")
    m = kk.date.isin(days).to_numpy() & kk.lift.notna().to_numpy()
    x = np.log1p(kk.boardings.to_numpy(dtype=float))
    x[m] += k * kk.lift.to_numpy()[m]
    kk["boardings"] = np.expm1(x)
    out = sub[["example_id"]].merge(kk[["example_id", "boardings"]], on="example_id", how="left")
    return out, int(m.sum())


"""두 번째 학교 무리 (group B): 부평삼거리 13시·독정 10~12시 등은 검바위와 다른 날 시험을 본다.
2025: 04-28~30, 07-01~02, 09-29~10-02 / 2026: 04-28~30, 06-29~30 (2025-07-01의 364일 뒤 = 2026-06-30 이 학습 데이터에서 재현됨).
예측 구간 추정일(364일 이동, 검바위 DAYS와 겹치지 않는 날): 2026-07-01, 09-28, 09-29, 09-30, 10-01 (가을은 2026 추석 9/24~27 직후라 불확실).
상승표: data/derived/exam_day_lift_groupB.csv (블록 b25-07, b25-09, b26-06; 만드는 방법은 build와 같고 기준에서 A·B 시험일 모두 제외).
LOBO(3블록, 10~14시): k=0.48, 오차 b25-07 0.0187→0.0152, b25-09 0.0137→0.0118, b26-06 0.0087→0.0079.
사용법: python src/exam_all_adjust.py B 입력.csv 출력.csv [k=0.48] [days=all|jul]
"""
DAYS_B = ["2026-07-01", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01"]
LIFT_B = "data/derived/exam_day_lift_groupB.csv"


def adjust_b(sub, test, k=0.48, days=DAYS_B):
    t = pd.read_csv(LIFT_B).set_index(["station_id", "hour"])
    t = t.drop(index=3208, level=0, errors="ignore")
    t = t - t.groupby(level=1).transform("median")
    lift = t.mean(axis=1)
    lift = lift[lift.index.get_level_values(1).isin(range(10, 15))].rename("lift").reset_index()
    kk = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    kk["date"] = pd.to_datetime(kk["date"])
    kk = kk.merge(lift, on=["station_id", "hour"], how="left")
    m = kk.date.isin(pd.to_datetime(days)).to_numpy() & kk.lift.notna().to_numpy()
    x = np.log1p(kk.boardings.to_numpy(dtype=float))
    x[m] += k * kk.lift.to_numpy()[m]
    kk["boardings"] = np.expm1(x)
    return sub[["example_id"]].merge(kk[["example_id", "boardings"]], on="example_id", how="left"), int(m.sum())


def main():
    if sys.argv[1] == "build":
        return build()
    if sys.argv[1] == "B":
        te = pd.read_csv("data/raw/test.csv")
        sub = pd.read_csv(sys.argv[2])
        assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
        k = float(sys.argv[4]) if len(sys.argv) > 4 else 0.48
        days = DAYS_B[:1] if len(sys.argv) > 5 and sys.argv[5] == "jul" else DAYS_B
        out, n = adjust_b(sub, te, k, days)
        assert out.boardings.notna().all() and (out.boardings >= 0).all()
        out.to_csv(sys.argv[3], index=False)
        return print(f"보정 행 {n}, 저장 {sys.argv[3]}")
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    k = float(sys.argv[3]) if len(sys.argv) > 3 else 0.466
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    months = tuple(int(v) for v in sys.argv[5].split(",")) if len(sys.argv) > 5 else (7, 10)
    out, n = adjust(sub, te, k, thr, months)
    assert out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
