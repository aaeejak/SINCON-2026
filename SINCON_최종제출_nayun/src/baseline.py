"""인천 지하철 승차 인원 베이스라인 (SINCON 2026).

예측값 = (역, 요일유형, 시간)별 최근 W주 승차 인원의 대표값
         (log1p 평균 → expm1, 또는 중앙값)
       × (선택) 작년 같은 시기의 계절 보정 비율

설정 후보를 시간 기준 검증(RMSLE)으로 비교해 가장 좋은 것으로 test 전체를 예측한다.
로컬과 Kaggle 노트북 모두에서 실행된다 (데이터·출력 경로 자동 탐색).

사용법 (프로젝트 루트에서):
    python src/baseline.py [--data data/raw] [--out outputs/submission_baseline.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # Kaggle 노트북: calendar_data가 앞 셀에 정의되어 있음
    pass
try:
    from calendar_data import HOLIDAYS
except ImportError:
    HOLIDAYS = globals()["HOLIDAYS"]

N_TEST_ROWS = 157_194  # OT 자료 p.11
HOLIDAY_DATES = pd.to_datetime(list(HOLIDAYS))

# 검증 구간: (학습 마지막 날, 검증 시작, 검증 끝)
FOLDS = [
    ("2025-06-30", "2025-07-01", "2025-10-31"),  # 예측 기간과 같은 계절
    ("2026-02-28", "2026-03-01", "2026-06-30"),  # OT 베이스라인과 비슷한 최근 구간
]


# ---------------------------------------------------------------- 경로
def find_data_dir(arg):
    if arg:
        return Path(arg)
    for c in [Path("data/raw"), Path(__file__).resolve().parent.parent / "data/raw"
              if "__file__" in globals() else None]:
        if c is not None and (c / "train.csv").exists():
            return c
    hits = sorted(Path("/kaggle/input").glob("**/train.csv"))
    hits = [h for h in hits if (h.parent / "test.csv").exists()]
    if hits:
        return hits[0].parent
    raise FileNotFoundError("train.csv/test.csv 폴더를 찾지 못했습니다. --data로 지정하세요.")


def default_out():
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/submission.csv")
    return Path("outputs/submission_baseline.csv")


# ---------------------------------------------------------------- 예측
def add_daytype(df, mode):
    """mode='3': 평일/토/일·공휴일, mode='7': 월~일(공휴일은 일요일로 취급)."""
    dow = df["date"].dt.day_of_week
    hol = df["date"].isin(HOLIDAY_DATES)
    if mode == "3":
        dt = np.select([hol | (dow == 6), dow == 5], [2, 1], 0)
    else:
        dt = np.where(hol, 6, dow)
    return df.assign(daytype=dt)


def summarize(df, keys, stat):
    """keys별 승차 인원 대표값. logmean = expm1(mean(log1p(x)))."""
    if stat == "logmean":
        return np.expm1(np.log1p(df["boardings"]).groupby([df[k] for k in keys]).mean())
    return df.groupby(keys)["boardings"].median()


def seasonal_ratio(train, win_start, win_end, tgt_start, tgt_end):
    """작년 (예측구간 / 학습창) 비율을 요일유형별로, log 공간에서 계산. 1년 전 자료가 없으면 None."""
    off = pd.DateOffset(years=1)
    a = train[(train.date >= win_start - off) & (train.date <= win_end - off)]
    b = train[(train.date >= tgt_start - off) & (train.date <= tgt_end - off)]
    if a.empty or b.empty or a.date.min() > win_start - off + pd.Timedelta(days=7):
        return None
    common = set(a.station_id) & set(b.station_id)
    a, b = a[a.station_id.isin(common)], b[b.station_id.isin(common)]
    la = np.log1p(a.boardings).groupby(a.daytype).mean()
    lb = np.log1p(b.boardings).groupby(b.daytype).mean()
    return np.exp(lb - la).fillna(1.0)


def predict(train, target, *, weeks, stat, mode, seasonal):
    train = add_daytype(train, mode)
    target = add_daytype(target, mode)
    end = train.date.max()
    start = end - pd.Timedelta(weeks=weeks) + pd.Timedelta(days=1)
    recent = train[train.date >= start]

    keys = ["station_id", "daytype", "hour"]
    prof = summarize(recent, keys, stat).rename("pred")
    out = target.join(prof, on=keys)

    # 빈 조합 대체: (역, 시간) 최근 → (역, 요일유형, 시간) 전체 기간 → 0
    fb1 = summarize(recent, ["station_id", "hour"], stat)
    fb2 = summarize(train, keys, stat)
    out["pred"] = (out["pred"]
                   .fillna(out.join(fb1.rename("f"), on=["station_id", "hour"])["f"])
                   .fillna(out.join(fb2.rename("f"), on=keys)["f"])
                   .fillna(0.0))

    if seasonal:
        r = seasonal_ratio(train, start, end, target.date.min(), target.date.max())
        if r is not None:
            out["pred"] *= out["daytype"].map(r).fillna(1.0)

    return out["pred"].clip(lower=0).to_numpy()


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


# ---------------------------------------------------------------- 실행
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args(argv)  # 노트북 커널 인자 무시

    d = find_data_dir(args.data)
    out_path = Path(args.out) if args.out else default_out()
    print("데이터:", d)

    train = pd.read_csv(d / "train.csv", parse_dates=["date"])
    test = pd.read_csv(d / "test.csv", parse_dates=["date"])

    variants = [
        dict(weeks=w, stat=s, mode=m, seasonal=se)
        for w in (4, 8, 12)
        for s in ("logmean", "median")
        for m in ("3", "7")
        for se in (False, True)
    ]
    rows = []
    for v in variants:
        sc = []
        for cut, vs, ve in FOLDS:
            tr = train[train.date <= cut]
            va = train[(train.date >= vs) & (train.date <= ve)]
            # 검증 시점에 기록이 4주 미만인 역(신규 개통 역)은 제외
            first = tr.groupby("station_id").date.min()
            ok = first[first <= pd.Timestamp(cut) - pd.Timedelta(weeks=4)].index
            va = va[va.station_id.isin(ok)]
            p = predict(tr, va[["date", "station_id", "hour"]].reset_index(drop=True), **v)
            sc.append(rmsle(va.boardings.to_numpy(), p))
        rows.append({**v, "RMSLE": np.mean(sc), **{f"fold{i+1}": s for i, s in enumerate(sc)}})

    tbl = pd.DataFrame(rows).sort_values("RMSLE")
    print(tbl.round(4).head(8).to_string(index=False))
    best = tbl.iloc[0][["weeks", "stat", "mode", "seasonal"]].to_dict()
    best = dict(weeks=int(best["weeks"]), stat=best["stat"], mode=best["mode"],
                seasonal=bool(best["seasonal"]))
    print("선택된 설정:", best)

    sub = pd.DataFrame({
        "example_id": test["example_id"],
        "boardings": predict(train, test[["date", "station_id", "hour"]], **best),
    })

    # 제출 조건 검사 (OT p.11)
    assert len(sub) == N_TEST_ROWS, f"행 수 {len(sub)} != {N_TEST_ROWS}"
    assert sub.example_id.is_unique and sub.example_id.notna().all()
    assert np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"저장: {out_path} ({len(sub)}행)")
    return sub


if __name__ == "__main__":
    main()
