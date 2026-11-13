"""역별 최근 추세 보정: 모델이 역마다 다른 최근 전년 대비 증가율을 덜 따라가는 부분을 일부 되돌린다.

x_s = log(최근 3개월 실제 전년비) − log(예측 구간의 예측 전년비)   (평일, 하루 합계의 중앙값, 역 평균 0으로 맞춤)
보정: 역 s의 모든 행 log1p 예측 += k · clip(x_s, ±cap). 전체 수준은 그대로(역 간 상대 보정).
검증 B(2026-02까지 학습, 03~06 예측; 최근 = 2025-12~2026-02): 역 평균 잔차를 x로 회귀 → k (아래 fit 출력).
제외: 2025-06-28 검단 연장 개통 전후로 수준이 급변한 역(개통 전후 3주 비율이 전체 중앙값과 5% 넘게 다름),
      2025-01 이후 생긴 역(3107~3109), 문학경기장(3127, 야구 경기 영향).
사용법: python src/station_trend_adjust.py fit
        python src/station_trend_adjust.py 입력.csv 출력.csv [k] [cap=0.06] [workday_only=0]
"""
import sys

import numpy as np
import pandas as pd

tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
tr["off"] = tr.date.map(cal.is_offday)


def lvl(df, a, b, col="boardings", sts=None):
    w = df[(df.off == 0) & df.date.between(a, b)]
    if sts is not None:
        w = w[w.station_id.isin(sts)]
    return w.groupby(["station_id", "date"])[col].sum().groupby("station_id").median()


def stations():
    first = tr.groupby("station_id").date.min()
    old = first[first <= "2025-01-05"].index
    br = np.log(lvl(tr, "2025-07-07", "2025-07-25", sts=old) / lvl(tr, "2025-06-09", "2025-06-27", sts=old))
    br = br - br.median()
    keep = br[br.abs() <= 0.05].index.drop(3127, errors="ignore")
    return keep, br[br.abs() > 0.05]


def shift(a, b):
    return str(pd.Timestamp(a) - pd.Timedelta(days=364))[:10], str(pd.Timestamp(b) - pd.Timedelta(days=364))[:10]


def fit():
    keep, brk = stations()
    print("제외(개통 전후 급변)", brk.round(3).to_dict())
    v = pd.read_csv("share/for_team_valB/valB_nayun.csv", parse_dates=["date"])
    v["off"] = v.date.map(cal.is_offday)
    a, b = "2025-12-01", "2026-02-27"
    rec = np.log(lvl(tr, a, b, sts=keep) / lvl(tr, *shift(a, b), sts=keep))
    prev = lvl(tr, *shift("2026-03-02", "2026-06-30"), sts=keep)
    for col in ["pred_ours_tuned", "pred_lgbm_v9", "pred_xgb_v9"]:
        pyo = np.log(lvl(v, "2026-03-02", "2026-06-30", col, keep) / prev)
        act = np.log(lvl(v, "2026-03-02", "2026-06-30", "actual", keep) / prev)
        x = rec - pyo; y = act - pyo
        x, y = x - x.mean(), y - y.mean()
        k = (x * y).sum() / (x * x).sum()
        # 행 단위 효과: 잔차(log)에서 역 평균 편향을 얼마나 줄이는지
        vv = v[v.station_id.isin(keep)].copy()
        r = np.log1p(vv.actual) - np.log1p(vv[col])
        adj = vv.station_id.map(k * x.clip(-0.06, 0.06)).to_numpy()
        print(col, f"n={len(x)} corr={np.corrcoef(x, y)[0, 1]:.3f} k={k:.3f} 역편향 sd {y.std():.4f}→{(y - k * x).std():.4f}",
              f"| 검증B RMSLE(해당 역) {np.sqrt((r ** 2).mean()):.5f}→{np.sqrt(((r - adj) ** 2).mean()):.5f}")


def adjust(sub, k, cap=0.06, workday_only=False):
    keep, _ = stations()
    te = pd.read_csv("data/raw/test.csv", parse_dates=["date"]).merge(sub, on="example_id", how="left")
    te["off"] = te.date.map(cal.is_offday)
    rec = np.log(lvl(tr, "2026-04-01", "2026-06-30", sts=keep) / lvl(tr, *shift("2026-04-01", "2026-06-30"), sts=keep))
    pyo = np.log(lvl(te, "2026-07-01", "2026-10-31", sts=keep) / lvl(tr, *shift("2026-07-01", "2026-10-31"), sts=keep))
    x = rec - pyo
    x = (x - x.mean()).clip(-cap, cap)
    a = te.station_id.map(k * x).fillna(0.0).to_numpy()
    if workday_only:
        a = a * (te.off.to_numpy() == 0)
    out = te[["example_id"]].copy()
    out["boardings"] = np.expm1(np.log1p(te.boardings.to_numpy(dtype=float)) + a)
    return sub[["example_id"]].merge(out, on="example_id", how="left"), x


def main():
    if sys.argv[1] == "fit":
        return fit()
    sub = pd.read_csv(sys.argv[1])
    k = float(sys.argv[3]); cap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.06
    wo = len(sys.argv) > 5 and sys.argv[5] == "1"
    out, x = adjust(sub, k, cap, wo)
    assert len(out) == len(sub) and out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"역 {len(x)}개 보정, k={k}, x 범위 {x.min():.3f}~{x.max():.3f}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
