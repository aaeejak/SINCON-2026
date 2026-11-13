"""부천FC 홈경기(부천종합운동장 3754) 종료 시간대·관중 보정.

검증 두 구간(17경기)에서 우리 LightGBM은 경기 종료 시간대(rel 0 = 종료 시각의 정각)를 +0.66, 한 시간 전을 +0.20 과소 예측했고,
관중이 많을수록 더 컸다. rel -1, 0, 1, 2별로 잔차 ~ a + b·관중z (능형 회귀)를 맞춰, 경기 단위 leave-one-out으로
경기일 3754 RMSE 0.383 → 0.319. 현재 최고본(fix_cancel 계열)의 부천FC 경기일 모양은 우리 LightGBM과 거의 같아(rel 0 차이 -0.06) 중복이 아니다.
관중: data/external/events_master.csv의 ev_att (2026 경기는 K리그 공식 관중, 코리아컵 파주전·오픈트레이닝은 추정).
사용법: python src/fc_adjust.py fit                    # 계수 → data/derived/fc_end_profile.csv (약 4분)
        python src/fc_adjust.py 입력.csv 출력.csv [shrink=0.7]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STATION = 3754
REL = (-1, 0, 1, 2)


def games():
    ev = pd.read_csv("data/external/events_master.csv", parse_dates=["date"])
    fc = ev[(ev.station_id == STATION) & ev.ev_key.fillna("").str.startswith("fc")]
    return fc[["date", "ev_end", "ev_att"]].dropna().drop_duplicates("date")


def fit():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lightgbm as lgb
    import preprocess as P
    import train_lgbm as T
    data = P.load_raw()
    rows = []
    for fi, (cut, a, b) in enumerate(T.FOLDS):
        tr, va, feats = T.fold_data(data, cut, a, b)
        Xtr = T.to_lgb(tr, feats)
        p = lgb.train({**T.BASE, **T.GRID[0]}, lgb.Dataset(Xtr, tr.y.to_numpy()), [1434, 1036][fi]).predict(T.to_lgb(va, feats, Xtr)).clip(0)
        d = va[["date", "station_id", "hour"]].copy()
        d["r"] = va.y.to_numpy() - p
        rows.append(d[d.station_id == STATION])
    g = games()
    mu, sd = g.ev_att.mean(), g.ev_att.std()
    R = pd.concat(rows).merge(g, on="date")
    R["rel"] = (R.hour - np.floor(R.ev_end)).astype(int)
    R["z"] = (R.ev_att - mu) / sd
    out = []
    for k in REL:
        a = R[R.rel == k]
        X = np.c_[np.ones(len(a)), a.z]
        w = np.linalg.solve(X.T @ X + np.diag([0.0, 3.0]), X.T @ a.r.to_numpy())
        out.append(dict(rel=k, a=w[0], b=w[1], n=len(a), att_mean=mu, att_sd=sd))
    pd.DataFrame(out).round(4).to_csv("data/derived/fc_end_profile.csv", index=False)
    print(f"경기 {R.date.nunique()}개"); print(pd.DataFrame(out).round(3).to_string(index=False))


def adjust(sub, test, shrink=0.7):
    prof = pd.read_csv("data/derived/fc_end_profile.csv").set_index("rel")
    mu, sd = prof.att_mean.iloc[0], prof.att_sd.iloc[0]
    g = games()
    g = g[g.date >= "2026-07-01"]
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"])
    lp = np.log1p(k.boardings.to_numpy(dtype=float)).copy()
    n = 0
    for r in g.itertuples():
        for rel in prof.index:
            m = ((k.station_id == STATION) & (k.date == r.date) & (k.hour == int(np.floor(r.ev_end)) + int(rel))).to_numpy()
            lp[m] += shrink * (prof.loc[rel, "a"] + prof.loc[rel, "b"] * (r.ev_att - mu) / sd)
            n += int(m.sum())
    k["boardings"] = np.expm1(lp).clip(0)
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left"), n


def main():
    if sys.argv[1] == "fit":
        fit()
        return
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    out, n = adjust(sub, te, float(sys.argv[3]) if len(sys.argv) > 3 else 0.7)
    assert out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
