"""SSG 문학 홈경기 종료 시각 기준 승차 보정 계수 학습 (문학경기장 3127).

검증 두 구간(학습 ~2025-06-30 → 7~10월, 학습 ~2026-02-28 → 3~6월)에서 LightGBM(127리프) 예측의 잔차(y - 예측, log1p)를
'실제 경기 종료 시각이 속한 시간대(rel = 시간 - 종료 시각의 정각)'별로 모아, rel = -1, 0, 1, 2 마다 잔차 ~ a + b * 관중(5천 단위)를 능형 회귀로 맞춘다.
모델은 종료 시각을 '시작 + 3.2시간'으로 추정해 쓰기 때문에 실제 종료 시각과 어긋나면 종료 시간대(rel 0, 1)의 승차를 크게 과소 예측한다.
입력: data/external/ssg_munhak_game_times.csv (KBO 게임센터), 출력: data/derived/ssg_end_profile.csv
사용법: python src/fit_ssg_profile.py   (약 4분)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

REL = (-1, 0, 1, 2)
Z_MEAN, Z_SD, LAM_B, LAM_A = 16568.0, 5000.0, 3.0, 5.0


def residuals():
    data = P.load_raw()
    params = {**T.BASE, **T.GRID[0]}
    rows = []
    for fi, (cut, a, b) in enumerate(T.FOLDS):
        tr, va, feats = T.fold_data(data, cut, a, b)
        Xtr = T.to_lgb(tr, feats)
        p = lgb.train(params, lgb.Dataset(Xtr, tr.y.to_numpy()), [1434, 1036][fi]).predict(T.to_lgb(va, feats, Xtr)).clip(0)
        d = va[["date", "station_id", "hour"]].copy()
        d["r"], d["fold"] = va.y.to_numpy() - p, fi
        rows.append(d[d.station_id == 3127])
    return pd.concat(rows)


def main():
    g = pd.read_csv("data/external/ssg_munhak_game_times.csv", parse_dates=["date"])
    R = residuals().merge(g[["date", "crowd", "end_h"]], on="date")
    R["rel"] = (R.hour - np.floor(R.end_h)).astype(int)
    R["z"] = (R.crowd - Z_MEAN) / Z_SD
    print("경기 수", R.date.nunique(), "| rel별 평균 잔차 (fold별):")
    print(R.groupby(["fold", "rel"]).r.mean().unstack(0).round(3).loc[-3:3].to_string())
    out = []
    for k in REL:
        a = R[R.rel == k]
        X = np.c_[np.ones(len(a)), a.z]
        w = np.linalg.solve(X.T @ X + np.diag([0.0, LAM_B]), X.T @ a.r.to_numpy())
        w[0] = a.r.sum() / (len(a) + LAM_A) if False else w[0]
        out.append(dict(rel=k, a=w[0], b=w[1], n=len(a)))
    pd.DataFrame(out).round(4).to_csv("data/derived/ssg_end_profile.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
