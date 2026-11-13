"""우리 모델(LightGBM·CatBoost·MLP·XGBoost) 혼합 비율을 검증 구간 예측으로 정한다 (Public 점수를 쓰지 않음).

검증 두 구간(학습 ~2025-06-30 → 7~10월, 학습 ~2026-02-28 → 3~6월)마다 v9 피처로 네 모델을 학습해 예측을 저장하고,
log1p 공간 가중 평균의 제곱오차를 최소화하는 비율을 구한다(합 1, 음수 없음). 한 구간에서 구한 비율을 다른 구간에서 평가해 과적합을 확인한다.
사용법: python src/tune_ours_weights.py          → outputs/tune_ours/preds_fold{0,1}.npz, 콘솔에 결과
"""
import json
import sys
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402
import train_mlp as M  # noqa: E402
import train_xgb as X  # noqa: E402

OUT = Path("outputs/tune_ours")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["lgbm", "cat", "mlp", "xgb"]


def fold_preds(data, fi):
    f = OUT / f"preds_fold{fi}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    cut, a, b = T.FOLDS[fi]
    tr, va, feats = T.fold_data(data, cut, a, b)
    y = va.y.to_numpy()
    Xtr, Xva = T.to_lgb(tr, feats), T.to_lgb(va, feats, T.to_lgb(tr, feats))
    cv = json.load(open("outputs/models/cv_lgbm_v9.json", encoding="utf-8"))
    ps = []
    for g in cv["grid"]:  # 3개 설정 평균 (v9와 같은 방식), 반복 수는 구간별 최적 값
        m = lgb.train({**T.BASE, **g["params"]}, lgb.Dataset(Xtr, tr.y.to_numpy()), g["iters"][fi])
        ps.append(m.predict(Xva))
    out = {"y": y, "lgbm": np.mean(ps, 0).clip(0)}
    Ctr, Cva = tr[feats].copy(), va[feats].copy()
    for c in P.CATEGORICAL:
        Ctr[c], Cva[c] = Ctr[c].astype(int), Cva[c].astype(int)
    cb = catboost.CatBoostRegressor(loss_function="RMSE", depth=8, learning_rate=0.06, l2_leaf_reg=10, iterations=800,
                                    random_seed=42, thread_count=-1, verbose=0)
    cb.fit(Ctr, tr.y.to_numpy())
    out["cat"] = cb.predict(Cva).clip(0)
    net, fit, _ = M.train_mlp(tr, feats, epochs=14, seed=42)
    xc, xn, _ = M.encode(va, feats, fit)
    out["mlp"] = M.predict(net, xc, xn, M.base_of(va))
    xm = xgb.train({**X.PARAMS, "seed": 42}, X.dm(Xtr, tr.y.to_numpy()), 1300)
    out["xgb"] = xm.predict(X.dm(Xva)).clip(0)
    np.savez(f, **out)
    return out


def best_w(preds, names):
    A = [np.stack([p[n] for n in names], 1) for p in preds]
    y = [p["y"] for p in preds]
    f = lambda w: sum(np.mean((a @ w - t) ** 2) for a, t in zip(A, y)) * 1e3
    k = len(names)
    r = minimize(f, np.ones(k) / k, bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                 method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return r.x


def rmsle(p, w, names):
    return float(np.sqrt(np.mean((np.stack([p[n] for n in names], 1) @ w - p["y"]) ** 2)))


def main():
    data = P.load_raw()
    P0, P1 = fold_preds(data, 0), fold_preds(data, 1)
    cur = {"lgbm": 0.72, "cat": 0.18, "mlp": 0.10}
    print("단독 RMSLE (7~10월 모사, 3~6월):", {n: (round(rmsle(P0, np.eye(4)[i], NAMES), 4), round(rmsle(P1, np.eye(4)[i], NAMES), 4)) for i, n in enumerate(NAMES)})
    wc = np.array([cur.get(n, 0.0) for n in NAMES])
    print("현재 0.72/0.18/0.10 :", round(rmsle(P0, wc, NAMES), 4), round(rmsle(P1, wc, NAMES), 4))
    for names in (["lgbm", "cat", "mlp"], NAMES):
        wa = best_w([P0, P1], names)
        w0, w1 = best_w([P0], names), best_w([P1], names)
        full = {n: round(float(v), 3) for n, v in zip(names, wa)}
        print(f"[{'+'.join(names)}] 두 구간 최적 비율 {full} → {rmsle(P0, wa, names):.4f}, {rmsle(P1, wa, names):.4f}")
        print(f"    교차: 7~10월에서 구한 비율을 3~6월에 {rmsle(P1, w0, names):.4f} | 3~6월에서 구한 비율을 7~10월에 {rmsle(P0, w1, names):.4f}  (비율 {np.round(w0, 2)} / {np.round(w1, 2)})")


if __name__ == "__main__":
    main()
