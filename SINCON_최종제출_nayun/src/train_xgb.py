"""XGBoost 버전 (v9 피처 53개 그대로) — LightGBM v9와 같은 입력으로 학습한 트리 모델.

사용법 (프로젝트 루트에서):
    python src/train_xgb.py valid   # 두 검증 구간: XGBoost 단독, LightGBM과 섞은 이득
    python src/train_xgb.py final   # 최종 학습(~2026-06-30) → outputs/xgb_v9.csv (시드 3개 평균)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

PARAMS = dict(objective="reg:squarederror", eta=0.05, max_depth=9, min_child_weight=100, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist", max_bin=256, enable_categorical=True,
              nthread=8, verbosity=0)
FINAL_SEEDS = [42, 7, 2026]


def dm(X, y=None):
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
    data = P.load_raw()
    if mode == "valid":
        import lightgbm as lgb
        res = []
        for fi, (cut, a, b) in enumerate(T.FOLDS):
            t0 = time.time()
            tr, va, feats = T.fold_data(data, cut, a, b)
            Xtr = T.to_lgb(tr, feats)
            Xva = T.to_lgb(va, feats, Xtr)
            y = va.y.to_numpy()
            dtr, dva = dm(Xtr, tr.y.to_numpy()), dm(Xva, y)
            m = xgb.train({**PARAMS, "seed": 42}, dtr, 4000, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
            px = m.predict(dva, iteration_range=(0, m.best_iteration + 1)).clip(0)
            lg = lgb.train({**T.BASE, **T.GRID[0]}, lgb.Dataset(Xtr, tr.y.to_numpy()), [961, 1049][fi])
            pl = lg.predict(Xva).clip(0)
            print(f"fold{fi} XGB {T.rmsle_log(y, px):.4f} (반복 {m.best_iteration}) LGBM(127) {T.rmsle_log(y, pl):.4f} 거리 {np.sqrt(np.mean((px - pl) ** 2)):.4f} ({time.time() - t0:.0f}초)", flush=True)
            for w in (0.2, 0.3, 0.4, 0.5):
                print(f"   XGB 비중 {w}: {T.rmsle_log(y, (1 - w) * pl + w * px):.4f}", flush=True)
            res.append(m.best_iteration)
        print("반복 수:", res)
    else:
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 900
        tr, te, feats = P.make_dataset(data=data)
        Xtr = T.to_lgb(tr, feats)
        Xte = T.to_lgb(te, feats, Xtr)
        preds = []
        for sd in FINAL_SEEDS:
            m = xgb.train({**PARAMS, "seed": sd}, dm(Xtr, tr.y.to_numpy()), rounds)
            preds.append(m.predict(dm(Xte)))
            print(f"시드 {sd} 완료", flush=True)
        p = np.mean(preds, axis=0).clip(0)
        sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(p).clip(0)})
        order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
        assert len(sub) == 157_194 and (sub.example_id.to_numpy() == order.to_numpy()).all() and np.isfinite(sub.boardings).all()
        sub.to_csv("outputs/xgb_v9.csv", index=False)
        print("저장: outputs/xgb_v9.csv")


if __name__ == "__main__":
    main()
