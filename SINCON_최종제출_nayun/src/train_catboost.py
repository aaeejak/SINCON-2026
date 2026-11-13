"""CatBoost 보조 모델 (LightGBM과 다른 오차를 섞기 위함).

검증(두 구간, LightGBM과 log 공간 0.8 : 0.2 혼합): 0.1958 → 0.1950, 0.1714 → 0.1707.
CatBoost 단독은 0.205 / 0.178로 LightGBM보다 나쁘다. 반복 수 800은 검증의 조기 종료 반복(195, 1427) 평균 부근.
station_id/line은 범주 처리 없이 숫자로 넣었을 때 더 좋았다.

실행: python src/train_catboost.py  → outputs/submission_catboost.csv, 그 뒤 outputs/submission_lgbm_cat.csv (LightGBM 0.8 + CatBoost 0.2)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402

W_CAT = 0.2


def main():
    data = P.load_raw()
    tr, te, feats = P.make_dataset(data=data)
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    for c in P.CATEGORICAL:
        Xtr[c], Xte[c] = Xtr[c].astype(int), Xte[c].astype(int)
    preds = []
    for sd in (42, 7):
        cb = CatBoostRegressor(loss_function="RMSE", depth=8, learning_rate=0.06, l2_leaf_reg=10,
                               iterations=800, random_seed=sd, thread_count=-1, verbose=200)
        cb.fit(Xtr, tr.y.to_numpy())
        preds.append(cb.predict(Xte))
    p_cat = np.mean(preds, axis=0).clip(0)
    out = Path("outputs")
    sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(p_cat).clip(0)})
    assert len(sub) == 157_194 and np.isfinite(sub.boardings).all()
    sub.to_csv(out / "submission_catboost.csv", index=False)

    lgb_sub = pd.read_csv(out / "submission_lgbm_v6.csv").set_index("example_id").boardings.reindex(sub.example_id)
    mix = (1 - W_CAT) * np.log1p(lgb_sub.to_numpy()) + W_CAT * p_cat
    final = pd.DataFrame({"example_id": sub.example_id, "boardings": np.expm1(mix).clip(0)})
    order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
    assert (final.example_id.to_numpy() == order.to_numpy()).all() and np.isfinite(final.boardings).all()
    final.to_csv(out / "submission_lgbm_cat.csv", index=False)
    print("저장: outputs/submission_catboost.csv, outputs/submission_lgbm_cat.csv",
          "| 두 예측 거리", round(float(np.sqrt(np.mean((np.log1p(lgb_sub.to_numpy()) - p_cat) ** 2))), 4))


if __name__ == "__main__":
    main()
