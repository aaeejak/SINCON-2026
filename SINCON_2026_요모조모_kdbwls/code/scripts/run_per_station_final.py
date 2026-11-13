"""역별 LightGBM 최종 예측 (2026-06-30까지 학습). 사용: python scripts/run_per_station_final.py ps15"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import sincon_pipeline_v10 as P
full, te = P.load("data", "holidays_kr_2025_2026.csv"); tp = full[full.example_id.isna()]
RUNS = {"ps15": (dict(num_leaves=15, min_data_in_leaf=30), 600), "ps7": (dict(num_leaves=7, min_data_in_leaf=30), 900)}
for name in sys.argv[1:]:
    prm, n = RUNS[name]; P.PER_STATION = True; P.PER_STATION_PARAMS = {**P.PER_STATION_PARAMS, **prm}; P.PER_STATION_ROUNDS = n
    fut = full[full.example_id.notna()].copy()
    fut, preds, _ = P.predict_period(tp.copy(), fut, pd.Timestamp("2026-06-30"), n, verbose=False)
    p = np.expm1(np.clip(preds["v6_final"], 0, None))
    sub = te[["example_id"]].merge(pd.DataFrame({"example_id": fut.example_id.values, "boardings": p}), on="example_id", how="left")
    assert len(sub) == 157194 and sub.boardings.notna().all() and (sub.boardings >= 0).all()
    sub.to_csv(f"send/model_{name}.csv", index=False); print("저장", name, flush=True)
