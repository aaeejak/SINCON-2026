"""v3 LightGBM 재현 (Kaggle Public 0.16087, 팀 블렌드의 'nayun lgbm' 성분).

v3 = 피처 42개, 이상치 제외 없음, 단일 모델(255리프, min_data 200, lambda_l2 5.0), 시드 42, 1091 라운드, 베이스라인 혼합 0.
이후 개선(행사 종료 시각·인접역 피처, 이상치 제외, 환승 열, 날씨 확장, 3설정×시드3 평균)은 모두 끈다.
cv_lgbm_v3.json의 features 목록과 같은 피처만 쓴다.

사용법: python src/reproduce_v3.py [출력 csv]   (기본 outputs/submission_lgbm_v3_repro.csv)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

V3_FEATURES = ['station_id', 'hour', 'line', 'multi_line', 'days_since_open', 'time_idx', 'dow', 'is_holiday',
               'is_major_holiday', 'is_offday', 'offday_run_len', 'is_bridge', 'before_long_offday',
               'after_long_offday', 'holiday_type', 'is_parents_day', 'seasonal_period_code', 'univ_semester',
               'vac_elem', 'vac_middle', 'vac_high', 'vac_all', 'precip_day_mm', 'temp_max_c', 'temp_min_c',
               'snow_day_cm', 'precip_service_mm', 'gumdan_pop_adj', 'month', 'precip_mm', 'precip_3h_mm', 'temp_c',
               'event_any', 'event_start_hour', 'event_type', 'hours_from_event', 'daytype', 'te_st_dt_h',
               'te_st_off_h', 'te_st_h', 'te_st', 'alight_ratio']
PARAMS = {**T.BASE, "num_leaves": 255, "min_data_in_leaf": 200, "lambda_l2": 5.0, "seed": 42}
ROUNDS = 1091


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/submission_lgbm_v3_repro.csv"
    P.USE_WX_EXTRA = False
    data = P.load_raw()
    tr, te, feats = P.make_dataset(data=data, clean=False)
    feats = [f for f in V3_FEATURES if f in feats]
    assert len(feats) == 42, len(feats)
    Xtr = T.to_lgb(tr, feats)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, tr.y.to_numpy()), ROUNDS)
    p = np.expm1(m.predict(T.to_lgb(te, feats, Xtr)).clip(0)).clip(0)
    sub = pd.DataFrame({"example_id": te.example_id, "boardings": p})
    order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
    assert len(sub) == 157_194 and (sub.example_id.to_numpy() == order.to_numpy()).all()
    sub.to_csv(out, index=False)
    print("저장:", out)


if __name__ == "__main__":
    main()
