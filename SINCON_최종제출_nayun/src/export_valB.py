"""검증 B 예측 내보내기: 2026-02-28까지 학습 → 2026-03-01~06-30 예측 (팀 검증 기반 섞기용).

모델: LightGBM v9(3설정 평균), XGBoost v9, CatBoost v9, MLP v9 (outputs/archive/tune_ours/preds_fold1.npz에 저장된 값),
      ours_v9(0.72·LGBM+0.18·Cat+0.10·MLP), ours_tuned(0.80·LGBM+0.12·XGB+0.08·Cat), LightGBM v6(날씨 추가 전, 3설정 평균), LightGBM v3(처음 제출본 설정).
출력: share/for_team_valB/valB_nayun.csv (date, station_id, hour, actual, pred_*: 승차 인원 단위)
사용법: python src/export_valB.py   (v6·v3 재학습 때문에 약 5분)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import reproduce_v3 as V3  # noqa: E402
import train_lgbm as T  # noqa: E402

FI = 1  # T.FOLDS[1] = ("2026-02-28", "2026-03-01", "2026-06-30")


def keys_of(df):
    return df[["date", "station_id", "hour"]].reset_index(drop=True)


def main():
    data = P.load_raw()
    cut, a, b = T.FOLDS[FI]
    # v9 피처 기준 행과 저장된 예측
    P.USE_WX_EXTRA = True
    tr9, va9, f9 = T.fold_data(data, cut, a, b)
    z = np.load("outputs/archive/tune_ours/preds_fold1.npz")
    assert len(z["y"]) == len(va9) and np.allclose(z["y"], va9.y.to_numpy())
    out = keys_of(va9)
    out["actual"] = np.expm1(va9.y.to_numpy()).round(1)
    logs = {"lgbm_v9": z["lgbm"], "xgb_v9": z["xgb"], "catboost_v9": z["cat"], "mlp_v9": z["mlp"]}
    logs["ours_v9"] = 0.72 * z["lgbm"] + 0.18 * z["cat"] + 0.10 * z["mlp"]
    logs["ours_tuned"] = 0.80 * z["lgbm"] + 0.12 * z["xgb"] + 0.08 * z["cat"]
    # v6: 날씨 추가 전 피처, 3설정 평균
    P.USE_WX_EXTRA = False
    tr6, va6, f6 = T.fold_data(data, cut, a, b)
    cv6 = json.load(open("outputs/archive/cv_lgbm_v6.json", encoding="utf-8"))
    X6 = T.to_lgb(tr6, f6)
    p6 = np.mean([lgb.train({**T.BASE, **g["params"]}, lgb.Dataset(X6, tr6.y.to_numpy()), g["iters"][FI]).predict(T.to_lgb(va6, f6, X6)) for g in cv6["grid"]], axis=0)
    # v3: 이상치 제외 없음, 피처 42개, 255리프, 이 구간의 최적 반복 1030
    tr3, va3, f3 = P.make_dataset(cut, a, b, data=data, clean=False)
    first = data[2].set_index("station_id").first_date
    va3 = va3[va3.station_id.isin(first[first <= pd.Timestamp(cut) - pd.Timedelta(weeks=4)].index)].reset_index(drop=True)
    f3 = [f for f in V3.V3_FEATURES if f in f3]
    X3 = T.to_lgb(tr3, f3)
    p3 = lgb.train(V3.PARAMS, lgb.Dataset(X3, tr3.y.to_numpy()), 1030).predict(T.to_lgb(va3, f3, X3))
    for name, frame, p in (("lgbm_v6", va6, p6), ("lgbm_v3", va3, p3)):
        k = keys_of(frame).assign(**{name: p})
        out = out.merge(k, on=["date", "station_id", "hour"], how="left")
        logs[name] = out[name].to_numpy()
        out = out.drop(columns=name)
    y = va9.y.to_numpy()
    print("검증 B RMSLE:", {n: round(float(np.sqrt(np.nanmean((np.clip(v, 0, None) - y) ** 2))), 4) for n, v in logs.items()})
    for n, v in logs.items():
        out["pred_" + n] = np.expm1(np.clip(v, 0, None)).round(3)
    Path("share/for_team_valB").mkdir(parents=True, exist_ok=True)
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    out.to_csv("share/for_team_valB/valB_nayun.csv", index=False)
    print("저장 share/for_team_valB/valB_nayun.csv", out.shape, "결측", int(out.isna().sum().sum()))


if __name__ == "__main__":
    main()
