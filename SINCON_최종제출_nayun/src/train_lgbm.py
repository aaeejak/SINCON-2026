"""LightGBM 학습·튜닝·최종 예측 (SINCON 2026).

1. 설정 후보마다 두 검증 구간에서 학습해 RMSLE와 최적 반복 수를 잰다 (early stopping).
2. 가장 좋은 설정에 베이스라인 예측을 log 공간에서 섞는 비중을 고른다.
3. 2026-06-30까지 전체로 다시 학습해 test를 예측하고 제출 파일을 저장한다.

결과:
    outputs/cv_lgbm.json            설정별 검증 점수, 선택된 설정
    outputs/submission_lgbm.csv     제출 파일 (157,194행)

사용법 (프로젝트 루트에서):
    python src/train_lgbm.py
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as B  # noqa: E402
import preprocess as P  # noqa: E402

SEED = 42
N_TEST_ROWS = 157_194
FOLDS = [  # (학습 마지막 날, 검증 시작, 검증 끝) — 베이스라인과 같은 구간
    ("2025-06-30", "2025-07-01", "2025-10-31"),
    ("2026-02-28", "2026-03-01", "2026-06-30"),
]
BASE = dict(objective="regression", learning_rate=0.05, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1, num_threads=0)
# 이전 실험(6개 설정)에서 min_data_in_leaf 200 계열이 일관되게 좋아 그 계열 3개만 비교한다
GRID = [
    dict(num_leaves=127, min_data_in_leaf=200, lambda_l2=5.0),
    dict(num_leaves=255, min_data_in_leaf=200, lambda_l2=5.0),
    dict(num_leaves=511, min_data_in_leaf=200, lambda_l2=10.0),
]
FINAL_SEEDS = [42, 7, 2026]
BLEND_W = [0.0, 0.1, 0.2, 0.3, 0.4]  # 베이스라인 비중
BASELINE_CFG = dict(weeks=8, stat="logmean", mode="7", seasonal=True)


def rmsle_log(y_log, p_log):
    return float(np.sqrt(np.mean((y_log - p_log) ** 2)))


def to_lgb(df, feats, cats_from=None):
    X = df[feats].copy()
    for c in P.CATEGORICAL:
        cats = cats_from[c].cat.categories if cats_from is not None else sorted(X[c].unique())
        X[c] = pd.Categorical(X[c], categories=cats)
    return X


def fold_data(data, cut, a, b):
    tr, va, feats = P.make_dataset(cut, a, b, data=data)
    meta = data[2]
    first = meta.set_index("station_id").first_date
    ok = first[first <= pd.Timestamp(cut) - pd.Timedelta(weeks=4)].index  # 베이스라인과 같은 조건
    va = va[va.station_id.isin(ok)].reset_index(drop=True)
    return tr, va, feats


def baseline_log(train_raw, target):
    p = B.predict(train_raw, target[["date", "station_id", "hour"]].reset_index(drop=True), **BASELINE_CFG)
    return np.log1p(p)


def main():
    t0 = time.time()
    data = P.load_raw()
    train_raw = data[0]
    folds = []
    for cut, a, b in FOLDS:
        tr, va, feats = fold_data(data, cut, a, b)
        Xtr = to_lgb(tr, feats)
        folds.append(dict(cut=cut, Xtr=Xtr, ytr=tr.y.to_numpy(), Xva=to_lgb(va, feats, Xtr),
                          yva=va.y.to_numpy(),
                          base=baseline_log(train_raw[train_raw.date <= cut], va)))
    print(f"검증 데이터 준비 {time.time() - t0:.0f}초, 피처 {len(feats)}개", flush=True)

    # 1) 설정 비교
    results = []
    for g in GRID:
        params = {**BASE, **g}
        scores, iters, preds = [], [], []
        for f in folds:
            dtr = lgb.Dataset(f["Xtr"], f["ytr"])
            dva = lgb.Dataset(f["Xva"], f["yva"], reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            p = m.predict(f["Xva"], num_iteration=m.best_iteration).clip(0)
            scores.append(rmsle_log(f["yva"], p))
            iters.append(m.best_iteration)
            preds.append(p)
        results.append(dict(params=g, rmsle=float(np.mean(scores)), folds=scores, iters=iters, preds=preds))
        print(f"{g} → RMSLE {np.mean(scores):.4f} {np.round(scores, 4)} 반복 {iters}", flush=True)

    best = min(results, key=lambda r: r["rmsle"])

    # 2) 베이스라인 앙상블 비중
    blend = []
    for w in BLEND_W:
        s = [rmsle_log(f["yva"], (1 - w) * p + w * f["base"]) for f, p in zip(folds, best["preds"])]
        blend.append(dict(w=w, rmsle=float(np.mean(s)), folds=s))
        print(f"베이스라인 비중 {w:.1f} → RMSLE {np.mean(s):.4f} {np.round(s, 4)}")
    best_w = min(blend, key=lambda r: r["rmsle"])
    ens = [rmsle_log(f["yva"], np.mean([r["preds"][i] for r in results], axis=0)) for i, f in enumerate(folds)]
    print(f"3설정 평균 검증 RMSLE {np.mean(ens):.4f} {np.round(ens, 4)} (최종 예측에 사용)", flush=True)
    best_w = dict(best_w, rmsle=float(np.mean(ens)), folds=ens) if best_w["w"] == 0.0 else best_w

    # 3) 최종 학습 (~2026-06-30) → test 예측
    # 검증에서 3개 설정 평균이 단일 최고 설정보다 좋았다(0.1826 vs 0.1831) → 3설정 × 시드 3개를 평균
    tr, te, feats = P.make_dataset(data=data)
    Xtr = to_lgb(tr, feats)
    Xte = to_lgb(te, feats, Xtr)
    preds = []
    model = None
    for r in results:
        n_rounds_r = int(np.mean(r["iters"]) * 1.1)  # 학습 데이터가 늘어난 만큼 조금 더
        for sd in FINAL_SEEDS:  # 시드 여러 개 평균 (log 공간): 분산을 줄인다
            model_i = lgb.train({**BASE, **r["params"], "seed": sd}, lgb.Dataset(Xtr, tr.y.to_numpy()),
                                num_boost_round=n_rounds_r)
            preds.append(model_i.predict(Xte))
            if model is None:
                model = model_i
            print(f"최종 모델 {r['params']['num_leaves']}리프 시드 {sd} 완료 ({time.time() - t0:.0f}초)", flush=True)
    n_rounds = int(np.mean(best["iters"]) * 1.1)
    p_lgb = np.mean(preds, axis=0).clip(0)
    p_base = baseline_log(train_raw, te)
    p = (1 - best_w["w"]) * p_lgb + best_w["w"] * p_base

    sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(p).clip(0)})
    assert len(sub) == N_TEST_ROWS and sub.example_id.is_unique
    assert np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
    assert (sub.example_id.to_numpy() == order.to_numpy()).all()

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    sub.to_csv(out / "submission_lgbm.csv", index=False)
    if Path("/kaggle/working").exists():  # Kaggle 노트북: 제출용 위치에도 저장
        sub.to_csv("/kaggle/working/submission.csv", index=False)
    imp = pd.Series(model.feature_importance("gain"), feats).sort_values(ascending=False)
    summary = dict(
        features=feats,
        grid=[{k: v for k, v in r.items() if k != "preds"} for r in results],
        best_params={**BASE, **best["params"]}, n_rounds=n_rounds,
        blend=blend, blend_weight=best_w["w"], cv_rmsle=best_w["rmsle"],
        top_features=imp.head(15).round(0).to_dict(), seconds=round(time.time() - t0),
    )
    (out / "cv_lgbm.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n선택: {best['params']}, 반복 {n_rounds}, 베이스라인 비중 {best_w['w']}"
          f" → 검증 RMSLE {best_w['rmsle']:.4f}")
    print(f"저장: outputs/submission_lgbm.csv ({len(sub)}행), outputs/cv_lgbm.json, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
