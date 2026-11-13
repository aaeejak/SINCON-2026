"""기준선 위 잔차 LightGBM (SINCON 2026) — 다양성을 위한 보조 모델.

기준선 = 예측 시작 직전 8주의 (역, 요일, 시간) log 평균. 이 값을 피처(base_log)로 넣고,
타깃을 y - base_log (기준선 대비 잔차)로 학습한다. 기존 LightGBM은 전 기간 평균 + 시간 흐름으로 수준을 잡고,
이 모델은 '가장 최근 수준'에 붙어서 예측하므로 오차 방향이 다르다.

학습 행의 기준선은 누수 없이 만든다: 2개월 블록마다, 블록 시작 직전(거리 0~2개월)과
2개월 더 앞(거리 2~4개월)을 기준으로 한 두 벌을 쌓는다. 예측 대상(7~10월)은 cutoff 기준 0~4개월 거리다.
gap_days(기준 시점과의 거리)를 피처로 준다.

사용법 (프로젝트 루트에서):
    python src/train_residual.py valid   # 3~6월 구간 검증 (학습 ~2026-02-28)
    python src/train_residual.py final   # 최종 학습 (~2026-06-30) → outputs/submission_residual.csv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

WEEKS = 8
PARAMS = {**T.BASE, **T.GRID[0]}


def base_for(rows, hist, origin_end):
    """rows(역, daytype, hour)에 origin_end 직전 8주 (역, 요일, 시간) log 평균을 붙인다. 없으면 (역, 시간) → 전체 평균 순."""
    win = hist[(hist.date > origin_end - pd.Timedelta(weeks=WEEKS)) & (hist.date <= origin_end)]
    keys = ["station_id", "daytype", "hour"]
    p1 = win.groupby(keys).y.mean().rename("b1")
    p2 = win.groupby(["station_id", "hour"]).y.mean().rename("b2")
    p3 = hist[hist.date <= origin_end].groupby(keys).y.mean().rename("b3")
    out = rows[keys].join(p1, on=keys).join(p2, on=["station_id", "hour"]).join(p3, on=keys)
    return out.b1.fillna(out.b2).fillna(out.b3).fillna(0.0).to_numpy()


def stack_train(hist, first_date):
    """2개월 블록 × (거리 0~2개월, 2~4개월) 두 벌의 학습 행."""
    parts = []
    t0 = first_date + pd.Timedelta(weeks=WEEKS)
    start = pd.Timestamp(t0.year, t0.month, 1)
    if start < t0:
        start += pd.offsets.MonthBegin(1)
    last = hist.date.max()
    while start <= last:
        end = min(start + pd.offsets.MonthBegin(2) - pd.Timedelta(days=1), last)
        rows = hist[(hist.date >= start) & (hist.date <= end)]
        for lag in (0, 2):
            origin = start - pd.Timedelta(days=1) - pd.DateOffset(months=lag)
            if origin - pd.Timedelta(weeks=WEEKS) < first_date - pd.Timedelta(days=1):
                continue
            r = rows.copy()
            r["base_log"] = base_for(r, hist, origin)
            r["gap_days"] = (r.date - origin).dt.days
            parts.append(r)
        start = start + pd.offsets.MonthBegin(2)
    return pd.concat(parts, ignore_index=True)


def prepare(data, cut, a=None, b=None):
    hist, target, feats = P.make_dataset(cut, a, b, data=data)
    first = hist.date.min()
    tr = stack_train(hist, first)
    origin = pd.Timestamp(cut)
    target = target.copy()
    target["base_log"] = base_for(target, hist, origin)
    target["gap_days"] = (target.date - origin).dt.days
    feats = feats + ["base_log", "gap_days"]
    return tr, target, feats


def fit(tr, feats, rounds=None, va=None, seed=42):
    Xtr = T.to_lgb(tr, feats)
    ytr = (tr.y - tr.base_log).to_numpy()
    prm = {**PARAMS, "seed": seed}
    if va is not None:
        Xva = T.to_lgb(va, feats, Xtr)
        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, (va.y - va.base_log).to_numpy(), reference=dtr)
        m = lgb.train(prm, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, Xtr
    return lgb.train(prm, lgb.Dataset(Xtr, ytr), rounds), Xtr


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
    data = P.load_raw()
    if mode == "valid":
        cut, a, b = T.FOLDS[1]
        tr, va, feats = prepare(data, cut, a, b)
        ok = data[2].set_index("station_id").first_date
        ok = ok[ok <= pd.Timestamp(cut) - pd.Timedelta(weeks=4)].index
        va = va[va.station_id.isin(ok)].reset_index(drop=True)
        print(f"학습 행 {len(tr)}, 검증 행 {len(va)}, 피처 {len(feats)}", flush=True)
        m, Xtr = fit(tr, feats, va=va)
        Xva = T.to_lgb(va, feats, Xtr)
        p = (m.predict(Xva, num_iteration=m.best_iteration) + va.base_log.to_numpy()).clip(0)
        y = va.y.to_numpy()
        print("잔차 모델 단독", round(T.rmsle_log(y, p), 4), "반복", m.best_iteration, flush=True)
        # 기존 LightGBM(같은 구간) 예측과 섞기
        trb, vab, fb = T.fold_data(data, cut, a, b)
        Xb = T.to_lgb(trb, fb)
        mb = lgb.train(PARAMS, lgb.Dataset(Xb, trb.y.to_numpy()), 1049)
        pb = mb.predict(T.to_lgb(vab, fb, Xb)).clip(0)
        assert len(pb) == len(p)
        print("기존 LightGBM 단독", round(T.rmsle_log(y, pb), 4), "| 두 예측 거리",
              round(float(np.sqrt(np.mean((p - pb) ** 2))), 4))
        for w in (0.1, 0.2, 0.3, 0.4, 0.5):
            print(f"  잔차 모델 비중 {w}: {T.rmsle_log(y, (1 - w) * pb + w * p):.4f}", flush=True)
    else:
        tr, te, feats = prepare(data, "2026-06-30")
        print(f"학습 행 {len(tr)}, 피처 {len(feats)}", flush=True)
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        preds = []
        for sd in (42, 7):
            m, Xtr = fit(tr, feats, rounds=rounds, seed=sd)
            preds.append(m.predict(T.to_lgb(te, feats, Xtr)) + te.base_log.to_numpy())
        p = np.mean(preds, axis=0).clip(0)
        sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(p).clip(0)})
        order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
        assert len(sub) == 157_194 and (sub.example_id.to_numpy() == order.to_numpy()).all()
        assert np.isfinite(sub.boardings).all()
        sub.to_csv("outputs/submission_residual.csv", index=False)
        print("저장: outputs/submission_residual.csv")


if __name__ == "__main__":
    main()
