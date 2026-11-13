"""앵커 스태커 (SINCON 2026) — 작년 같은 요일 앵커로 모델 예측을 보정하는 2단계 모델.

1) fold2(학습 ~2026-02-28, 예측 3~6월)에서 LightGBM 단일 예측 pb를 만들고, 같은 행의 앵커 두 개를 붙인다.
     앵커1 = 364·371·357일 전 (0.5/0.25/0.25), 앵커2 = 7개 lag 평균 (364 ±7·14·21일, 가중 0.3/0.2/0.2/0.1/0.1/0.05/0.05)
   공휴일 구분이 어긋나거나, 올해와 작년의 행사 유무가 다르거나, 작년 날이 이상치 목록에 있으면 그 lag는 버린다.
   역별 성장분(최근 8주 대 1년 전 8주)을 더한다. 1년 전 기록이 없는 역은 NaN.
2) 보정 모델: 타깃 = y - pb, 피처 = [앵커1-pb, 앵커2-pb, hour, daytype, event_any, 앵커유무, te_st_dt_h, pb], 작은 LightGBM(300라운드).
3) 예측 대상(7~10월)에서는 기준 예측(base_log)에 보정 모델이 낸 값을 더한다 (shrink 비율 곱).

검증(3~6월, 시간 순서): 앞 달로 학습해 뒤 달 평가 — 5~6월 0.1620→0.1578, 4월 0.1929→0.1834, 6월 0.1614→0.1531.
이득 중 일부(5~6월 약 0.005)는 앵커 없이 수준·시간 피처만으로도 얻어져, 최근 달 라벨로 모델 편향을 고친 효과일 수 있다.
앵커가 있어야만 생기는 이득은 4월(0.1929→0.1834 vs 앵커 없이 0.1916)이다.

사용법: python src/stack_anchor.py <기준 예측 csv> <출력 csv> [shrink=0.7]
        (기준 예측 = example_id, boardings 형식의 제출 후보 파일)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

LAGS_A = [(364, .5), (371, .25), (357, .25)]
LAGS_B = [(364, .3), (371, .2), (357, .2), (378, .1), (350, .1), (385, .05), (343, .05)]
COLS = ["d1", "d4", "hour", "daytype", "event_any", "ok1", "te", "lvl"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20, verbose=-1, seed=1)
ROUNDS = 300


def anchors(train, rows, cutoff, lag_sets):
    cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
    ev = pd.read_csv("data/derived/station_events.csv", parse_dates=["date"])
    evset = set(zip(ev[ev.source == "event"].station_id, ev[ev.source == "event"].date))
    od = pd.read_csv("data/derived/outlier_days.csv", parse_dates=["date"])
    odset = set(zip(od.station_id, od.date))
    h = train[train.date <= cutoff].copy()
    h["y"] = np.log1p(h.boardings)
    key = h.set_index(["station_id", "date", "hour"]).y
    cd, W, off = pd.Timestamp(cutoff), pd.Timedelta(weeks=8), pd.DateOffset(years=1)
    lvl = lambda s, e: h[(h.date > s) & (h.date <= e)].groupby("station_id").y.mean()
    gs = rows.station_id.map((lvl(cd - W, cd) - lvl(cd - W - off, cd - off)).dropna()).to_numpy()
    e_now = np.array([(s, x) in evset for s, x in zip(rows.station_id, rows.date)])
    out = []
    for lags in lag_sets:
        parts, wts = [], []
        for days, w in lags:
            d = rows.date - pd.Timedelta(days=days)
            v = key.reindex(pd.MultiIndex.from_arrays([rows.station_id, d, rows.hour])).to_numpy()
            same = ((cal.holiday_type.reindex(rows.date).fillna(0).to_numpy() == cal.holiday_type.reindex(d).fillna(0).to_numpy())
                    & (cal.is_offday.reindex(rows.date).to_numpy() == cal.is_offday.reindex(d).to_numpy()))
            same &= np.array([(s, x) in evset for s, x in zip(rows.station_id, d)]) == e_now
            same &= ~np.array([(s, x) in odset for s, x in zip(rows.station_id, d)])
            parts.append(np.where(same, v, np.nan))
            wts.append(w)
        L, w = np.stack(parts, 1), np.array(wts)
        m = ~np.isnan(L)
        out.append(np.where(m.any(1), np.nansum(L * w, 1) / np.maximum((m * w).sum(1), 1e-9), np.nan) + gs)
    return out


def feats(df, base, a1, a4):
    return pd.DataFrame({"d1": a1 - base, "d4": a4 - base, "hour": df.hour.to_numpy(), "daytype": df.daytype.to_numpy(),
                         "event_any": df.event_any.to_numpy(), "ok1": (~np.isnan(a1)).astype(int),
                         "te": df.te_st_dt_h.to_numpy(), "lvl": base})


def main():
    base_csv, out_csv = sys.argv[1], sys.argv[2]
    shrink = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    data = P.load_raw()
    train = data[0]
    # 1) fold2 기준 예측과 앵커
    cut, a, b = T.FOLDS[1]
    tr, va, fe = T.fold_data(data, cut, a, b)
    Xtr = T.to_lgb(tr, fe)
    m = lgb.train({**T.BASE, **T.GRID[0]}, lgb.Dataset(Xtr, tr.y.to_numpy()), 1049)
    pb = m.predict(T.to_lgb(va, fe, Xtr)).clip(0)
    a1, a4 = anchors(train, va, cut, [LAGS_A, LAGS_B])
    F = feats(va, pb, a1, a4)
    mdl = lgb.train(PARAMS, lgb.Dataset(F[COLS], va.y.to_numpy() - pb), ROUNDS)
    # 2) 예측 대상에 적용
    _, te, _ = P.make_dataset(data=data)
    order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
    base = np.log1p(pd.read_csv(base_csv).set_index("example_id").boardings.reindex(te.example_id).to_numpy())
    b1, b4 = anchors(train, te, "2026-06-30", [LAGS_A, LAGS_B])
    corr = mdl.predict(feats(te, base, b1, b4)[COLS])
    final = np.expm1((base + shrink * corr).clip(0)).clip(0)
    sub = pd.DataFrame({"example_id": te.example_id, "boardings": final})
    assert len(sub) == 157_194 and (sub.example_id.to_numpy() == order.to_numpy()).all() and np.isfinite(final).all()
    sub.to_csv(out_csv, index=False)
    print(f"보정 평균 {corr.mean():.4f}, 표준편차 {corr.std():.4f}, 분위수(1/50/99%) {np.round(np.quantile(corr, [.01, .5, .99]), 3)}, "
          f"앵커 가용 {np.isfinite(b1).mean():.3f}")
    print(f"저장: {out_csv} (shrink {shrink})")


if __name__ == "__main__":
    main()
