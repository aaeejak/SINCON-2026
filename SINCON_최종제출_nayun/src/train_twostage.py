"""일 합계 × 시간대 비율로 나눠 예측하는 LightGBM (SINCON 2026) — 보조 모델.

1단계(수준): 역·날짜의 하루 평균 log 승차 L = mean_h y 를 예측 (날짜 단위 피처, 행 수 약 1/18).
2단계(모양): 시간대 편차 y - L 을 예측 (시간 단위 피처 전부).
예측 = L_hat + shape_hat. 단일 모델과 같은 정보를 쓰지만 분해해서 학습하므로 오차가 다르다.

사용법 (프로젝트 루트에서):
    python src/train_twostage.py valid   # 두 검증 구간, 혼합 이득
    python src/train_twostage.py final   # 최종 학습 → outputs/submission_twostage.csv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

HOURLY = {"hour", "hours_from_event", "hours_from_end", "adj_hours_from_end", "precip_mm",
          "precip_3h_mm", "temp_c", "te_st_dt_h", "te_st_off_h", "te_st_h", "alight_ratio"}
PARAMS = {**T.BASE, **T.GRID[0]}


def day_frame(df, feats):
    """역·날짜 단위로 집계: 시간 피처는 평균(날씨·평균 통계) 또는 제외."""
    g = df.groupby(["station_id", "date"])
    catc = ["station_id", "line", "daytype", "month", "event_type", "adj_event_type"]
    cols = [c for c in feats if c not in ("hour", "hours_from_event", "hours_from_end", "adj_hours_from_end") + tuple(catc)]
    d = g[cols].mean().reset_index()
    d["event_any"] = (d.event_any > 0).astype(int)
    first = g[[c for c in catc if c != "station_id"]].first().reset_index(drop=True)
    for c in first:
        d[c] = first[c].to_numpy()
    d["precip_max"] = g.precip_mm.max().to_numpy()
    d["temp_max"] = g.temp_c.max().to_numpy()
    if "y" in df:
        d["L"] = g.y.mean().to_numpy()
    return d, [c for c in feats if c in d.columns] + ["precip_max", "temp_max"]


def run(tr, te, feats, va_y=None, rounds=(None, None), seed=42):
    """tr/te: 시간 단위 프레임. 반환: te 행 순서의 예측 log1p."""
    dtr, dcols = day_frame(tr, feats)
    dte, _ = day_frame(te, feats)
    prm = {**PARAMS, "seed": seed}
    # 1단계
    Xd = T.to_lgb(dtr, dcols)
    Xdt = T.to_lgb(dte, dcols, Xd)
    if rounds[0] is None:
        dv = None
        if va_y is not None:
            dv = lgb.Dataset(Xdt, dte.L.to_numpy(), reference=lgb.Dataset(Xd, dtr.L.to_numpy()))
        m1 = lgb.train(prm, lgb.Dataset(Xd, dtr.L.to_numpy()), 4000, valid_sets=[dv],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        r1 = m1.best_iteration
    else:
        m1 = lgb.train(prm, lgb.Dataset(Xd, dtr.L.to_numpy()), rounds[0])
        r1 = rounds[0]
    dte["L_hat"] = m1.predict(Xdt, num_iteration=r1)
    # 2단계: 편차
    tr = tr.merge(dtr[["station_id", "date", "L"]], on=["station_id", "date"], how="left")
    Xs = T.to_lgb(tr, feats)
    ys = (tr.y - tr.L).to_numpy()
    te2 = te.merge(dte[["station_id", "date", "L_hat"]], on=["station_id", "date"], how="left")
    Xst = T.to_lgb(te2, feats, Xs)
    if rounds[1] is None:
        te_y = (te2.y - te2.groupby(["station_id", "date"]).y.transform("mean")).to_numpy()
        dtr2 = lgb.Dataset(Xs, ys)
        m2 = lgb.train(prm, dtr2, 4000, valid_sets=[lgb.Dataset(Xst, te_y, reference=dtr2)],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        r2 = m2.best_iteration
    else:
        m2 = lgb.train(prm, lgb.Dataset(Xs, ys), rounds[1])
        r2 = rounds[1]
    shape = m2.predict(Xst, num_iteration=r2)
    return (te2.L_hat.to_numpy() + shape).clip(0), te2, (r1, r2)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
    data = P.load_raw()
    if mode == "valid":
        for fi, (cut, a, b) in enumerate(T.FOLDS):
            tr, va, feats = T.fold_data(data, cut, a, b)
            p, va2, its = run(tr, va, feats, va_y=True)
            y = va2.y.to_numpy()
            Xtr = T.to_lgb(tr, feats)
            pb = lgb.train(PARAMS, lgb.Dataset(Xtr, tr.y.to_numpy()), [961, 1049][fi]).predict(T.to_lgb(va2, feats, Xtr)).clip(0)
            print(f"fold{fi} 2단계 {T.rmsle_log(y, p):.4f} 단일 {T.rmsle_log(y, pb):.4f} 거리 {np.sqrt(np.mean((p - pb) ** 2)):.4f} 반복 {its}", flush=True)
            for w in (0.2, 0.3, 0.4, 0.5):
                print(f"   2단계 비중 {w}: {T.rmsle_log(y, (1 - w) * pb + w * p):.4f}", flush=True)
    else:
        tr, te, feats = P.make_dataset(data=data)
        p, te2, _ = run(tr, te, feats, rounds=(int(sys.argv[2]) if len(sys.argv) > 2 else 800,
                                                int(sys.argv[3]) if len(sys.argv) > 3 else 1000))
        sub = pd.DataFrame({"example_id": te2.example_id, "boardings": np.expm1(p).clip(0)})
        order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
        assert len(sub) == 157_194 and (sub.example_id.to_numpy() == order.to_numpy()).all(), "행 순서 불일치"
        assert np.isfinite(sub.boardings).all()
        sub.to_csv("outputs/submission_twostage.csv", index=False)
        print("저장: outputs/submission_twostage.csv")


if __name__ == "__main__":
    main()
