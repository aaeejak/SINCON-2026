"""작년 같은 요일 값 기준(앵커) 예측 (SINCON 2026) — 모델과 완전히 다른 방식의 보조 예측.

예측 대상 행(역, 날짜, 시간)마다 364일 전(같은 요일)과 ±7일(371, 357일 전)의 같은 역·같은 시간 승차를
0.5:0.25:0.25로 log 공간에서 평균하고, 역별 최근 8주 대 1년 전 같은 8주의 성장분을 더한다.
공휴일 구분이 어긋나는 날(예: 올해는 평일인데 작년 같은 요일은 공휴일)은 해당 lag를 버린다.
1년 전 8주 기록이 없는 역(검단 신설 3개)과 lag가 없는 행은 NaN → 이 앵커를 쓰지 않는다.

검증(3~6월 구간): 앵커 단독 0.212, LightGBM과 log 공간 0.85 : 0.15 혼합 0.1681 → 0.1659.
(7~10월 모사 구간은 1년 전 기록이 없어 검증 불가.)

사용법: python src/yoy_anchor.py  → outputs/anchor_lastyear_log.npy (test.csv 순서, NaN 포함)
함수 anchor_log(train, rows, cutoff, cal)을 다른 스크립트에서 불러 쓸 수도 있다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LAGS = [(364, 0.5), (371, 0.25), (357, 0.25)]


def anchor_log(train, rows, cutoff, cal):
    h = train[train.date <= cutoff].copy()
    h["y"] = np.log1p(h.boardings)
    key = h.set_index(["station_id", "date", "hour"]).y
    parts, wts = [], []
    for days, w in LAGS:
        d = rows.date - pd.Timedelta(days=days)
        v = key.reindex(pd.MultiIndex.from_arrays([rows.station_id, d, rows.hour])).to_numpy()
        same = ((cal.holiday_type.reindex(rows.date).fillna(0).to_numpy() == cal.holiday_type.reindex(d).fillna(0).to_numpy())
                & (cal.is_offday.reindex(rows.date).to_numpy() == cal.is_offday.reindex(d).to_numpy()))
        parts.append(np.where(same, v, np.nan))
        wts.append(w)
    L, w = np.stack(parts, 1), np.array(wts)
    m = ~np.isnan(L)
    ly = np.where(m.any(1), np.nansum(L * w, 1) / np.maximum((m * w).sum(1), 1e-9), np.nan)
    # 역별 성장분: 최근 8주 vs 1년 전 같은 8주
    cd, W, off = pd.Timestamp(cutoff), pd.Timedelta(weeks=8), pd.DateOffset(years=1)
    lvl = lambda s, e: h[(h.date > s) & (h.date <= e)].groupby("station_id").y.mean()
    g = (lvl(cd - W, cd) - lvl(cd - W - off, cd - off)).dropna()
    gs = rows.station_id.map(g).to_numpy()  # 성장분을 못 구한 역은 NaN → 앵커 제외
    return ly + gs


def main():
    train = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    test = pd.read_csv("data/raw/test.csv", parse_dates=["date"])
    cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
    a = anchor_log(train, test, "2026-06-30", cal)
    np.save("outputs/anchor_lastyear_log.npy", a)
    print(f"앵커 가용 {np.isfinite(a).mean():.3f}, 평균 {np.nanmean(np.expm1(a)):.1f}")
    print("월별 가용:", pd.Series(np.isfinite(a)).groupby(test.date.dt.month.to_numpy()).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
