"""역·날짜별 행사 표 data/derived/station_events.csv를 만든다.

행 종류 (source 열)
- event : 검증된 행사 (data/external/events_master.csv). SSG 홈경기, 부천FC 경기, 공연, 축제 등.
- spike : train에서 19시 이후 승차가 앞뒤 2주 중앙값의 2배 이상인 경기장 역의 날.
          정답(승차량)으로 만든 표시라서 **검증 구간 피처로 쓰면 점수가 부풀려진다.**
          행사 목록에 없는 이상치를 찾아 학습 데이터를 점검하는 용도로만 쓴다 (2026-06-30까지만 존재).

열: date, station_id, source, event, ev_key, ev_start (시작 시각, 시간 단위 소수), start_hour (정수, -1 = 모름)

사용법 (프로젝트 루트에서):
    python src/build_event_days.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path("data/external")
SPIKE_STATIONS = [3127, 3209, 3754, 3759, 3139]
RATIO = 2.0


def find_outliers(ev):
    """행사 목록·달력으로 설명되지 않는 급증 역·날짜를 찾는다 (학습에서 제외할 이상치).

    기준: 같은 역·같은 시간·같은 요일 구분(평일/휴일)의 ±28일 중앙값 대비 3배 이상이고 시간당 +400명 이상.
    설·추석 연휴(holiday_type 3, 4)는 피처로 설명되므로 제외하지 않는다.
    train 승차량으로 만든 목록이라 피처로 쓰면 안 되고, 학습 데이터 정리에만 쓴다 (2026-06-30까지).
    """
    tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
    tr["off"] = tr.date.map(cal.is_offday)
    tr = tr.sort_values("date")
    med = tr.groupby(["station_id", "hour", "off"]).boardings.transform(
        lambda s: s.rolling(57, center=True, min_periods=15).median())
    tr["ratio"] = tr.boardings / med.clip(lower=20)
    tr["inc"] = tr.boardings - med
    cand = tr[(tr.ratio >= 3) & (tr.inc >= 400)]
    lunar = cal.index[cal.holiday_type.isin([3, 4])]
    cand = cand[~cand.date.isin(lunar)]
    day = cand.groupby(["station_id", "date"]).agg(max_ratio=("ratio", "max"), inc_sum=("inc", "sum")).reset_index()
    listed = {(r.station_id, pd.Timestamp(r.date)) for r in ev.itertuples()}  # 날짜 형식을 맞춰 비교
    day = day[[(s, d) not in listed for s, d in zip(day.station_id, day.date)]]
    return day.assign(date=day.date.dt.strftime("%Y-%m-%d")).round(1)


def main():
    ev = pd.read_csv(EXT / "events_master.csv")
    parts = [pd.DataFrame({
        "date": ev.date, "station_id": ev.station_id, "source": "event", "event": ev.event_name,
        "ev_key": ev.ev_key, "ev_start": ev.ev_start, "ev_end": ev.ev_end, "ev_att": ev.ev_att,
        "start_hour": np.floor(ev.ev_start.fillna(-1)).astype(int),
    })]

    tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    tr = tr[tr.station_id.isin(SPIKE_STATIONS) & (tr.hour >= 19)]
    master_days = ev.groupby("station_id").date.apply(set).to_dict()
    for st in SPIKE_STATIONS:
        d = tr[tr.station_id == st].groupby("date").boardings.sum()
        r = (d / d.rolling(29, center=True, min_periods=10).median()).dropna()
        days = [x for x in r[r >= RATIO].index if x.strftime("%Y-%m-%d") not in master_days.get(st, set())]
        parts.append(pd.DataFrame({"date": [x.strftime("%Y-%m-%d") for x in days], "station_id": st,
                                   "source": "spike", "event": "", "ev_key": "", "ev_start": np.nan,
                                   "ev_end": np.nan, "ev_att": np.nan, "start_hour": -1}))

    out = pd.concat(parts, ignore_index=True).sort_values(["date", "station_id", "source"])
    path = Path("data/derived/station_events.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} ({len(out)}행)", out.source.value_counts().to_dict(),
          "| spike는 행사 목록에 없는 이상치만 남김")

    # 학습에서 뺄 이상치 역·날짜: 일반 탐지 결과 + 경기장 역 spike
    od = find_outliers(ev)
    sp = out[out.source == "spike"][["station_id", "date"]].assign(max_ratio=np.nan, inc_sum=np.nan)
    od = pd.concat([od, sp]).drop_duplicates(["station_id", "date"]).sort_values(["date", "station_id"])
    od.to_csv("data/derived/outlier_days.csv", index=False, encoding="utf-8-sig")
    print(f"저장: data/derived/outlier_days.csv ({len(od)}개 역·날짜, {od.station_id.nunique()}개 역)")


if __name__ == "__main__":
    main()
