"""반복되는 역·시간 급증 찾기 (검바위 패턴을 찾은 방법) — 다른 역에도 쓸 수 있다.

1) 역·시간·요일구분(평일/휴일)별 ±14일 중앙값 대비 배율 >= 2.5, 증가 >= 150명인 (역, 날짜, 시간)을 '이상'으로 본다 (행사 목록 날짜·설/추석 제외).
2) 2025 상반기 이상이 364일 뒤(같은 요일) 같은 역·시간에 다시 나타나는 비율(재현률)을 센다. 무작위로 맞을 확률은 약 0.1%.
3) 역·시간별로 이상 건수·재현 건수·날짜를 표로 낸다 → data/derived 아래가 아니라 현재 폴더의 spike_candidates.csv.
검바위(3208): 12시(평일 연속 시험일 블록, +450명), 15시(월·금), 16시(화~목)가 재현률이 높았다. 부천종합운동장 3~4월 행사는 7~10월과 무관.
사용법: python code/find_recurring_spikes.py   (프로젝트 루트에 data/raw/train.csv, metaData.csv 필요)
"""
import numpy as np
import pandas as pd

meta = pd.read_csv("data/raw/metaData.csv")
name = meta.drop_duplicates("station_id").set_index("station_id").station_name
tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
ev = pd.read_csv("data/external/events_master.csv", parse_dates=["date"])
evset = set(zip(ev.station_id, ev.date))

tr["off"] = tr.date.map(cal.is_offday)
tr = tr.sort_values("date")
tr["med"] = tr.groupby(["station_id", "hour", "off"]).boardings.transform(
    lambda s: s.rolling(29, center=True, min_periods=9).median())
tr["ratio"] = tr.boardings / tr.med.clip(lower=15)
tr["extra"] = tr.boardings - tr.med
a = tr[(tr.ratio >= 2.5) & (tr.extra >= 150) & tr.med.notna()].copy()
a = a[[(s, d) not in evset for s, d in zip(a.station_id, a.date)]]
a = a[~a.date.map(cal.holiday_type).isin([3, 4])]

key = set(zip(a.station_id, a.hour, a.date))
h1 = a[a.date <= "2025-06-30"]
hit = [(r.station_id, r.hour, (r.station_id, r.hour, r.date + pd.Timedelta(days=364)) in key) for r in h1.itertuples()]
h = pd.DataFrame(hit, columns=["station_id", "hour", "hit"])
print(f"이상 {len(a)}건, 2025 상반기 이상 {len(h)}건 → 364일 뒤 재현률 {h.hit.mean():.3f} (무작위 기준 약 0.001)")

rec = h.groupby(["station_id", "hour"]).hit.agg(h1_n="size", h1_hit="sum")
allc = a.groupby(["station_id", "hour"]).agg(total=("date", "size"), mean_extra=("extra", "mean"),
                                             jul_oct_2025=("date", lambda s: int(s.between("2025-07-01", "2025-10-31").sum())))
out = allc.join(rec).fillna({"h1_n": 0, "h1_hit": 0}).reset_index()
out["station"] = out.station_id.map(name)
out = out.sort_values(["h1_hit", "total"], ascending=False)
out.round(1).to_csv("spike_candidates.csv", index=False, encoding="utf-8-sig")
print(out.head(12).round(1).to_string(index=False))
