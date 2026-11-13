"""검바위(3208) 반복 패턴 보정 — 학교 시험일(12시)과 요일별 오후 승차.

학습 기간의 근거 (`src/exam_adjust.py` 설명 참고): 12시 +450명 평일 연속 블록이 같은 주(364일 뒤 같은 요일)에 반복된다.
추가로 확인한 것:
- 시험일에는 11시 +72, 13시 +88명도 늘고(평균), 시험 블록의 전·후 날은 정상이다 (경계가 뚜렷함).
- 15시(월·금 +400명)와 16시(화~목 +400명)에도 반복 이상이 있고, 2025 상반기 이상의 364일 뒤 재현률은 15시 7/11, 16시 4/4 (무작위 기준 0.1%).
- 2026 공휴일(제헌절 7/17, 대체 8/17, 추석 9/24~26 등)과 겹치는 날짜는 뺀다.
사용법: python src/geombawi_adjust.py 입력.csv 출력.csv <옵션...>
  hours    시험일 11시 +72, 13시 +88 (각 ×0.6)
  h1516    2025 7~10월 15·16시 이상일을 364일 뒤로 옮긴 날의 15·16시에 +100
  spread   12시 보정을 정렬일 ×0.5 + 앞뒤 하루 ×0.1 로 날짜 불확실성을 나눠 줌 (exam 보정 대체)
"""
import sys

import numpy as np
import pandas as pd

STATION = 3208
EXAM = ["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08",
        "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"]


def aligned_1516():
    tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
    g = tr[(tr.station_id == STATION) & tr.hour.isin([15, 16])].copy()
    g["off"] = g.date.map(cal.is_offday)
    g = g.sort_values("date")
    g["med"] = g.groupby(["hour", "off"]).boardings.transform(lambda s: s.rolling(29, center=True, min_periods=9).median())
    g["ratio"] = g.boardings / g.med.clip(lower=15)
    g["extra"] = g.boardings - g.med
    x = g[(g.ratio >= 2.5) & (g.extra >= 150) & g.date.between("2025-07-01", "2025-10-31")]
    out = []
    for r in x.itertuples():
        d = r.date + pd.Timedelta(days=364)
        if d > pd.Timestamp("2026-10-31") or (d in cal.index and cal.loc[d, "is_holiday"] == 1):
            continue
        out.append((d, r.hour))
    return out


def adjust(sub, test, opts):
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"])
    b = k.boardings.to_numpy(dtype=float).copy()
    st = (k.station_id == STATION).to_numpy()
    n = 0

    def add(days, hour, amount):
        nonlocal n
        m = st & (k.hour == hour).to_numpy() & k.date.isin(pd.to_datetime(list(days))).to_numpy()
        b[m] += amount
        n += int(m.sum())

    if "spread" in opts:  # 기본 보정(0.6)이 이미 들어 있으면 쓰지 않는다 (OPT5 원본에 적용)
        add(EXAM, 12, 450 * 0.5)
        nb = sorted({(pd.Timestamp(d) + pd.Timedelta(days=s)).strftime("%Y-%m-%d") for d in EXAM for s in (-1, 1)} - set(EXAM))
        cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
        nb = [d for d in nb if cal.loc[d, "is_offday"] == 0]
        add(nb, 12, 450 * 0.1)
    if "hours" in opts:
        add(EXAM, 11, 72 * 0.6)
        add(EXAM, 13, 88 * 0.6)
    if "h1516" in opts:
        for h in (15, 16):
            add([d for d, hh in aligned_1516() if hh == h], h, 100)
    k["boardings"] = b
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left"), n


def main():
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    out, n = adjust(sub, te, set(sys.argv[3:]))
    assert out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
