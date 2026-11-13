"""검단 신설 3개 역(검단호수공원·신검단중앙·아라)과 계양의 7~10월 수준 보정 (팀원 분석 메모 `team_inputs/team_geomdan_note.md` 기반 재구현).

팀원의 원본 스크립트(geomdan_adjust.py)는 전달받지 못해 메모의 설명대로 다시 만들었다. 아래 가정이 들어 있다.
- 목표(평일 하루 평균 승차) = 2026-06 실제 × 성숙역 계절 × 입주 세대 증가 (탄력성 0.5, 검단 3개 역만; 계양은 세대 효과 없음)
- 성숙역 계절(2025년 6월 대비 log): 7월 -0.069, 8월 -0.084, 9월 -0.004, 10월 +0.004 (메모 값)
- 세대 증가율 = 6월 말 이후 신규 입주 세대(월 안에서 시간 평균) / 17,500 (메모의 "10월까지 6월 대비 +30% 안팎"에 맞춘 분모. 입주 파일 자체는 6월 말까지 9,933세대)
- 보정 = 0.5 × (log 목표 − log 예측의 평일 하루 평균) 을 그 역·그 달의 모든 행(log1p)에 더한다
사용법: python src/geomdan_adjust.py 입력.csv 출력.csv
"""
import sys

import numpy as np
import pandas as pd

SEASON = {7: -0.069, 8: -0.084, 9: -0.004, 10: 0.004}
STATIONS = {3107: True, 3108: True, 3109: True, 3110: False}   # True = 입주 세대 효과 적용
BASE_HOUSEHOLDS = 17_500.0
ELASTICITY, SHRINK = 0.5, 0.5


def household_growth(month_start, month_end):
    h = pd.read_csv("data/external/geomdan_movein_team.csv", parse_dates=["start_date"])
    h = h[h.start_date > "2026-06-30"]
    days = pd.date_range(month_start, month_end)
    cum = np.array([h[h.start_date <= d].households.sum() for d in days]).mean()
    return cum / BASE_HOUSEHOLDS


def adjust(sub, test, train, cal):
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"])
    k["lp"] = np.log1p(k.boardings)
    k["off"] = k.date.map(cal.is_offday)
    tr = train[(train.date >= "2026-06-01") & (train.date <= "2026-06-30")]
    tr = tr.assign(off=tr.date.map(cal.is_offday))
    info = {}
    for sid, use_h in STATIONS.items():
        d = tr[(tr.station_id == sid) & (tr.off == 0)].groupby("date").boardings.sum().mean()
        for m, s in SEASON.items():
            ms, me = pd.Timestamp(2026, m, 1), pd.Timestamp(2026, m, 1) + pd.offsets.MonthEnd(0)
            g = household_growth(ms, me) if use_h else 0.0
            tgt = d * np.exp(s) * (1 + g) ** ELASTICITY
            sel = (k.station_id == sid) & (k.date.dt.month == m)
            pred = k[sel & (k.off == 0)].assign(b=np.expm1(k.lp)).groupby("date").b.sum().mean()
            delta = SHRINK * np.log(tgt / pred)
            k.loc[sel, "lp"] += delta
            info[(sid, m)] = (round(float(pred)), round(float(tgt)), round(float(delta), 3))
    k["boardings"] = np.expm1(k.lp).clip(lower=0)
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left"), info


def main():
    te = pd.read_csv("data/raw/test.csv")
    tr = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    cal = pd.read_csv("data/derived/calendar_features.csv", parse_dates=["date"]).set_index("date")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    out, info = adjust(sub, te, tr, cal)
    assert out.boardings.notna().all() and len(out) == len(sub)
    out.to_csv(sys.argv[2], index=False)
    for key, (p, t, d) in info.items():
        print(f"역 {key[0]} {key[1]}월 예측 평일평균 {p} → 목표 {t}, 보정 {d:+.3f}")


if __name__ == "__main__":
    main()
