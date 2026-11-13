"""data/derived/calendar_features.csv 생성 (2025-01-01 ~ 2026-10-31, 하루 1행).

데이터 기준 (2026-11-11 변경): 7월 이후 외부 데이터(행사·날씨·학사일정)도 사용 가능.
2026-07 이후 지하철 승하차 실적만 사용 금지. 출처는 data/external/SOURCES.md 참고.

입력 (data/external/):
    events_master.csv (src/merge_team_data.py, 검증된 행사),
    school_vacation_daily.csv (src/fetch_school_vacation.py), weather_hourly.csv (src/fetch_weather.py),
    gumdan_population_monthly.csv (src/fetch_population.py)

사용법 (프로젝트 루트에서):
    python src/build_calendar.py [--ext data/external]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendar_data import (HOLIDAYS, LABOR, LUNAR_MAIN, LUNAR_SUB, MAJOR,  # noqa: E402
                           PARENTS_DAYS, SEASONAL_PERIODS)

# 대학 학기 (학기 중 = 1): 인천대학교 공식 학사일정 (개강 ~ 보강주간 마지막 날)
# https://www.inu.ac.kr/inu/651/subview.do
UNIV_SEMESTERS = [
    ("2025-03-04", "2025-06-23"), ("2025-09-01", "2025-12-19"),
    ("2026-03-03", "2026-06-22"), ("2026-09-01", "2026-12-21"),
]


def offday_features(cal):
    off = cal.is_offday.to_numpy()
    run, i = [0] * len(off), 0
    while i < len(off):
        if off[i]:
            j = i
            while j < len(off) and off[j]:
                j += 1
            run[i:j] = [j - i] * (j - i)
            i = j
        else:
            i += 1
    cal["offday_run_len"] = run
    nxt = cal.is_offday.shift(-1, fill_value=0)
    prv = cal.is_offday.shift(1, fill_value=0)
    nrun = cal.offday_run_len.shift(-1, fill_value=0)
    prun = cal.offday_run_len.shift(1, fill_value=0)
    work = cal.is_offday == 0
    cal["is_bridge"] = (work & (nxt == 1) & (prv == 1)).astype(int)
    cal["before_long_offday"] = (work & (nrun >= 3)).astype(int)
    cal["after_long_offday"] = (work & (prun >= 3)).astype(int)


def school_vacation(cal, ext):
    """초·중·고 방학 비율. NEIS에 2024학년도(2025-01~02) 기록이 없어 그 기간은
    요일을 맞춘 1년 뒤(+364일) 값으로 채우고 vac_estimated=1로 표시한다."""
    path = ext / "school_vacation_daily.csv"
    cols = ["vac_elem", "vac_middle", "vac_high", "vac_all"]
    if not path.exists():
        print("주의: school_vacation_daily.csv 없음 → 방학 피처 생략")
        return
    v = pd.read_csv(path, parse_dates=["date"]).set_index("date")[cols]
    cal[cols] = v.reindex(cal.date).to_numpy()
    est = cal.date.between("2025-01-01", "2025-03-03")
    proxy = v.reindex(cal.loc[est, "date"] + pd.Timedelta(days=364)).to_numpy()
    cal.loc[est, cols] = proxy
    cal["vac_estimated"] = est.astype(int)
    cal[cols] = cal[cols].fillna(0.0)


def daily_weather(cal, ext):
    path = ext / "weather_hourly.csv"
    if not path.exists():
        print("주의: weather_hourly.csv 없음 → 날씨 피처 생략")
        return
    w = pd.read_csv(path, parse_dates=["date"])
    day = w.groupby("date").agg(precip_day_mm=("precip_mm", "sum"),
                                temp_max_c=("temp_c", "max"),
                                temp_min_c=("temp_c", "min"),
                                snow_day_cm=("snow_cm", "sum"))
    # 운행 시간대(06~23시) 강수
    day["precip_service_mm"] = w[w.hour.between(6, 23)].groupby("date").precip_mm.sum()
    for c in day:
        cal[c] = cal.date.map(day[c]).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", type=Path, default=Path("data/external"))
    args = ap.parse_args()
    ext = args.ext

    cal = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-10-31")})
    dow = cal.date.dt.day_of_week
    key = cal.date.dt.strftime("%Y-%m-%d")
    cal["dow"] = dow
    cal["holiday_name"] = key.map(HOLIDAYS).fillna("")
    cal["is_holiday"] = (cal.holiday_name != "").astype(int)
    cal["is_major_holiday"] = cal.holiday_name.isin(MAJOR).astype(int)
    cal["is_offday"] = ((dow >= 5) | (cal.is_holiday == 1)).astype(int)
    offday_features(cal)

    # 팀 달력 열 (holiday_type: 0 평일, 1 일반 공휴일, 2 근로자의날·노동절, 3 설·추석 전후일, 4 설·추석 당일)
    cal["holiday_type"] = np.select(
        [key.isin(LUNAR_MAIN), key.isin(LUNAR_SUB), key.isin(LABOR), cal.is_holiday == 1], [4, 3, 2, 1], 0)
    cal["is_parents_day"] = key.isin(PARENTS_DAYS).astype(int)
    cal["eff_weekday"] = np.where(cal.is_holiday == 1, 6, dow)  # 공휴일은 일요일로 취급
    cal["eff_is_wknd"] = (cal.eff_weekday >= 5).astype(int)
    md = cal.date.dt.strftime("%m-%d")
    cal["seasonal_period_code"] = 0
    for s, e, code in SEASONAL_PERIODS:
        cal.loc[(md >= s) & (md <= e), "seasonal_period_code"] = code
    assert not cal.loc[key == "2026-09-28", "is_holiday"].any(), "2026-09-28은 평일이어야 함"

    cal["univ_semester"] = 0
    for s, e in UNIV_SEMESTERS:
        cal.loc[cal.date.between(s, e), "univ_semester"] = 1

    school_vacation(cal, ext)

    # 펜타포트 (송도달빛축제공원), SSG 홈경기 (문학): 검증된 행사 목록(events_master.csv)에서
    ev = pd.read_csv(ext / "events_master.csv", parse_dates=["date"])
    cal["pentaport"] = cal.date.isin(ev[ev.ev_key == "pentaport"].date).astype(int)
    regular = ev.ev_key.str.startswith("bb") & ~ev.ev_key.isin(["bb_pre", "bb_cancel"])  # 정규시즌 경기만
    bb = ev[regular].groupby("date").ev_start.min()
    cal["ssg_home_game"] = cal.date.isin(bb.index).astype(int)
    cal["ssg_start_hour"] = cal.date.map(bb).fillna(-1).astype(float).astype(int)  # 18.5시 → 18

    daily_weather(cal, ext)

    # 검단 지역 월별 인구 (경계 변경 보정값, src/fetch_population.py)
    pop_path = ext / "gumdan_population_monthly.csv"
    if pop_path.exists():
        pop = pd.read_csv(pop_path)
        pop = pop[pop.dong == "검단합계"].set_index("month").population_adj
        cal["gumdan_pop_adj"] = cal.date.dt.strftime("%Y-%m").map(pop)

    out = Path("data/derived/calendar_features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.assign(date=cal.date.dt.strftime("%Y-%m-%d")).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(cal)}행, {cal.shape[1]}열)")


if __name__ == "__main__":
    main()
