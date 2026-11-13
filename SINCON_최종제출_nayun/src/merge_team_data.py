"""팀원이 모은 외부 데이터를 검증하면서 data/external/ 로 합친다 (SINCON 2026).

입력 폴더(--src)에 있어야 하는 것
    events_master.csv, station_master.csv, SINCON_데이터_CSV/ (02~05 폴더)

검증 원칙
    1. 과거(2026-06-30까지) 행사는 실제 승차량으로 열렸는지 확인한다 (저녁·오후 승차가 평소의 몇 배인지).
    2. SSG 홈경기는 KBO 공식 일정(data/external/ssg_home_games_2025_2026.csv)과 대조한다.
    3. 부천FC 리그 경기는 K리그 공식 일정(data/external/bucheon_fc_home_games.csv)과 대조한다.
    4. 예측 기간 행사는 출처(공식 일정, 예매처, 주요 언론)가 있는 것만 남긴다. 출처 URL은 행마다 기록한다.
    5. 과거 승차 자료는 2023·2024년만 가져온다 (2025년은 대회 train.csv와 같아서 중복, 2026년은 사용 금지).
    6. 관중 수는 K리그 공식 기록만 'official', 나머지는 추정치라 'estimate'로 표시한다.

결과 (data/external/)
    events_master.csv                  검증된 행사 (모델에 사용)
    events_reference_unverified.csv    2차 출처·영향 미검증 행사/장애 (모델에 사용하지 않음)
    events_2023_2024_reference.csv     2023~2024 행사 (참고용)
    station_master.csv                 역 정보 (노선, 개통일, 환승)
    past_ridership_2023_2024_monthly.csv   인천교통공사 월보 7~10월 시간대별 승차 (2023·2024)
    past_ridership_correction_2025.csv     2025 train ÷ 월보 보정비율

사용법 (프로젝트 루트에서):
    python src/merge_team_data.py --src "C:/Users/linha/Downloads"
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/external")
CUTOFF = pd.Timestamp("2026-06-30")

# 직접 확인한 출처 (예매 페이지, 공식 기사). 키워드는 행사명에 들어 있는 글자.
URLS = {
    ("PLAVE", 2026): "https://nol.yanolja.com/ticket/products/26008189",
    ("GHOSTIVAL", 2026): "https://nol.yanolja.com/ticket/products/26014180",
    ("Sustainable Wave", 2026): "https://sports.khan.co.kr/article/202607062240003",
    ("싸이흠뻑쇼", 2026): "https://nol.yanolja.com/ticket/products/26007871",
    ("송도맥주축제", 2026): "https://nol.yanolja.com/ticket/products/26010853",
    ("이슬라이브", 2026): "https://nol.yanolja.com/ticket/products/26010722",
    ("펜타포트", 2026): "https://www.hankyung.com/article/202603051090H",
    ("펜타포트", 2025): "https://www.incheon.go.kr/ml_kr/MLKR020401/3033581",
    ("코리아컵 3R", 2026): "https://www.mt.co.kr/sports/2026/07/29/202607291843774934O",
    ("오픈트레이닝", 2026): "https://www.newsis.com/view/NISX20260804_0003736434",
}
KBO_URL = "https://www.koreabaseball.com/Schedule/Schedule.aspx"
KLEAGUE_URL = "https://www.kleague.com/schedule.do"


def read(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def surge_ratio(train, st, d, other_event_days):
    """당일 저녁(20~23시)·오후(16~19시) 승차가 같은 역·같은 요일 구분의 ±4주 중앙값의 몇 배인지 (큰 쪽)."""
    x = train[train.station_id == st]
    best = []
    for hrs in ([20, 21, 22, 23], [16, 17, 18, 19]):
        s = x[x.hour.isin(hrs)].groupby("date").boardings.sum()
        if d not in s.index:
            return np.nan
        wk = d.day_of_week >= 5
        pool = s[(abs((s.index - d).days) <= 28) & ((s.index.day_of_week >= 5) == wk)
                 & (~s.index.isin(other_event_days)) & (s.index != d)]
        best.append(s[d] / pool.median() if len(pool) >= 3 else np.nan)
    return float(np.nanmax(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    args = ap.parse_args()
    src, pkg = args.src, args.src / "SINCON_데이터_CSV"
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    meta = pd.read_csv("data/raw/metaData.csv")

    # ---------------------------------------------------------------- 행사
    ev = read(src / "events_master.csv")
    ev["date"] = pd.to_datetime(ev["date"])
    n0 = len(ev)

    # 04 폴더의 행별 출처 (date, station_id 기준)
    f04 = read(glob.glob(str(pkg / "04_*" / "전체_2023-2026.csv"))[0])
    c = list(f04.columns)
    f04 = f04.rename(columns={c[0]: "date", c[6]: "station_id", c[12]: "src04", c[7]: "att04", c[2]: "kind04"})
    f04["date"] = pd.to_datetime(f04["date"])
    src04 = f04.drop_duplicates(["date", "station_id"]).set_index(["date", "station_id"]).src04

    # SSG, 부천FC 공식 일정
    ssg = pd.read_csv("data/external/ssg_home_games_2025_2026.csv", parse_dates=["date"])
    ssg_ok = set(ssg[ssg.played == 1].date)
    fc_ok = set(pd.read_csv("data/external/bucheon_fc_home_games.csv", parse_dates=["date"]).date)

    by_station = ev.groupby("station_id").date.apply(set).to_dict()
    rows = []
    for r in ev.itertuples():
        past = r.date <= CUTOFF
        key, name = r.ev_key, r.event_name
        source, verif, att_basis, ratio = "", "", "", np.nan
        if key.startswith("bb"):
            source = KBO_URL
            verif = "KBO 공식 일정과 일치" if r.date in ssg_ok else "KBO 일정에 없음"
        else:
            if past:
                ratio = surge_ratio(train, r.station_id, r.date, by_station.get(r.station_id, set()))
                verif = f"승차량 확인 (저녁·오후 승차 평소의 {ratio:.1f}배)"
            for (kw, yr), url in URLS.items():
                if kw in name and r.date.year == yr:
                    source = url
                    verif = (verif + " / " if verif else "") + "예매처·언론 확인"
            league = name.startswith("부천FC vs") and "코리아컵" not in name
            if league and r.date in fc_ok:
                source, verif = KLEAGUE_URL, (verif + " / " if verif else "") + "K리그 공식 일정과 일치"
            if not source and (r.date, r.station_id) in src04.index:
                source = str(src04[(r.date, r.station_id)])
        if key.startswith("fc") and name.startswith("부천FC vs") and "코리아컵" not in name:
            att_basis = "official" if pd.notna(r.ev_att) else ""
        elif pd.notna(r.ev_att):
            att_basis = "estimate"
        rows.append(dict(date=r.date.strftime("%Y-%m-%d"), station_id=r.station_id, venue=r.venue,
                         event_name=name, ev_key=key, ev_start=r.ev_start,
                         ev_att=r.ev_att, ev_att_basis=att_basis, source=source,
                         verification=verif, surge_ratio=round(ratio, 2) if pd.notna(ratio) else np.nan))
    out = pd.DataFrame(rows)

    # KBO 공식 일정에서 추가 (2026-11-12 오차 분석으로 발견): 문학경기장 시범경기와 취소 경기.
    # 시범경기는 정규시즌 목록에 없어 훈련 데이터의 설명 안 되는 급증(예: 2026-03-22)이었고,
    # 취소 경기일에는 경기 직전 취소되면 18~19시·16~17시 승차가 몰린다 (2025-06-13, 09-06 등).
    # 같은 날 다른 경기가 열린 날(2025-05-10)은 정규 경기 행으로 이미 표시되어 있어 뺀다.
    pre = pd.read_csv(OUT / "ssg_preseason_2025_2026.csv", parse_dates=["date"])
    canc = ssg[(ssg.played == 0) & (~ssg.date.isin(ssg_ok))]
    extra = [(r.date, "SSG랜더스 시범경기", "bb_pre", r.start_time, "KBO 공식 일정(시범경기)")
             for r in pre.itertuples()]
    extra += [(r.date, f"SSG랜더스 경기 취소({r.note})", "bb_cancel", r.start_time, "KBO 공식 일정(취소)")
              for r in canc.itertuples()]
    add = []
    for d, name, key, st, verif in extra:
        ratio = surge_ratio(train, 3127, d, by_station.get(3127, set())) if d <= CUTOFF else np.nan
        add.append(dict(date=d.strftime("%Y-%m-%d"), station_id=3127, venue="문학경기장", event_name=name,
                        ev_key=key, ev_start=int(st[:2]) + int(st[3:5]) / 60, ev_att=np.nan, ev_att_basis="",
                        source=KBO_URL,
                        verification=verif + (f" / 승차량 확인 (평소의 {ratio:.1f}배)" if pd.notna(ratio) else ""),
                        surge_ratio=round(ratio, 2) if pd.notna(ratio) else np.nan))
    out = pd.concat([out, pd.DataFrame(add)], ignore_index=True)
    print(f"[KBO 추가] 시범경기 {len(pre)}일, 취소 경기 {len(canc)}일")

    # 행사 종료 시각 (공연·페스티벌은 경기보다 늦게 끝나 저녁 승차 시각이 다르다).
    # known: 예매처·공식 자료의 운영 시간, estimate: 종류별 평균 길이로 추정
    def end_hour(r):
        n, k, s = r.event_name, r.ev_key, r.ev_start
        if "GHOSTIVAL" in n:
            return 22.0, "known"            # KOPIS·예매처 15:00~22:00
        if "펜타포트" in n or "송도맥주축제" in n:
            return 22.0, "known"            # 11:00~22:00, 평일 16~22시·주말 15~22시 운영
        if "이슬라이브" in n:
            return s + 7.5, "known"         # 예매 페이지: 공연 시간 450분(인터미션 포함)
        if "Sustainable Wave" in n:
            return 22.0, "estimate"         # 종료 시각 미확인, 페스티벌 하루 일정으로 추정
        if k == "bb_cancel":
            return s + 0.5, "estimate"      # 취소 결정은 대개 경기 시작 무렵
        if k.startswith("bb"):
            return s + 3.2, "estimate"      # 야구 평균 경기 시간 + 퇴장
        if k.startswith("fc"):
            return s + 2.3, "estimate"      # 축구 경기 + 퇴장
        if k == "concert":
            return s + 3.5, "estimate"
        return s + 4.0, "estimate"

    ends = out.apply(end_hour, axis=1, result_type="expand")
    out["ev_end"], out["ev_end_basis"] = ends[0].round(2), ends[1]

    # 검증 실패 행 제외 (공식 일정에 없는 SSG 경기, 출처·확인 근거가 없는 행)
    bad = out[(out.verification == "KBO 일정에 없음") | (out.verification == "") | (out.source == "")]
    keep = out.drop(bad.index)
    keep = keep.sort_values(["date", "station_id", "ev_key"]).reset_index(drop=True)
    keep.to_csv(OUT / "events_master.csv", index=False, encoding="utf-8-sig")
    print(f"[행사] 입력 {n0}행 → 검증 통과 {len(keep)}행, 제외 {len(bad)}행")
    if len(bad):
        print(bad[["date", "event_name", "verification", "source"]].to_string(index=False))
    weak = keep[keep.surge_ratio.notna() & (keep.surge_ratio < 1.5)]
    print(f"  승차량 서지가 약한 행사(1.5배 미만) {len(weak)}건: 기록은 유지, verification 열에 배수 표시")

    # 참고용 (2차 출처, 영향 미검증): 02 폴더 행사·장애 목록 중 경기·공휴일 제외
    f02 = read(glob.glob(str(pkg / "02_*" / "행사·장애_목록.csv"))[0])
    k = list(f02.columns)
    ref = f02[~f02[k[2]].isin(["경기", "공휴일"])].copy()
    ref = ref.rename(columns={k[0]: "start_date", k[1]: "end_date", k[2]: "kind", k[3]: "content",
                              k[4]: "place", k[5]: "stations", k[6]: "train_evidence", k[7]: "source"})
    ref["note"] = "2차 출처·영향 미검증. 모델 피처에 쓰지 않음"
    ref.to_csv(OUT / "events_reference_unverified.csv", index=False, encoding="utf-8-sig")
    print(f"[참고용] 행사·장애 {len(ref)}건 → events_reference_unverified.csv (모델 제외)")

    # 2023~2024 행사 (참고)
    old = f04[f04.date < "2025-01-01"].rename(columns={c[3]: "event_name", c[4]: "start_time", c[5]: "venue"})
    old[["date", "station_id", "kind04", "event_name", "start_time", "venue", "att04", "src04"]].rename(
        columns={"kind04": "kind", "att04": "attendance", "src04": "source"}
    ).to_csv(OUT / "events_2023_2024_reference.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- 역 정보
    sm = read(src / "station_master.csv")
    ok_ids = set(sm.station_id) == set(meta.station_id)
    names = (meta.set_index("station_id").station_name == sm.set_index("station_id").station_name).all()
    lines = (meta.set_index("station_id").canonical_line == sm.set_index("station_id").canonical_line).all()
    assert ok_ids and names and lines, "station_master가 metaData와 맞지 않음"
    sm.to_csv(OUT / "station_master.csv", index=False, encoding="utf-8-sig")
    print(f"[역 정보] {len(sm)}개 역, metaData와 ID·이름·노선 일치, 환승역 {int(sm.is_transfer.sum())}개")

    # ---------------------------------------------------------------- 과거 승차 (2023~2024)
    pr = read(glob.glob(str(pkg / "03_*" / "월별_*"))[0])
    p = list(pr.columns)
    hrs = p[7:-1]
    assert len(hrs) == 18 and (pr[hrs].sum(axis=1) - pr[p[-1]]).abs().max() < 1
    # 2025행은 train 합산과 같은지 확인한 뒤 버린다 (대회 데이터 중복)
    t = train[(train.date >= "2025-07-01") & (train.date <= "2025-10-31")].copy()
    t["m"] = t.date.dt.month
    tt = t.groupby(["station_id", "m", "hour"]).boardings.sum().unstack("hour")
    tt.columns = [f"{h:02d}시" for h in tt.columns]
    x = pr[pr[p[0]] == 2025].set_index([p[3], p[1]])[hrs]
    tt.index.names = [p[3], p[1]]
    joined = x.join(tt, rsuffix="_tr")
    diff = joined[hrs].values - joined[[h + "_tr" for h in hrs]].values
    assert (np.abs(diff) < 0.5).all(), "2025 값이 train 합산과 다름"
    assert not (pr[p[0]] >= 2026).any(), "2026년 승차 자료가 들어 있음 (사용 금지)"
    old_r = pr[pr[p[0]].isin([2023, 2024])]
    old_r.to_csv(OUT / "past_ridership_2023_2024_monthly.csv", index=False, encoding="utf-8-sig")
    read(glob.glob(str(pkg / "03_*" / "보정비율*"))[0]).to_csv(
        OUT / "past_ridership_correction_2025.csv", index=False, encoding="utf-8-sig")
    print(f"[과거 승차] 2023·2024 {len(old_r)}행 저장 (2025년 {int((pr[p[0]] == 2025).sum())}행은 train과 같아 제외, 2026년 없음)")


if __name__ == "__main__":
    main()
