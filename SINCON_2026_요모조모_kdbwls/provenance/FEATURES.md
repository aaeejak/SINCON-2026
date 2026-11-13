# 특징 설명 (pipeline v7·v10, 모델 공통 60개 + 기준 수준)
목표: log1p(승차) − base_tr (학습 구간 역·시간·요일유형 평균 log1p 승차). 예측 = 모델 출력 + base.
참조 기간 표기: [학습] = 2026-06-30까지 train.csv, [외부] = 외부 자료(아래 출처).

| 특징 | 의미 | 원천·계산 | 참조 기간 |
|---|---|---|---|
| station_id, line, line_seq | 역 번호(범주), 노선, 노선 내 순번 | metaData.csv | - |
| hour, dow, month, doy | 시간대, 요일, 월, 연중 일 | 날짜 | - |
| day_type | 0 평일 / 1 토 / 2 일·공휴일 | 날짜 + 공휴일 목록 | - |
| is_holiday, is_myeongjeol, is_labor_day_2025, near_myeongjeol | 공휴일, 설·추석 연휴, 2025 노동절, 명절 전후 | 공휴일 목록(인사혁신처) | 2025-01~2026-10 |
| is_before_offday, is_after_offday, is_bridge_day, offday_block_len, offday_block_pos | 쉬는 날 앞뒤, 징검다리, 연휴 길이·순서 | 공휴일 목록 + 요일 | - |
| precip_mm, precip_hours, tmax, rain_now, rain_prev, rain_06_09, rain_17_20 | 일 강수량·강수 시간·최고기온, 해당·직전 시간 비, 시간대별 비 | Open-Meteo 인천 시간별 기상 (CC BY 4.0) | 2025-01~2026-10 |
| munhak_sched, munhak_game, munhak_ngames, munhak_start, munhak_rel_hour, munhak_crowd_log, munhak_opp_pop | SSG 문학 홈경기 일정·개최·시작시각·경기 기준 상대 시간·관중(log)·상대팀 인기도 | KBO 일정·관중. 인기도 = 학습 기간 경기일 문학경기장역 하루 승차 편차의 상대팀별 평균(축소) | 일정·관중 2025-03~2026-10, 인기도 [학습] |
| ev_kind, ev_start, ev_rel_hour | 행사 종류(축구·공연·축제)·시작 시각·시작 기준 상대 시간 | 행사 일정 (external/events_2025_2026.csv) | 2025-02~2026-10 |
| geomdan_hh_log | 검단신도시 누적 입주 세대(log) | 단지별 입주 시작일, 60일 선형 입주 가정 | 2024-11~2026-10 |
| park_* (7개) | 인천대공원역 단풍·벚꽃 가중, 행사 규모·진행·퇴장, 물놀이장 운영 | 인천대공원 공지·상수 | 2025-01~2026-10 |
| ops_l1_ext_open, ops_l1_trial_run | 1호선 검단연장 개통(2025-06-28)·시운전 | 언론 보도 | - |
| rank_*, share_board_recent, offday_weekday_ratio, board_growth_log_8w, rank_tier | 노선 내 승차 순위·비중, 쉬는 날/평일 비, 최근 8주 성장 | train.csv | [학습] 최근 8주·전체 |
| univ_semester | 인천대 학기 중 여부 | 학사 일정 | 2025~2026 |
| mp_sh, mp_st | 역x시간대·역 월별 계절 모양 (월 평균 log − 연평균) | 인천교통공사 월보 2023·2024 | 2023~2024 |
| ly_dev, ly_st_dev | 작년 같은 시기 편차 (역x시간, 역) | train.csv 2025년 | [학습] |
| base / base_tr, lvl_* | 기준 수준: 역·시간·요일유형 평균, 최근 8·26주 수준 | train.csv | [학습] |

## 수동 보정값 근거 (후처리)
README 4절 표 참조. 배율: 축제 사전값, 명절 0.85/0.5/0.6·역별 0.5(검증 A 되짚기), 공연장 0.4/0.7, 검단 0.5, SSG 종료 0.6(검증 A·B leave-one-out), 취소 행사(언론 보도).
