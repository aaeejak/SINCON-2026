# 데이터 출처·생성 기록 (최종 A 경로)

예측 기간(2026-07-01~10-31)의 실제 지하철 승하차 인원은 어떤 파일에도 없고 쓰지 않았다. 승차 인원 열이 있는 외부 파일은 `past_monthly_2023_2024.csv`(2023·2024년)와 `monthly_prior_2023_2024.csv`(같은 월보에서 계산한 계절 모양)뿐이다. 해시는 `SHA256SUMS.txt`.

## 1. 대회 제공 자료 (변형 없음, `data/raw/`에 포함)
| 파일 | SHA-256 |
|---|---|
| train.csv | 1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc |
| test.csv | d257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa |
| metaData.csv | 9870dc6c9cbe7e6fc810bbe589dd9e06f309bb173a268a1bd63478f430785380 |
| sample_submission.csv | 8c5e10d4201460ddff2d78bf10a229ec8b604e86ae2f18bcd96969093981c93d |

## 2. 외부 자료 (`data/`)
| 파일 | 내용 | 기간 | 출처 | 이용 조건 | 조회일 | 사용 |
|---|---|---|---|---|---|---|
| holidays_kr_2025_2026.csv | 공휴일·대체공휴일 | 2025-01~2026-10 | 한국천문연구원 월력요항 자료, 고용노동부 보도자료(2026-04-06), 인사혁신처 보도자료(2026-04-29) | 정부 발표 날짜 사실 | 2026-11-11 | v7·v10 피처, 후처리 |
| external/weather_incheon_2025_2026.csv | 일 강수량·강수 시간·최고·평균기온 | 2025-01~2026-10 | Open-Meteo Historical Weather API, 인천 37.456N 126.705E | CC BY 4.0 | 2026-11-11 | 피처 |
| external/rain_blocks_incheon.csv | 시간대 묶음별 강수 | 2025-01~2026-10 | 같은 API 시간별 값 집계 | CC BY 4.0 | 2026-11-11 | 피처 |
| external/kbo_munhak_2025_2026.csv | SSG 문학 홈경기 일정·시작 시각·상대 | 2025-03~2026-10 | KBO 경기일정 https://www.koreabaseball.com/Schedule/Schedule.aspx | 사이트 이용 조건(비상업 확인 필요) | 2026-11-11 | 피처, 후처리 |
| external/kbo_munhak_crowd.csv | 경기별 관중 | 2025-03~2026-10 | KBO 관중 현황 https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx | 같음 | 2026-11-11 | 피처 |
| external/ssg_munhak_game_times.csv | 경기별 실제 개시·종료 시각, 관중 | 2025-03~2026-10 | KBO 게임센터 https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx (nayun 수집) | 같음 | 2026-11-12 | SSG 종료 보정 |
| external/events_2025_2026.csv | 행사(축구·공연·축제) 날짜·역·시작 시각 | 2025-02~2026-10 | K리그 kleague.com, 인천시·연수구 보도자료, 예매처 공지 (팀 행사 시트) | 사실 정보 | 2026-11-11~12 | 피처, 후처리 |
| external/geomdan_movein.csv | 검단신도시 단지별 입주 시작일·세대 | 2024-11~2026-10 | 단지별 입주 공고·언론 보도 (블로그 제외) | 사실 정보 | 2026-11-11 | 피처 |
| external/park_3226_features.csv | 인천대공원 단풍·벚꽃·행사·물놀이장 | 2025-01~2026-10 | 인천대공원 공지, 상수(단풍 절정 2025-11-09·2026-11-01, 벚꽃 2025-04-14·2026-04-09) | 사실 정보 | 2026-11-11 | 피처 |
| external/ops_daily_features.csv | 검단연장 개통(2025-06-28)·시운전, 배차 계획 | 2025-01~2026-10 | 언론 보도(아시아경제 2025-04-09 등), metaData first_date | 사실 정보 | 2026-11-11 | 피처(개통·시운전 2열만) |
| external/ops_incident_hourly.csv | 운행 장애 기록 | 2026-04~09 | 인천교통공사 공지 | 공공 정보 | 2026-11-11 | 읽기만 하고 피처 미사용 (해당 없음) |
| external/facility_changes_2025_2026.csv | 역 주변 시설 개장·폐장 | 2025-09~2026-07 | 지자체·운영사 공지 | 사실 정보 | 2026-11-13 | 실험용, 최종 경로 미사용 (EXTRA_COLS 빈 값) |
| external/school_calendar_2025_2026.csv | 초중고 방학, 인천대 학기 | 2025-01~2026-10 | NEIS 학사일정, 인천대 학사일정 | 공공 정보 | 2026-11-11 | 인천대 학기 1열만 피처 |
| external/transfer_events_2025_2026.csv | 환승역 행사 | 2025-02~2026-10 | 팀 행사 시트 | 사실 정보 | 2026-11-11 | 읽기만 하고 피처 미사용 (USE_XFER=False) |
| external/past_monthly_2023_2024.csv | 역별 월 승차 합계 | 2023·2024 | 인천교통공사 수송 통계(월보) | 공공 자료 | 2026-11-11 | 피처(예측 기간 이전) |
| external/monthly_prior_2023_2024.csv | 역x시간대 월별 계절 모양 | 2023·2024 | 위 월보에서 계산 | 공공 자료 | 2026-11-11 | v7 피처 mp_sh·mp_st |
| external_geombawi/calendar_features.csv | 날짜별 달력·공휴일 | 2025-01~2026-10 | nayun `build_calendar.py`로 생성 (코드 사본 `code/geombawi_nayun/`) | - | 2026-11-13 | 검바위 보정 |
| external_geombawi/events_master.csv | 팀 행사 목록 통합본 | 2023~2026 | nayun `merge_team_data.py` (nayun 패키지) | 사실 정보 | 2026-11-13 | 검바위 보정 |

취소 행사 근거 (코드 안 상수): OKF 2026 연기·취소 iMBC 연예 2026-09-15 https://enews.imbc.com/News/ViewAmp/518304 · GHOSTIVAL 2026 전면 취소 아시아투데이 2026-10-21 https://www.asiatoday.co.kr/kn/view.php?key=20261021010007097 (조회 2026-11-13)

## 3. 수집했지만 쓰지 않은 자료 (해당 없음)
| 파일 | 내용 | 이유 |
|---|---|---|
| bus500_selected.csv | 역 500m 버스 월별 이용 (2026-07~09 포함, 지하철 아님) | 검증 B에서 효과가 약해 미사용. 코드에서 읽지 않음 |
| ts_stats_station*.csv | 역x시간 시계열 통계 | 분석용. 코드에서 읽지 않음 |
| 초중고 방학(vac_*) | 학사일정 | 검증에서 이득 없어 끔 |

## 4. 파생 데이터·중간 산출물 (`artifacts/`)
| 파일 | 만든 사람 | 생성 방법 | 재현 |
|---|---|---|---|
| models/v7.csv | kdbwls | `code/sincon_pipeline_v7.py --mode submit` | 코드 (`run_all.py --retrain`으로 비교) |
| models/ps15.csv | kdbwls | `code/scripts/run_per_station_final.py ps15` (파이프라인 v10, 역별 LightGBM) | 코드 |
| models/ps15_lvl.csv | kdbwls | ps15 + (OPT4_raw log 평균 − ps15 log 평균) = log −0.021095 | 코드 |
| models/new_v5data.csv | kdbwls (다른 대화에서 작성) | `feat_v5a_package` (`code/feat_v5a_package.zip`) README의 `train_final_kfold.py ... 3 0.1 0.1` + 성장률 +0.0081 | 코드 (약 40분) |
| models/F_v10_stack.csv, models/nayun_v3.csv | nayun | nayun 패키지 `src/make_F_v10.py`, `src/reproduce_v3.py` | nayun 코드 |
| models/dh_v2.csv, models/dh_mlp.csv | Dohyeong | Dohyeong `train.py`, `train_mlp_ensemble.py` | Dohyeong 코드 (팀 패키지 확인 필요) |
| blend_chain/OPT4_raw.csv, OPT5_raw.csv | kdbwls | 위 모델들의 log1p 가중 평균. 비율은 Public 점수 C 행렬(`code/scripts/blend_public*.py`)로 한 번 계산. 최종 비율을 저장한 스크립트가 없어 실제 파일로 제출 | 파일 |
| blend_chain/OPT5_pp.csv | kdbwls | `full_pp.py` | 코드, 차이 0 |
| blend_chain/team_blend_opt4_75_v6pp_25.csv | nayun | OPT4_final_geo 0.75 + nayun v6 후처리본 0.25 (log1p) | nayun 코드 |
| blend_chain/OPT5pp_team50.csv | kdbwls | OPT5_pp 0.5 + 위 파일 0.5 | 코드, 차이 0 |
| blend_chain/OPT5pp_team50_fix.csv | kdbwls | 반반 섞기로 절반이 된 문학 SSG 종료·OKF 보정을 원래 세기로 복원 (243행). 당시 바로 계산해 스크립트 없음 | 파일 |
| blend_chain/OPT5pp_team50_fix_cancel.csv | kdbwls | `cancel_adjust.py ... 1.0` | 코드, 차이 0 |
| blend_chain/submission_OPT4_final_geo*.csv | kdbwls | OKF 보정량 계산용 (cancel_adjust 입력) | 파일 |
