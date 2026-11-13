# SINCON 2026 — 인천 지하철 시간대별 승차 예측 (팀 프로젝트, nayun 담당 부분)

7~10월(2026-07-01~10-31) 71개 역 × 시간대(06~23시) 승차 인원 157,194행 예측, 지표 RMSLE. 마감 2026-11-13 21:00.

## 1. 최종 제출 (Kaggle 최종 선택 2개)
| 선택 | 파일 | Public |
|---|---|---|
| 1 | `FINAL/final_submission.csv` (= `cand_best_examall_jul`) | **0.14399** |
| 2 | `FINAL/final_submission_2.csv` (= `cand_gb_fc_Ftuned_v3swap`) | 0.14403 |

설명과 마감 전 체크리스트: `FINAL/README.md`. 규칙 점검: `docs/최종2개_규칙점검_2026-11-13.md`. 점수 이력: `outputs/SCORES.md`.
우리 단독 LightGBM(v3 0.16087)에서 시작해 팀 블렌드(0.1518), 팀 후처리(축제·명절·검단·취소 행사)와 우리 보정(검바위·부천FC·시험일)을 거쳐 0.14399까지 내려갔다.

### 최종 제출 재현
**한 번에:** `python src/run_final.py` (약 1분) → `outputs/repro/`에 다시 만들고 `FINAL/`과 비교, "재현 성공" 출력.

**단계 (모든 계수는 Public을 보기 전에 학습 데이터 검증이나 팀 설정에서 정함):**
| 단계 | 명령 | 입력 | 근거 |
|---|---|---|---|
| 0 | (팀원) `team_inputs/input_team_gb_v6_orig.csv` | 팀 블렌드 + 후처리, Public 0.144110 | 팀원 코드 필요 |
| 1 | `python src/fc_adjust.py team_inputs/input_team_gb_v6_orig.csv a.csv 0.7` | `data/derived/fc_end_profile.csv`, `data/external/events_master.csv`(부천FC 일정·관중) | 부천종합운동장 경기 종료 전후 시간대. 경기 단위 LOO로 RMSE 0.383→0.319 |
| 2 | `python src/misc_adjust.py a.csv FINAL/final_submission_2.csv ftuned v3swap` | `outputs/models/{lgbm_v9,xgb_v9,catboost_v9,lgbm_v3}.csv`, `outputs/archive/candidates_unscored/{cand_F_tuned,cand_F_v10_stack}.csv` | 블렌드 안 우리 몫을 검증 B 최적 혼합으로 교체(비중은 `team_inputs/team_blend_weights_OPT.json`) |
| 3 | `PYTHONPATH=src python src/exam_all_adjust.py FINAL/final_submission_2.csv FINAL/final_submission.csv 0.466 0 7` | `data/derived/exam_day_lift.csv` | 학습 데이터 시험 기간 5블록 교차 검증으로 k=0.466 |

**재료를 처음부터 다시 만들기 (선택, 오래 걸림):**
```bash
python src/build_calendar.py                      # data/derived/calendar_features.csv
python src/reproduce_v3.py                        # outputs/models/lgbm_v3.csv (약 3분)
python src/train_lgbm.py                          # lgbm_v9 (약 25분)
python src/train_xgb.py final 1300 ; python src/train_catboost.py ; python src/train_mlp.py
python src/make_F_v10.py                          # cand_F_v10_stack (D + 앵커 스태커, 약 3분) — 저장본과 같음 확인
python src/make_F_xgb.py outputs/archive/candidates_unscored/cand_F_tuned.csv tuned   # cand_F_tuned — 저장본과 같음 확인
python src/exam_all_adjust.py build               # data/derived/exam_day_lift.csv
python src/fc_adjust.py fit                       # data/derived/fc_end_profile.csv (약 4분)
```

## 2. 폴더
```
FINAL/                 최종 제출 파일과 체크리스트
src/                   코드 (아래 표)
data/raw/              대회 원본 (train/test/metaData)
data/derived/          달력·행사·이상치·SSG 보정 계수
data/external/         외부 자료 (출처·조회일: SOURCES.md)
outputs/scored/        점수가 확인된 파일 / outputs/models/ 우리 모델 예측 / outputs/archive/ 나머지
team_inputs/          팀원에게 받은 제출 파일·입력 (재현용)
share/                 팀원에게 전달한 자료(코드 ZIP, SSG 종료 시각 데이터셋)
docs/                  이전 작업 로그 (README_작업로그.md: 시험 결과와 시행착오가 길게 적혀 있음)
kaggle/                Kaggle 노트북 (LightGBM만, 최신 코드 아님·미검증)
```

## 3. 코드: 최종에 쓰인 것과 시험만 한 것
| 파일 | 역할 | 최종 |
|---|---|---|
| `preprocess.py`, `train_lgbm.py` | 피처 53개, LightGBM 3설정×시드3 (v9, 검증 RMSLE 0.1808) | ✔ (F의 재료) |
| `reproduce_v3.py` | 처음 제출한 LightGBM v3를 똑같이 재현 | ✔ (팀 블렌드의 nayun_v3) |
| `train_xgb.py`, `train_catboost.py`, `train_mlp.py`, `tune_ours_weights.py` | v9 피처 XGBoost·CatBoost·MLP, 혼합 비율 검증(LightGBM 0.80 + XGBoost 0.12 + CatBoost 0.08) | F의 재료 |
| `stack_anchor.py`, `make_F_xgb.py`, `make_final.py` | 작년 같은 요일 앵커 스태커, F·K 만들기 | ✔ (F_v10_stack) |
| `apply_postprocess.py` + `postprocess/` | 팀원 후처리(축제·명절) | 팀원 코드 사본 |
| `geomdan_adjust.py` | 검단 개통 보정(팀원 메모로 재구현) | 팀원 코드와 같은지 미확인 |
| `ssg_adjust.py`, `fit_ssg_profile.py`, `fit_ssg_margin.py` | SSG 경기 종료 시각·점수 차 보정 | OPT4에서 0.00022 개선, **OPT5 위에서는 나빠져 미사용** |
| `exam_adjust.py` | 검바위 인근 학교 시험 기간(12시 승차 +450명) 보정, 9행 | ✔ (gb_v6_orig 안에 포함, 시험일 날짜 목록 제공) |
| `fc_adjust.py` | 부천FC 경기 종료 시간대·관중 보정 | ✔ 최종 1·2 |
| `misc_adjust.py` (ftuned, v3swap) | 팀 블렌드 안 우리 몫을 검증 최적 혼합으로 교체 | ✔ 최종 1·2 |
| `exam_all_adjust.py` | 다른 역 시험일 10~14시 보정 (7월) | ✔ 최종 1 |
| `run_final.py` | 최종 2개 일괄 재현·검사 | ✔ |
| `station_trend_adjust.py`, `festival_extra_adjust.py`, `exam_all_adjust.py B` | 역별 추세, 풍물·국화 축제, 두 번째 학교 무리 | 시험만 함 (Public 악화·무변화) |
| `geombawi_adjust.py` | 검바위 11·13시, 15·16시 반복 패턴·날짜 불확실성 변형 | 시험만 함 |
| `bus_adjust.py`, `yoy_anchor.py`, `train_residual.py`, `train_twostage.py` | 버스 변화 보정, 앵커, 잔차 모델, 2단계 모델 | 시험만 함 (효과 없음) |
| `baseline.py`, `calendar_data.py`, `build_*.py`, `fetch_*.py`, `merge_team_data.py` | 베이스라인·외부 자료 생성 | 데이터 준비 |

## 4. 재현 (우리 몫)
```bash
pip install -r requirements.txt            # Python 3.14.3, pandas 3.1.1, numpy 2.5.4, lightgbm 4.7.0 …
python src/reproduce_v3.py                 # v3 (약 3분, 원본과 같음)
python src/train_lgbm.py                   # v9 (약 25분)
python src/train_xgb.py final 1300 ; python src/train_catboost.py
python src/make_F_xgb.py outputs/F.csv tuned   # F (앵커 스태커 포함)
```
OPT5까지의 팀 블렌드·후처리는 팀원 코드로 재현해야 한다 (`FINAL/README.md` 참고).

## 5. 규칙 준수 메모
- 2026-07 이후 **지하철 승하차 실적은 어떤 파일에도 없다.** 날씨·행사·KBO 경기 기록·버스 이용량 등 외부 자료는 출처와 조회일을 `data/external/SOURCES.md`에 기록했다 (팀원이 준 버스 자료는 원본 출처를 받지 못해 확인 필요).
- Public 점수에 맞춰 예측이나 비중을 반복 조정하지 않았다. 점수는 이미 만든 후보 중 고르는 데만 썼다 (RULES.md 금지 목록 3번).
- 시드 고정: LightGBM 42·7·2026, CatBoost 42·7, XGBoost 42·7·2026, MLP 42.

## 6. 알려진 한계
- 검증은 학습 데이터 안의 두 구간이라 Public·Private과 다를 수 있다.
- 2026-07~10월 같은 시기 학습 기록은 2025년 한 해뿐이다.
- 이후 시험한 보정(SSG 종료 시각 등)은 일부 기반에서만 이득이 있었고, 팀원 후처리와 겹치면 오히려 나빠졌다.
