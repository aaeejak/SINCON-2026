# outputs 폴더

- `scored/` — Public 점수가 확인된 파일 (이름 앞이 점수). 점수표: `SCORES.md`
- `models/` — 우리 모델 예측: `lgbm_v3`, `lgbm_v9`, `catboost_v9`, `mlp_v9`, `xgb_v9`, `ours_v9`(= LightGBM 0.72 + CatBoost 0.18 + MLP 0.10), `baseline`, `cv_lgbm_v3/v9.json`
- `archive/` — 점수를 확인하지 않은 후보, 이전 버전, 학습 로그. 필요 없으면 폴더째 삭제해도 된다. (스크립트를 다시 돌리면 `outputs/`에 `submission_lgbm.csv` 등이 새로 생긴다.)
- 최종 제출 파일은 `../FINAL/final_submission.csv`

## candidates/ — 아직 점수를 확인하지 않은 새 후보
- `cand_OPT5_exam.csv` = 팀원 OPT5(0.146311) + 검바위(3208) 인근 학교 시험 기간 보정 (`src/exam_adjust.py`, 9행: 2026-07-02·03·06·07·08, 10-15·16·19·20의 12시 +270명). 제출 전.
- `cand_exam_hours.csv` = 최고본 + 시험일 11시 +43, 13시 +53 (18행). `cand_exam_1516.csv` = 최고본 + 검바위 15·16시 반복 패턴 364일 이동일에 +100 (16행). `cand_exam_hours_1516.csv` = 둘 다 (34행). `cand_exam_spread.csv` = OPT5 + 12시 보정을 정렬일 ×0.5 + 앞뒤 하루 ×0.1로 나눈 것 (13행, 날짜 불확실성 대응). 모두 제출 전. 코드 `src/geombawi_adjust.py`.

## 2026-11-13 오후: fix_cancel 기반 후보 (제출 전)
기준 `team_inputs/input_team_OPT5pp_team50_fix_cancel.csv` (0.144518). 우리 검증된 검바위 보정을 얹음.
- `cand_FC_exam_hours_1516.csv` = fix_cancel + 시험일 12시 +270, 11시 +43, 13시 +53 + 15·16시 반복 패턴 +100 (43행) — OPT5에서 같은 조합이 0.146180(가장 좋음)
- `cand_FC_exam_1516.csv` = fix_cancel + 시험일 12시 + 15·16시 (25행, OPT5에서 0.146200)
- `cand_FC_exam.csv` = fix_cancel + 시험일 12시만 (9행, OPT5에서 0.146293)
- `cand_FC_pungmul1002.csv` = fix_cancel + 부평풍물대축제 10/2(금) 보정 (68행, 팀원 10/3~4 보정값의 0.5배를 복제, 근거 약함 — 축제가 10/2~4 3일임은 확인했으나 금요일 효과 크기는 가정)

## 2026-11-13 저녁: 현재 최고본(cand_FC_exam_hours_1516, 0.144388) 위 새 후보 (제출 전)
- `cand_best_fc.csv` — + 부천FC 홈경기 종료 시간대·관중 보정 (`src/fc_adjust.py`, 12경기 47행). 검증 17경기 경기 단위 LOO로 경기일 3754 RMSE 0.383→0.319. 최고본의 부천FC 경기일 모양은 우리 LightGBM과 거의 같아(rel 0 차이 -0.06) 중복 아님.
- `cand_best_pungmul.csv` — + 부평풍물대축제 10/2(금) (실제 개최 10/2~4 확인, 효과 크기 0.5배는 가정, 68행).
- `cand_best_fc_pungmul.csv` — 위 둘 다 (115행).
- `cand_best_fc_pungmul_chrys.csv` — + 드림파크 국화축제 금요일 10/16·10/23 (근거 약함, 133행).

## 팀원 gb_v6_orig(0.144110, 현재 최고) 위 후보 (제출 전)
gb_v6_orig는 우리 최고본(0.144388)과 거리 0.009 — 검바위 보정이 그대로 들어 있고, 전체에 작은 혼합 변화(표준편차 0.009)와 문학 10/24~25 소폭 상향이 있다. 부천FC 경기일은 우리 최고본과 같다(rel 0 차이 +0.006) → 부천FC 보정은 중복이 아니다.
- `cand_gb_fc.csv` (+부천FC 47행), `cand_gb_pungmul.csv` (+풍물 10/2 68행), `cand_gb_fc_pungmul.csv` (둘 다), `cand_gb_fc_pungmul_chrys.csv` (+국화축제 금요일, 근거 약함)
- 추가(시험용, 근거 약함): `cand_gb_Ftuned` (F 안의 우리 몫을 검증 B 비율로 교체한 근사, src/misc_adjust.py ftuned), `cand_gb_hangul` (10/9 한글날 전 역 -0.035), `cand_gb_fc_pungmul_Ftuned`, `cand_gb_fc_pungmul_Ftuned_hangul`.
- SSG 종료 시각 보정은 gb_v6_orig가 이미 종료 1시간 전 -0.17, 종료 시간대 +0.22, 2시간 뒤 -0.31로 자체 보정을 갖고 있고, OPT5 위에서 우리 보정을 더하거나 바꾼 후보가 모두 나빴으므로 다시 만들지 않음.
- 추가: `cand_gb_fc_Ftuned` (부천FC + F 교체), `cand_gb_v3swap` (팀 블렌드의 nayun_v3 자리를 검증 최적 혼합으로 교체한 근사, src/misc_adjust.py v3swap), `cand_gb_Ftuned_v3swap`, `cand_gb_fc_Ftuned_v3swap`.
- 추가: 시험일 일찍 하교 보정을 다른 역으로 넓힘 (`src/exam_all_adjust.py`, 상승표 `data/derived/exam_day_lift.csv`). 기준 = `cand_gb_fc_Ftuned_v3swap`(0.14403).
  - `cand_best_examall`: 시험 추정일 9일 × 10~14시, k=0.466 (3,096행, 합계 약 +3,300명)
  - `cand_best_examall_thr`: 상승 |0.1| 이상인 역·시간만, k=0.548 (153행)
  - `cand_best_examall_jul`: 7월 시험일만 (검증에서 7월 블록은 개선, 10월 블록은 개선 없음) (1,720행)
- 점수: examall 0.14399, examall_thr 0.14401, examall_jul 0.14399.
- 추가: 두 번째 학교 무리(부평삼거리 13시·독정 10~12시 등, 검바위와 다른 날 시험) 보정 `src/exam_all_adjust.py B`, 상승표 `data/derived/exam_day_lift_groupB.csv`.
  2025-07-01~02의 364일 뒤(2026-06-29~30)에 학습 데이터에서 재현됨. 추정일 2026-07-01, 09-28~10-01. LOBO k=0.48.
  - `cand_best_examall_B`: examall + B 5일 (1,720행)
  - `cand_best_examall_Bjul`: examall + B 7/1만 (344행, 근거 가장 강함)
  - `cand_best_examjul_Bjul`: examall_jul + B 7/1만 (7월만 쓰는 쪽)
- 시험일 15~17시 하락은 검증에서 신호 없음(k≈0) → 미적용.
- 점수: examall_B 0.14405(악화), examall_Bjul 0.14399, examjul_Bjul 0.14399 → 두 번째 학교 무리 보정은 효과 없음.
- 확인만 함: 전체 수준(오래된 68역 평일 전년비 관측 1.03 ≈ 예측 1.02~1.04)은 맞음. 검증 B의 주말 과소 예측은 4월(벚꽃 철) 영향이라 미적용.
- 추가: 역별 최근 추세 보정 `src/station_trend_adjust.py` (기준 cand_best_examall 0.14399). 최근 3개월 실제 전년비 − 예측 전년비의 차이를
  역 간 상대 보정으로 일부 반영. 검증 B: 상관 0.62, k=0.49, 해당 56역 RMSLE 0.14697→0.14612. 검단 개통 전후 급변 역·신설 역·문학경기장 제외.
  최고 파일에서의 차이 폭(±0.04)이 검증 때보다 작아 실제 효과는 더 작을 것.
  - `cand_best_sttrend` (k=0.49), `cand_best_sttrend_half` (k=0.25), `cand_best_sttrend_wd` (k=0.49, 평일만)
- 점수: sttrend / sttrend_half / sttrend_wd 셋 다 0.14399보다 나쁨(사용자 보고, 숫자 미수령) → 역별 추세 보정 미사용.
