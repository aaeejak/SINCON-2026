# outputs 폴더 안내

**현재 확인된 최고 점수: 팀원 `OPT5pp_team50.csv` 0.146311** (`final/input_team_OPT5pp_team50.csv`). 우리 기여 후보: `candidates/cand_AK_OPT5_ssg_add.csv`, `cand_AL_OPT5_ssg_replace.csv` (OPT5 + SSG 종료 시각 보정).

**최종 제출용 CSV: `final_submission_0.147532.csv`** (= `candidates/cand_AF_OPT4_ssg.csv`, Kaggle Public 0.147532 확인됨, `final/`에도 사본). 재생성: 팀원 `submission_OPT4_final_geo.csv`(`final/input_team_OPT4_final_geo.csv`)에 `python src/ssg_adjust.py 입력 출력` (0.7).

**현재 확인된 최고 점수: `candidates/cand_AF_OPT4_ssg.csv` 0.147532** (팀원 OPT4 0.147755 + SSG 종료 시각 보정, 개선 0.00022). 이전 최고: 팀원 `submission_OPT4_final_geo.csv` 0.147755 (`final/input_team_OPT4_final_geo.csv`). 우리 기여 후보: `candidates/cand_AF_OPT4_ssg.csv` (OPT4 + SSG 종료 시각 보정).

**최신 제출 후보: `final_KP_postproc.csv`** = `final_K_0.151321.csv`(K)에 팀원 후처리(축제 보정 + 명절 보정 역별 0.5배, `src/apply_postprocess.py`)를 적용한 것. 팀원 보고상 같은 후처리가 팀 블렌드를 0.151361 → 0.148399로 낮춤. K에 대한 점수는 제출 전.

**최종 제출 파일: `final_K_0.151321.csv`** (= `candidates/cand_K_B45_F35_blend5_20.csv`, Public 0.151321). `python src/make_final.py`로 입력 파일들에서 똑같이 다시 만들 수 있음 (log 거리 4e-16 확인).

이름 규칙: `<종류>_v<버전>.csv` (모델 예측), `cand_<알파벳>_<설명>.csv` (제출 후보). 모든 CSV는 `example_id, boardings` 157,194행.

## models/ — 우리 모델 예측
| 파일 | 내용 |
|---|---|
| `lgbm_v3.csv` | 처음 제출한 기존 lgbm (Public 0.16087, 팀 블렌드의 "nayun lgbm"). `src/reproduce_v3.py`로 똑같이 재현됨 |
| `lgbm_v9.csv` | 강화한 LightGBM (피처 53개, 3설정×시드3 평균, 검증 0.1808) |
| `catboost_v9.csv`, `mlp_v9.csv` | v9 피처로 학습한 CatBoost, 임베딩 MLP |
| `xgb_v9.csv` | v9 피처로 학습한 XGBoost (시드 3개 평균, 검증 0.1981/0.1696, LightGBM과 섞으면 개선) |
| `ours_v9.csv` | `lgbm_v9` 0.72 + `catboost_v9` 0.18 + `mlp_v9` 0.10 (log 공간) |
| `baseline.csv` | 베이스라인(최근 8주 평균) |
| `cv_lgbm_v3.json`, `cv_lgbm_v9.json` | 설정·검증 점수 기록 |

## candidates/ — 제출 후보 (최종 선택은 직접)
| 파일 | 내용 | 점수 |
|---|---|---|
| `cand_A_blend5.csv` | 팀 blend5 그대로 (v3 성분 포함) | 0.151760 |
| `cand_B_blend5_swap_lgbm_v9.csv` | A에서 v3 성분만 `lgbm_v9`로 교체 | 0.151721 |
| `cand_C_scored_0.15307.csv` | 우리 v7 기반 블렌드 (제출 완료) | 0.15307 |
| `cand_D_v10.csv` | v9 파이널 + 새 팀원 모델(v5data) 0.15 | 0.152404 |
| `cand_E_v10_anchor.csv` | D에 작년 앵커 (낮은 점수, 사용 안 함) | 별로 (사용자 보고) |
| `cand_G_v9.csv`, `cand_H_v9_anchor.csv`, `cand_I_v9_stack.csv` | v9 기반 같은 변형 (G 0.152912, I 0.152881, H 앵커는 별로) | 위 참고 |
| `cand_J_blend_of_8.csv` | 점수가 괜찮았던 8개(`cand_C`, blend4, blend5, `cand_B`, `cand_D`, `cand_F`, `cand_G`, `cand_I`)를 log 공간 균등 평균 (점수 기반 비중 조정 없음) | 추정 0.15184 |
| `cand_K_B45_F35_blend5_20.csv` | **B 0.45 + F 0.35 + blend5 0.20** (log 공간, 후보 8개의 Public 점수로 거리 항등식 계산한 비중) | **0.151321 (현재 최고)** |
| `cand_L_K_stack.csv` | `cand_K`에 앵커 스태커 보정을 0.7배로 한 번 더 적용 (K 안의 F가 이미 보정을 포함해 이중 적용됨) | 미확인 |
| `cand_M_K_festival.csv` | `cand_K`에 소규모 축제 보정(계양아라온 +0.07, 드림파크 국화축제 +0.12 log, 10시 이후)만 적용. 추정 이득 약 0.0001 이하 | 미확인 |
| `cand_N_K_postproc.csv` | **K + 팀원 후처리** (= `final_KP_postproc.csv`) | 제출 전 (OPT와 거리 0.004라 약 0.1483 예상) |
| `cand_O_KP_OPT_5050.csv` | `cand_N`(K+후처리)과 팀원 `submission_OPT_ev_hol_s05.csv`(0.148301)를 log 공간 0.5/0.5 (둘 사이 거리 0.004로 사실상 같은 파일) | 미확인 (예상 0.1482~0.1484) |
| `cand_P_Kx_PP.csv` | K에서 F를 XGBoost 섞은 F(`cand_F_xgb`)로 바꾸고 후처리 | 제출 전 |
| `cand_Q_B_PP.csv` | 스태커 없이 B(blend5에서 v3→v9) + 후처리 | 제출 전 |
| `cand_R_OPT_Fx.csv` | 팀원 최종 OPT의 F 자리(0.405)만 새 F로 교체 | 제출 전 |
| `cand_S_avg4.csv` | N, OPT, P, Q의 log 평균 | 제출 전 |
| `cand_T_ours_PP.csv` | 우리 모델(0.7·ours_v9 + 0.3·xgb_v9)만 + 후처리 (진단용, 우리 단독 실력 확인) | 제출 전 |
| `cand_U_OPT_v3swap.csv` | OPT의 nayun_v3 자리(0.085)를 `lgbm_v9`로 교체 | 제출 전 |
| `cand_V_K_festival_only.csv` | K + 축제 보정만 (후처리 분해용) | 제출 전 |
| `cand_W_K_holiday_only_s05.csv` | K + 명절 보정(역별 0.5배)만 | 제출 전 |
| `cand_X_K_PP_holiday_s10.csv` | K + 축제 + 명절(역별 1.0배, 팀원이 시험한 다른 값) | 제출 전 |
| `cand_Y_OPT_geomdan.csv` | 팀원 OPT(0.148301) + 검단 개통 보정 (`src/geomdan_adjust.py`, 팀원 메모 기반 재구현. 팀원 `submission_best_geomdan.csv`와 같은 방식) | 제출 전 |
| `cand_Z_N_geomdan.csv` | `cand_N`(K+후처리) + 검단 개통 보정 | 제출 전 |
| `cand_AA_O_geomdan.csv` | `cand_O`(N·OPT 평균) + 검단 개통 보정 | 제출 전 |
| `cand_F_tuned.csv` | F에서 우리 모델 자리를 검증으로 정한 비율(LightGBM 0.80 + XGBoost 0.12 + CatBoost 0.08, MLP 제외)로 바꾼 F (`src/make_F_xgb.py ... tuned`, 비율 근거 `src/tune_ours_weights.py`) | 제출 전 |
| `cand_AG_AF_bus_all.csv` | `cand_AF`(0.147532) + 역 주변 버스 이용량 변화 보정 k=0.25, 7~9월 전체 71개 역 (`src/bus_adjust.py`) | 제출 전 |
| `cand_AH_AF_bus_excl5.csv` | 위에서 검단 3역·계양·귤현(팀 검단 보정과 겹침) 제외 | 제출 전 |
| `cand_AI_AF_bus_half.csv` | `cand_AG`의 절반 세기(k=0.12) | 제출 전 |
| `cand_AJ_OPT4_ssg_margin.csv` | `cand_AF`에 SSG 점수 차 보정 추가 (`src/ssg_adjust.py ... margin`, 계수 `data/derived/ssg_margin_profile.csv`). 기대 이득 약 0.0001, 행별 변화 -0.69~+0.44로 큼 | 제출 전 |
| `cand_AK_OPT5_ssg_add.csv` | 팀원 `OPT5pp_team50.csv`(**0.146311**) + 우리 SSG 종료 시각 보정(0.7, 136행)을 그대로 더함. OPT5가 이미 같은 136행에 일부 변화(종료 시간대 +0.14 등)를 넣어 부분 중복 | 제출 전 |
| `cand_AL_OPT5_ssg_replace.csv` | OPT5에서 같은 136행만 `cand_AF`의 값(OPT4 + 우리 보정)으로 교체 — 중복 없이 우리 보정만 적용 | 제출 전 |
| `cand_AE_OPT_ssg_only.csv` | 팀원 OPT + SSG 경기 종료 시각 보정만 (`src/ssg_adjust.py`, 136행) | 제출 전 |
| `cand_AB_Y_ssg.csv` | `cand_Y`(OPT + 검단 보정) + SSG 보정 | 제출 전 |
| `cand_AC_Z_ssg.csv` | `cand_Z`(N + 검단 보정) + SSG 보정 | 제출 전 |
| `cand_AD_AA_ssg.csv` | `cand_AA`(O + 검단 보정) + SSG 보정 | 제출 전 |
| **`cand_AF_OPT4_ssg.csv`** | 팀원 `submission_OPT4_final_geo.csv`(0.147755) + SSG 경기 종료 시각 보정. OPT4는 이미 검단 보정을 포함하므로 우리 검단 보정은 더하지 않음 | **0.147532** (보고된 점수, 현재 최고) |

## archive/ — 이전 단계·중복·로그 (필요 없으면 폴더째 삭제해도 됨)
이전 버전 `lgbm_v1~v6`, 예전 블렌드(`blend_*`), v8 시절 파일(`v8_*`), 학습 로그, 앵커 배열 등.
`dup_*` 파일은 다른 곳에 같은 내용이 있는 사본.

## 코드를 다시 돌릴 때
스크립트는 기존 이름으로 `outputs/`에 새로 씁니다 (`submission_lgbm.csv`, `cv_lgbm.json`, `submission_catboost.csv`, `submission_mlp.csv` 등).
결과를 보관하려면 위 규칙대로 이름을 바꿔 `models/`에 넣으세요. `train_catboost.py`는 `outputs/submission_lgbm.csv`가 있어야 돌아갑니다 (`train_lgbm.py`를 먼저 실행).
