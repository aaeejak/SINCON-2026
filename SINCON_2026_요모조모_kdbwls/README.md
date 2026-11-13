# SINCON_2026_요모조모 — 최종 제출 A (`submission_fixcancel_geombawi`, Public 0.144388)

팀 요모조모(kdbwls · nayun · Dohyeong)의 Kaggle 최종 선택 2개 중 **최종 A**의 재현 자료다. 이 경로의 섞기·후처리 체인은 kdbwls가, 마지막 검바위 보정은 nayun이 만들었다. 최종 B(`cand_best_examall_jul`, 0.14399)는 nayun 패키지 `SINCON_최종제출_nayun/`에 있다.

## 0. 빠른 재현
```bash
pip install -r configs/requirements.txt
# 대회 원본 4개 파일은 data/raw/ 에 들어 있음 (SHA-256으로 원본과 같은지 자동 확인)
python code/run_all.py              # 약 2분: 저장된 개별 모델 예측 → 최종 A, 단계마다 저장본과 비교
python code/run_all.py --retrain    # 약 10분 추가: kdbwls 모델(v7, 역별 LightGBM)도 다시 학습해 비교
```
마지막 줄에 `재현 성공`이 나오면 `work/final_A.csv`가 `predictions/final_A_fixcancel_geombawi.csv`와 같다. 실행 로그: `records/run_log.txt`(기본), `records/run_log_retrain.txt`(재학습 포함 — v7·역별 LightGBM 재학습 결과가 저장본과 log 차이 0).

## 1. 제출 자료 목록

| 구분 | 자료 | 위치 |
|---|---|---|
| 필수 | 최종 예측 파일 | `predictions/final_A_fixcancel_geombawi.csv` (Kaggle에 제출한 원본, 파일명만 바꿈) |
| 필수 | 전체 실행 코드 | `code/` — 실행 순서는 `code/run_all.py` 한 파일에 있음 (아래 3절) |
| 필수 | 데이터 출처·생성 기록 | `provenance/DATA.md`, 특징 설명 `provenance/FEATURES.md`, 해시 `provenance/SHA256SUMS.txt` |
| 필수 | 실행 환경·방법 | 2절, `configs/requirements.txt` |
| 필수 | 설명서 | 4~5절 |
| 사용 시 필수 | 모델·중간 산출물 | `artifacts/models/`(개별 모델 예측), `artifacts/blend_chain/`(섞기·후처리 단계별 결과) |
| 보유 시 제출 | 기존 개발 기록 | `records/` (실행 로그, 검증 결과 json, 실험 기록 목록) |

## 2. 실행 환경과 방법

| 단계 | 작성할 정보 |
|---|---|
| 환경 준비 | Linux, CPU 2코어, GPU 불필요, 메모리 4GB 이하. Python 3.13.18. `pip install -r configs/requirements.txt` (pandas 3.0.5 · numpy 2.5.3 · lightgbm 4.7.0 · scipy 1.18.1). Kaggle Notebook·Colab에서 그대로 실행 가능한 규모 |
| 학습 및 선택 | `python code/run_all.py --retrain`이 v7(`sincon_pipeline_v7.py --mode submit`)과 역별 LightGBM(`scripts/run_per_station_final.py ps15`)을 다시 학습. 사전학습 모델 없음. 검증 분할은 4절. 시드: LightGBM `SEED=42`, 시드 3개(42·43·44) 평균. 생성 모델 파일은 저장하지 않고 매번 학습(모델 1개 약 2~4분) |
| 최종 추론 | 같은 명령이 2026-06-30까지 학습해 2026-07-01~10-31 157,194행을 예측 → 섞기 원본(`artifacts/blend_chain/OPT5_raw.csv`) → 후처리 → 취소 행사 → 검바위 보정. 경로는 모두 `run_all.py`에 고정 |
| 재현 변동 | 트리 모델은 같은 버전·시드·스레드 수에서 같은 결과. 스레드 수나 라이브러리 버전이 다르면 소수점 아래 미세한 차이가 날 수 있음. 섞기 이후 단계는 저장본과 log1p 차이 0(최종 파일 최대 9×10⁻¹⁶)으로 재현됨을 확인 |

## 3. 실행 순서 (`code/run_all.py`)

| 단계 | 내용 | 코드 | 결과 비교 대상 | 재현 형태 |
|---|---|---|---|---|
| 0 | 원본 SHA-256 확인, `work/` 준비 | `run_all.py` | - | - |
| 1 | (선택) kdbwls 모델 재학습: v7, 역별 LightGBM, 수준 맞춤 | `sincon_pipeline_v7.py`, `scripts/run_per_station_final.py`, `sincon_pipeline_v10.py` | `artifacts/models/v7·ps15·ps15_lvl.csv` | 코드 |
| 2 | 섞기 원본 OPT5_raw | (4절 설명, `scripts/blend_public*.py` 방식) | - | **실제 파일** |
| 3 | 후처리: 축제 → 명절 0.5 → 공연장·인접역+OKF → 검단 0.5 → SSG 종료 | `scripts/full_pp.py` (+ event_adjust, holiday_shape, venue_adjust, geomdan_adjust, ssg_end_adjust) | `OPT5_pp.csv` | 코드, 차이 0 |
| 4 | 팀 섞기: 3단계 0.5 + nayun `team_blend_opt4_75_v6pp_25` 0.5 (log1p) | `run_all.py` | `OPT5pp_team50.csv` | 코드, 차이 0 |
| 5 | 4단계에서 절반이 된 문학 SSG 종료·OKF 보정을 원래 세기로 (243행) | - | - | **실제 파일** |
| 6 | 취소 행사: OKF(9/26~27) 상승분 제거, GHOSTIVAL(10/24~25) 문학 3역을 같은 요일 평소 날 평균으로 | `scripts/cancel_adjust.py ... 1.0` | `OPT5pp_team50_fix_cancel.csv` (Public 0.144518) | 코드, 차이 0 |
| 7 | 검바위(3208) 시험 기간: 추정 시험일 9일 11·12·13시 + 반복 패턴 15·16시 (43행) | `code/geombawi_nayun/exam_adjust.py 0.6` → `geombawi_adjust.py hours h1516` (nayun) | `predictions/final_A_fixcancel_geombawi.csv` (0.144388) | 코드, 차이 9×10⁻¹⁶ |

2·5단계는 당시 대화에서 바로 계산해 만들고 스크립트를 남기지 않았다. 규칙 7절("가공 데이터는 실제 파일 또는 동일 생성 코드")에 따라 실제 파일로 제출한다. 개별 팀원 모델 예측(`artifacts/models/`의 F_v10_stack·nayun_v3·dh_v2·dh_mlp)은 각 팀원 코드로 만들며, new_v5data는 `code/feat_v5a_package.zip`으로 만든다(`provenance/DATA.md` 4절).

## 4. 설명서

### 학습과 검증 분할
- 학습: 2025-01-01~2026-06-30 전체. 목표 = log1p(승차) − 학습 구간 역·시간·요일유형 평균. 손실 = 제곱오차(RMSLE와 같은 척도).
- 검증 A: 2025-06-30까지 학습 → 2025-07~10 예측 (같은 계절, 작년 이력 없음).
- 검증 B: 2026-02-28까지 학습 → 2026-03~06 예측 (가장 최근).
- 피처는 두 검증에서 같은 방향으로 좋아질 때만 넣었다. 검증 결과: `records/results_v*.json`, 실행 로그.

### 주요 변수 (자세히는 `provenance/FEATURES.md`)
역·노선·시간·요일·공휴일·연휴 위치·징검다리, 날씨(일 강수·시간대 비·최고기온), SSG 문학 홈경기(일정·시작 시각·관중·상대 인기도), 행사(종류·시작 시각·상대 시간), 검단 입주 세대, 인천대공원 계절·행사, 검단연장 개통, 노선 내 승차 순위·최근 8주 성장, 인천대 학기, 2023·2024 월보 계절 모양, (v10) 작년 같은 시기 편차.

### 모델 선택 근거
- LightGBM log 잔차 회귀가 Poisson·Tweedie 목표, 재귀 예측, 2단계(하루 합 × 시간 비중) 모델보다 두 검증 모두에서 나았다.
- 역별 LightGBM(잎 15)은 단독은 약하지만 오차 모양이 달라 섞기에 도움이 되어 넣었다(검증 B 30% 섞기 0.15824 → 0.15704).
- 팀원 모델(F_v10_stack, new_v5, nayun_v3, Dohyeong v2·MLP)은 서로 오차가 달라 섞을수록 좋아졌다.

### 앙상블 비중과 후처리 계수 결정 방법

| 항목 | 값 | 정한 방법 |
|---|---|---|
| OPT 섞기 비율 | F_v10_stack 0.405 · v7 0.140 · new_v5 0.166 · dh_v2 0.110 · dh_mlp 0.094 · nayun_v3 0.085 (`configs/team_blend_weights_OPT.json`) | 완성된 제출들의 Public 점수와 제출 간 예측 차이로 C 행렬을 만들어 한 번 계산(앙상블 비율 조정, 팀이 운영진에 허용 여부 확인). 비율을 ±30% 흔들어도 예상 점수 변화 0.0002 이하인 평평한 최적점 |
| OPT5 섞기 | OPT4 원본 + 역별 LightGBM(수준 맞춤) 약 7.7% 등 | 같은 방식. 결과를 실제 파일로 제출 |
| 역별 모델 수준 | log −0.021095 | OPT4 원본과 log 평균을 같게 (Public 미사용) |
| 팀 섞기 | 0.5 / 0.5 | 사전에 정한 값 |
| 축제 보정 | 축제별 과거 상승폭 × 사전 배율 | 과거 같은 축제 학습 데이터, 배율은 사전값 |
| 명절 보정 | 모양 0.85(일요일 0.5, 8/15 0.6), 역별 차이 0.5배 | 0.5배: 검증 A(2025 추석 되짚기) RMSE 보정 전 0.4524 → 0배 0.3622 / 0.5배 0.3478 / 1배 0.3547 |
| 공연장·인접역 | 행사장 역 0.4, 인접역 0.7 | 사전값 (행사장 역은 모델에 행사 피처가 있어 약하게) |
| 검단 | 0.5배 | 사전값 |
| SSG 종료 시간대 | 종료 기준 −2~+2시간, 0.6배 | 2025-03~2026-06 124경기 회귀, 검증 A·B 경기 단위 leave-one-out (상승폭 RMSE A 1.003→0.73, B 0.758→0.60) |
| 취소 행사 | 상승분 전부 제거 / 평소 날 평균 | 언론 보도 (DATA.md) |
| 검바위 시험 기간 (nayun) | 12·11·13시 과거 증가 × 0.6, 15·16시 +100명 | 학습 데이터에서 평일 4~5일 연속 12시 3~5배 블록 5개 발견, 2025·2026년 4월 블록이 364일 간격으로 일치 → 2025년 7·10월 블록을 364일 옮겨 날짜 추정. 배율은 제출 전에 정함 |
| 시험 후 뺀 보정 | 부천FC 종료 시간대(kdbwls판), 토요일 오후 | 검증 근거로 만든 값을 한 번 제출해 Public이 나빠져 통째로 뺌. 계수를 바꿔 다시 내지 않음 |

### Public 점수 사용 범위
완성된 후보 중 고르기와 앙상블 비율 계산에만 썼다. 전체 수준을 올려 Public 반응을 본 탐색 제출(`X3_ev_hol_p015` 등)은 최종 경로에 쓰이지 않았다(OPT 원본이 json 비율 그대로이고 상수 이동이 없음을 확인).

## 5. 모델 및 중간 산출물
- 트리 모델 파일은 저장하지 않는다. 같은 코드·시드로 다시 학습한다(`--retrain`).
- 개별 모델 예측과 단계별 결과는 `artifacts/`. 누가 어떤 코드로 만들었는지는 `provenance/DATA.md` 4절.
- 사전학습 모델·임베딩·캐시는 쓰지 않았다.

## 6. 기존 개발 기록 (`records/`)
- `run_log_retrain.txt`: 이 패키지를 처음부터 실행한 로그(검증 A·B 점수 포함)
- `results_v4~v8*.json`: 피처 버전별 검증 결과
- 실험 기록 문서 40여 개(2026-11-11~13)는 팀 Claude 프로젝트 "SINCON"에 있음. 목록은 `records/EXPERIMENT_LOG.md`

## 7. 제출 전 자체 점검
- [x] 최종 예측 파일과 코드·모델·설정의 관계: 3절 표, `run_all.py`가 단계마다 저장본과 비교
- [x] 데이터·모델 작성표에 사용 항목 기록, 미사용 항목은 해당 없음 표시: `provenance/DATA.md` 2~4절
- [x] 제공 자료부터 최종 예측까지 경로·명령·의존성·시드: 0·2·3절
- [x] 평가 정답 관련 자료: 2026-07 이후 지하철 승하차 인원을 열람·확보·사용한 사실 없음. 외부 파일 전수 점검 결과 2026-07~10 승차 인원 행 0개. 운영진 공지(공개 데이터 중 정답 포함 자료 존재) 이후 사용 자료를 다시 점검함
