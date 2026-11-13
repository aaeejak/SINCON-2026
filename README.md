# SINCON_2026_요모조모 — 제출 자료

팀 요모조모 (kdbwls · nayun · Dohyeong). 인천 지하철 71개 역의 2026-07-01 ~ 10-31 시간대별(06~24시) 승차 인원 157,194행 예측, 지표 RMSLE.
이 문서는 「SINCON 2026 제출 자료 안내」의 항목 순서를 따른다.

> **GitHub 저장소에는 CSV 파일을 올리지 않았다.** 최종 예측 파일, 개별 모델 예측, 섞기 단계별 결과, 외부 자료 CSV를 포함한 전체 자료는 운영진에 제출한 ZIP(`SINCON_2026_요모조모.zip`)에 같은 폴더 구조로 들어 있다. 재현은 ZIP을 푼 폴더에서 아래 명령으로 한다. 이 문서에 적힌 CSV 경로는 ZIP 안의 경로다.

## 0. 최종 예측 파일 (Kaggle 최종 선택 1개)

| Kaggle 제출 파일명 | Public | 이 자료 안의 원본 | SHA-256 |
|---|---|---|---|
| `submission_fixcancel_geombawi_0.144388.csv` | 0.144388 | `SINCON_2026_요모조모_kdbwls/predictions/final_A_fixcancel_geombawi.csv` (Kaggle에 올린 파일, 이름만 바꿈) | `a9c43388136c4132b1964dd748b81842d84e1008687e2362e3146dc03e1a1938` |

생성 경로:

```
개별 모델 8개 예측 (kdbwls v7·역별 LightGBM·new_v5 / nayun F_v10_stack·v3 / Dohyeong v2·MLP)
  → [1] log1p 가중 평균 = OPT5_raw
  → [2] 후처리: 축제 → 명절(역별 0.5배) → 공연장·인접역(+OKF) → 검단(0.5배) → SSG 경기 종료 시간대 = OPT5_pp
  → [3] 팀 섞기: OPT5_pp 0.5 + nayun 후처리본 0.5 = OPT5pp_team50
  → [4] 반반 섞기로 절반이 된 문학경기장 보정을 원래 세기로 복원 (243행) = OPT5pp_team50_fix
  → [5] 취소 행사 반영 (OKF 9/26~27, GHOSTIVAL 10/24~25) = OPT5pp_team50_fix_cancel (Public 0.144518)
  → [6] 검바위(3208) 시험 기간 보정 (43행) = 최종 제출 (Public 0.144388)
```

## 빠른 재현 (CPU, 약 2분)

```bash
python setup_raw_data.py <train.csv·test.csv·metaData.csv·sample_submission.csv 가 있는 폴더>   # 대회 원본 배치 + SHA-256 확인
cd SINCON_2026_요모조모_kdbwls
python code/run_all.py              # 저장된 개별 모델 예측 → 최종 제출, 단계마다 저장본과 비교
python code/run_all.py --retrain    # (선택, 약 10~30분 추가) kdbwls 모델 v7·역별 LightGBM도 다시 학습해 비교
```

마지막 줄에 `재현 성공`이 나오면 `work/final_A.csv`가 제출 원본과 같다 (최대 log 차이 8.9×10⁻¹⁶).
Kaggle Notebook·Google Colab에서는 `run_on_kaggle_colab.ipynb`를 위에서부터 실행하면 된다 (7절).

## 1. 제출 자료 목록과 위치

| 구분 | 자료 | 위치 |
|---|---|---|
| 필수 | 최종 예측 파일 | `SINCON_2026_요모조모_kdbwls/predictions/final_A_fixcancel_geombawi.csv` |
| 필수 | 전체 실행 코드 | 최종 경로: `SINCON_2026_요모조모_kdbwls/code/` (실행 순서는 `run_all.py` 한 파일). 개별 모델: 4절 표의 각 코드 |
| 필수 | 데이터 출처 및 생성 기록 | 5절, `SINCON_2026_요모조모_kdbwls/provenance/DATA.md`·`FEATURES.md`·`SHA256SUMS.txt`, `SINCON_최종제출_nayun/data/external/SOURCES.md`, `SINCON_2026_요모조모_Dohyeong/code/dh_models/sincon_추가데이터2/external/SOURCES.md` |
| 필수 | 실행 환경 및 방법 | 2절, `SINCON_2026_요모조모_kdbwls/configs/requirements.txt` |
| 필수 | 설명서 | 3절 (상세: `SINCON_2026_요모조모_kdbwls/README.md` 4절) |
| 사용 시 필수 | 모델 및 중간 산출물 | 6절 |
| 보유 시 제출 | 기존 개발 기록 | 8절 |

폴더:

| 경로 | 담당 | 내용 |
|---|---|---|
| `README.md`, `setup_raw_data.py`, `run_on_kaggle_colab.ipynb` | 팀 | 이 문서, 대회 원본 배치 스크립트, Kaggle/Colab 실행 노트북 |
| `SINCON_2026_요모조모_kdbwls/` | kdbwls (+ nayun 검바위 보정) | **최종 제출 재현 패키지**: `code/`, `configs/`, `data/`(외부 자료), `provenance/`, `artifacts/`(개별 모델 예측·섞기 단계별 결과), `predictions/`, `records/` |
| `SINCON_최종제출_nayun/` | nayun | 개별 모델 F_v10_stack·nayun_v3를 만든 코드와 입력 (`src/`, `data/`, `team_inputs/`, `outputs/models/`). 이 폴더의 `FINAL/`과 "최종 1·2"는 선택하지 않은 후보이며 최종 제출과 무관하다 |
| `SINCON_2026_요모조모_Dohyeong/` | Dohyeong | 개별 모델 dh_v2·dh_mlp를 만든 코드와 외부 자료 (`code/dh_models/`) |

## 2. 실행 환경과 방법

| 단계 | 내용 |
|---|---|
| 환경 준비 | 개발 환경: Linux, CPU 2코어, GPU 없음, 메모리 4GB 이하, Python 3.13.18. `pip install -r SINCON_2026_요모조모_kdbwls/configs/requirements.txt` (pandas 3.0.5 · numpy 2.5.3 · lightgbm 4.7.0 · scipy 1.18.1). 이 고정 버전은 Python 3.12 이상용이다. Python 3.10~3.11 런타임(Kaggle·Colab 등)에서는 설치하지 말고 기본 설치된 pandas·numpy·lightgbm·scipy를 쓰면 된다 — Python 3.10.12 · pandas 2.3.3 · numpy 2.2.6 · lightgbm 4.7.0 · scipy 1.15.3에서도 같은 결과로 재현됨을 확인했다 (`SINCON_2026_요모조모_kdbwls/records/run_log_py310.txt`) |
| 학습 및 선택 | `python code/run_all.py --retrain`: v7(`sincon_pipeline_v7.py --mode submit`)과 역별 LightGBM(`scripts/run_per_station_final.py ps15`)을 2026-06-30까지 학습해 다시 만든다. 사전학습(초기) 모델 없음. 검증 분할은 3절. 시드: LightGBM `SEED=42`, 시드 3개(42·43·44) 평균. 모델 파일은 저장하지 않고 매번 학습 (모델 1개 약 2~4분, 2코어 기준). 다른 팀원 모델의 재학습 명령은 4절 |
| 최종 추론 | `python code/run_all.py`: 저장된 개별 모델 예측(`artifacts/models/`) → 섞기 원본(`artifacts/blend_chain/OPT5_raw.csv`) → 후처리(`scripts/full_pp.py`) → 팀 섞기 → 보정 복원 → 취소 행사(`scripts/cancel_adjust.py ... 1.0`) → 검바위 보정(`geombawi_nayun/exam_adjust.py 0.6` → `geombawi_adjust.py hours h1516`). 경로·계수는 모두 `run_all.py`에 고정 |
| 재현 변동 | 섞기 이후 단계는 결정적 연산이며 저장본과 log1p 차이 최대 8.9×10⁻¹⁶. 트리 모델 재학습은 같은 버전·시드·스레드 수에서 같은 결과이고, 스레드 수나 라이브러리 버전이 다르면 소수점 아래 차이가 날 수 있다. 기존 재학습 실행 로그: `records/run_log_retrain.txt` (v7·역별 LightGBM 재학습 결과가 저장본과 log 차이 0) |

## 3. 설명서

### 학습과 검증 분할
- 학습: 2025-01-01 ~ 2026-06-30 전체. 2026-06-30을 기준일로 넉 달을 한 번에 예측한다. 목표 = log1p(승차) − 학습 구간 역·시간·요일유형 평균, 손실 = 제곱오차 (RMSLE와 같은 척도).
- 검증 A: 2025-06-30까지 학습 → 2025-07~10 예측 (같은 계절).
- 검증 B: 2026-02-28까지 학습 → 2026-03~06 예측 (가장 최근 넉 달).
- 피처는 두 검증에서 같은 방향으로 좋아질 때만 넣었다. 결과: `records/results_v*.json`.

### 주요 변수
역·노선·시간·요일·공휴일·연휴 위치·징검다리, 날씨(일 강수·시간대 비·최고기온), SSG 문학 홈경기(일정·시작 시각·관중·상대 인기도), 행사(종류·시작 시각·상대 시간), 검단 입주 세대, 인천대공원 계절·행사, 검단연장 개통, 노선 내 승차 순위·최근 8주 성장, 인천대 학기, 2023·2024 월보 계절 모양, 작년 같은 시기 편차.
이름·의미·원천 자료·계산 방식·참조 기간: `SINCON_2026_요모조모_kdbwls/provenance/FEATURES.md` (kdbwls 모델), `feat_v5a_package/feature_list.csv` (new_v5, `code/feat_v5a_package.zip` 안), `SINCON_최종제출_nayun/src/preprocess.py` (nayun 모델), `SINCON_2026_요모조모_Dohyeong/code/dh_models/export_processed_data.py` (Dohyeong 모델).

### 모델 선택 근거
- LightGBM log 잔차 회귀가 Poisson·Tweedie 목표, 재귀 예측, 2단계(하루 합 × 시간 비중) 모델보다 두 검증 모두에서 나았다.
- 역별 LightGBM(잎 15)은 단독으로는 약하지만 오차 모양이 달라 섞기에 도움이 되었다 (검증 B 30% 섞기 0.15824 → 0.15704).
- 팀원 모델(F_v10_stack, new_v5, nayun_v3, Dohyeong v2·MLP)은 서로 오차가 달라 섞을수록 좋아졌다.

### 앙상블 비중과 후처리 계수 결정 방법

| 항목 | 값 | 정한 방법 |
|---|---|---|
| 섞기 비율 [1] | F_v10_stack 0.405 · v7 0.140 · new_v5 0.166 · dh_v2 0.110 · dh_mlp 0.094 · nayun_v3 0.085 (`configs/team_blend_weights_OPT.json`), 여기에 역별 LightGBM(수준 맞춤) 약 7.7% | 완성된 제출들의 Public 점수와 제출 간 예측 차이로 행렬을 만들어 한 번 계산 (`code/scripts/blend_public.py`·`blend_public2.py`). 비율을 ±30% 흔들어도 예상 점수 변화 0.0002 이하인 평평한 최적점. 최종 비율을 저장한 스크립트가 없어 결과(`OPT5_raw.csv`)를 실제 파일로 제출 |
| 역별 모델 수준 | log −0.021095 | OPT4 원본 섞기와 log 평균을 같게 맞춤 (Public 미사용) |
| 축제 보정 [2] | 축제별 과거 상승폭 × 사전 배율 | 과거 같은 축제의 학습 자료 |
| 명절 보정 [2] | 모양 0.85 (일요일 0.5, 8/15 0.6), 역별 차이 0.5배 | 검증 A(2025 추석 되짚기) RMSE: 보정 전 0.4524 → 0배 0.3622 / 0.5배 0.3478 / 1배 0.3547 |
| 공연장·인접역 [2] | 행사장 역 0.4, 인접역 0.7 | 사전에 정한 값 (행사장 역은 모델에 행사 피처가 있어 약하게) |
| 검단 [2] | 0.5배 | 사전에 정한 값 |
| SSG 종료 시간대 [2] | 종료 기준 −2~+2시간, 0.6배 | 2025-03 ~ 2026-06 124경기 회귀, 검증 A·B 경기 단위 leave-one-out (상승폭 RMSE A 1.003→0.73, B 0.758→0.60) |
| 팀 섞기 [3] | 0.5 / 0.5 | 사전에 정한 값 |
| 보정 복원 [4] | 문학경기장 SSG 종료·OKF 보정 243행을 섞기 전 세기로 | [3]에서 절반이 된 값을 되돌림 |
| 취소 행사 [5] | 상승분 전부 제거 / 같은 요일 평소 날 평균 | 언론 보도 (5절) |
| 검바위 시험 기간 [6] | 추정 시험일 9일의 12·11·13시 과거 증가 × 0.6, 반복 패턴 15·16시 +100명 | 학습 자료에서 평일 4~5일 연속 12시 3~5배 블록 5개 발견, 2025·2026년 4월 블록이 364일 간격으로 일치 → 2025년 7·10월 블록을 364일 옮겨 날짜 추정. 배율 0.6은 제출 전에 정한 보수 계수 (`code/geombawi_nayun/`) |
| 시험 후 뺀 보정 | 부천FC 종료 시간대, 토요일 오후 등 | 검증 근거로 만든 값을 한 번 제출해 Public이 나빠져 통째로 뺌. 계수를 바꿔 다시 제출하지 않음 |

### Public 점수 사용 범위
완성된 후보 중 제출할 것 고르기, 완성된 제출들의 섞기 비율 계산([1]), 시험해 본 보정을 남길지 뺄지 결정에만 썼다. 개별 모델 학습·피처 선택·후처리 계수 값에는 쓰지 않았다. 전체 수준을 올려 Public 반응을 본 탐색 제출(`X3_ev_hol_p015` 등)은 최종 경로에 쓰이지 않았다 (섞기 원본이 json 비율 그대로이고 상수 이동이 없음을 확인).

## 4. 개별 모델 (섞기 입력 8개)

| 모델 | 담당 | 저장 예측 (`SINCON_2026_요모조모_kdbwls/artifacts/models/`) | 다시 만드는 코드 |
|---|---|---|---|
| v7 | kdbwls | `v7.csv` | `code/sincon_pipeline_v7.py --mode submit` (`run_all.py --retrain`) |
| 역별 LightGBM | kdbwls | `ps15.csv`, `ps15_lvl.csv` | `code/scripts/run_per_station_final.py ps15` (`run_all.py --retrain`), 수준 맞춤은 `run_all.py` |
| new_v5 | kdbwls | `new_v5data.csv` | `code/feat_v5a_package.zip` 의 README: `train_final_kfold.py ... 3 0.1 0.1` + 성장률 +0.0081 (약 40분) |
| F_v10_stack | nayun | `F_v10_stack.csv` | `SINCON_최종제출_nayun/`에서 `python src/make_F_v10.py` (재료 재학습 순서는 그 폴더 README 1절) |
| nayun_v3 | nayun | `nayun_v3.csv` | `SINCON_최종제출_nayun/`에서 `python src/reproduce_v3.py` |
| dh_v2 | Dohyeong | `dh_v2.csv` | `SINCON_2026_요모조모_Dohyeong/code/dh_models/train.py` (출력 중 `submission_v2.csv`) |
| dh_mlp | Dohyeong | `dh_mlp.csv` | `SINCON_2026_요모조모_Dohyeong/code/dh_models/`에서 `export_processed_data.py` → `train_mlp_ensemble.py` (저장 예측과 파일이 같음을 확인) |

- 최종 경로(`run_all.py`)는 위 저장 예측에서 시작한다. Dohyeong 코드는 같은 폴더에 `sincon-2026/`(대회 원본, `setup_raw_data.py`가 배치), `sincon_추가데이터/`·`sincon_추가데이터2/`(외부 자료)를 두고 실행하며, `train_mlp_ensemble.py`는 PyTorch(GPU 권장)를 쓴다.
- dh_v2는 `train.py` 계열로 만든 저장 예측을 그대로 제출한다. 현재 코드를 다시 돌린 결과가 저장 파일과 같은지는 확인하지 못했다.

## 5. 데이터 출처 및 생성 기록

### 대회 제공 자료 (변형 없음)
| 파일 | SHA-256 |
|---|---|
| train.csv | `1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc` |
| test.csv | `d257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa` |
| metaData.csv | `9870dc6c9cbe7e6fc810bbe589dd9e06f309bb173a268a1bd63478f430785380` |
| sample_submission.csv | `8c5e10d4201460ddff2d78bf10a229ec8b604e86ae2f18bcd96969093981c93d` |

공개 저장소에는 넣지 않았다. `python setup_raw_data.py <폴더>`가 위 해시를 확인하며 필요한 위치(`SINCON_2026_요모조모_kdbwls/data/raw/`, `SINCON_최종제출_nayun/data/raw/`, `SINCON_2026_요모조모_Dohyeong/code/dh_models/sincon-2026/`)에 복사한다. 줄바꿈이 CRLF로 바뀐 사본은 LF로 되돌려 확인한다.

### 외부 자료 (최종 경로, 전체 표는 `SINCON_2026_요모조모_kdbwls/provenance/DATA.md`)
| 자료 | 기간 | 출처 | 이용 조건 | 조회일 |
|---|---|---|---|---|
| 공휴일·대체공휴일 | 2025-01 ~ 2026-10 | 한국천문연구원 월력요항, 고용노동부(2026-04-06)·인사혁신처(2026-04-29) 보도자료 | 정부 발표 날짜 사실 | 2026-11-11 |
| 날씨 (일·시간 강수, 기온) | 2025-01 ~ 2026-10 | Open-Meteo Historical Weather API, 인천 37.456N 126.705E | CC BY 4.0 | 2026-11-11 |
| SSG 문학 홈경기 일정·관중·개시와 종료 시각 | 2025-03 ~ 2026-10 | KBO https://www.koreabaseball.com (경기일정·관중 현황·게임센터) | 날짜·시각·수치만 사용 | 2026-11-11~12 |
| 행사(축구·공연·축제) | 2025-02 ~ 2026-10 | K리그, 인천시·연수구 보도자료, 예매처 공지 | 사실 정보 | 2026-11-11~12 |
| 검단 입주 세대 | 2024-11 ~ 2026-10 | 단지별 입주 공고·언론 보도 | 사실 정보 | 2026-11-11 |
| 인천대공원 계절·행사 | 2025-01 ~ 2026-10 | 인천대공원 공지 | 사실 정보 | 2026-11-11 |
| 검단연장 개통·시운전 | 2025 | 언론 보도, metaData first_date | 사실 정보 | 2026-11-11 |
| 인천대 학기 | 2025-01 ~ 2026-10 | 인천대 학사일정 | 공공 정보 | 2026-11-11 |
| 인천교통공사 월보 (역별 월 승차) | **2023·2024년** | 인천교통공사 수송 통계 | 공공 자료 | 2026-11-11 |
| 취소 행사 | 2026-09, 10 | iMBC 연예 2026-09-15 https://enews.imbc.com/News/ViewAmp/518304 · 아시아투데이 2026-10-21 https://www.asiatoday.co.kr/kn/view.php?key=20261021010007097 | 인용 | 2026-11-13 |
| 검바위 보정용 달력·행사 목록 | 2023 ~ 2026 | 위 자료를 합친 가공 파일 (`data/external_geombawi/`, 생성 코드 `code/geombawi_nayun/build_calendar.py`) | - | 2026-11-13 |

- 승차 인원이 들어 있는 외부 파일은 2023·2024년 월보와 거기서 계산한 계절 모양뿐이다. **2026-07 이후 지하철 승하차 자료는 어떤 파일에도 없고 쓰지 않았다.**

### 수집했지만 최종 경로에 쓰지 않은 자료 (해당 없음)
| 자료 | 비고 |
|---|---|
| 역 주변 버스 정류장 월별 승하차 (2026-07~09 포함, 지하철 아님) | 버스 보정 후보를 따로 만들어 제출해 보았으나 최종 경로에는 쓰지 않았다. 최종 경로 코드는 이 자료를 읽지 않는다 |
| 운행 장애, 시설 개장·폐장, 환승역 행사, 초중고 방학 | 검증에서 이득이 없어 피처로 쓰지 않음 (`DATA.md` 3절) |

## 6. 모델 및 중간 산출물

| 산출물 | 위치 | 재생성 |
|---|---|---|
| 개별 모델 예측 8개 | `SINCON_2026_요모조모_kdbwls/artifacts/models/` | 4절 |
| 섞기 원본 OPT4_raw·OPT5_raw | `artifacts/blend_chain/` | 실제 파일 (비율 계산 방식은 3절, `code/scripts/blend_public*.py`) |
| OPT5_pp, OPT5pp_team50, fix_cancel | `artifacts/blend_chain/` | `run_all.py` (저장본과 차이 0) |
| OPT5pp_team50_fix | `artifacts/blend_chain/` | 실제 파일 (당시 바로 계산, 스크립트 없음) |
| nayun 후처리본 `team_blend_opt4_75_v6pp_25` | `artifacts/blend_chain/` | nayun: OPT4_final_geo 0.75 + nayun v6 후처리본 0.25 (log1p) |

트리 모델 파일은 저장하지 않고 같은 코드·시드로 다시 학습한다. 사전학습 모델·임베딩·외부 가중치는 쓰지 않았다. 중간 산출물 해시: `SINCON_2026_요모조모_kdbwls/provenance/SHA256SUMS.txt`.

## 7. Kaggle Notebook / Google Colab
`run_on_kaggle_colab.ipynb`: 저장소 받기 → 패키지 확인(없는 것만 설치) → 대회 원본 배치 → `run_all.py` → 결과 확인. Kaggle은 대회 데이터를 노트북에 추가하면 `/kaggle/input/`에서 원본을 자동으로 찾고, Colab은 원본 4개를 `/content/raw/`에 올린다. `RETRAIN_CPU=True`로 바꾸면 kdbwls 모델 재학습까지 한다. GPU 불필요, 메모리 4GB 이하.

## 8. 기존 개발 기록
- `SINCON_2026_요모조모_kdbwls/records/`: 실행 로그(`run_log.txt`, `run_log_retrain.txt`, `run_log_py310.txt`), 피처 버전별 검증 결과(`results_v*.json`), 실험 기록 목록과 Public 점수 이력(`EXPERIMENT_LOG.md`)
- `SINCON_최종제출_nayun/outputs/SCORES.md` (후보별 근거와 Public 점수), `docs/README_작업로그.md`, `docs/규칙_점검_2026-11-13.md`
- Kaggle 노트북 「2026 SINCON v6」 Version 1

## 9. 제출 전 자체 점검
- [x] 최종 예측 파일과 실제 사용한 코드, 모델, 설정의 관계를 명시했습니다. (0절, 4절, `run_all.py`가 단계마다 저장본과 비교)
- [x] 데이터와 모델 작성표에 사용 항목을 모두 기록하고 미사용 항목은 해당 없음으로 표시했습니다. (5절, 6절, `provenance/DATA.md`)
- [x] 제공해드린 자료부터 최종 예측까지 필요한 경로, 명령, 의존성, 시드를 설명했습니다. (빠른 재현, 2절, 4절)
- [x] 공개 데이터에서 평가 정답 또는 관련 자료를 열람, 확보, 사용한 사실: 2026-07 이후 지하철 승하차 인원을 열람·확보·사용한 사실이 없습니다. 외부 파일을 전수 점검해 2026-07~10 지하철 승차·하차 행이 0개임을 확인했습니다. 운영진 공지(공개 데이터 중 정답 포함 자료 존재) 이후 사용 자료를 다시 점검했습니다.
