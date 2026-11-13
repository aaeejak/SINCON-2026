# SINCON 2026 — 인천 지하철 승차 인원 예측

인천 1·2·7호선 71개 역, 06~24시 18개 시간대의 승차 인원을 예측한다
(학습 2025-01-01 ~ 2026-06-30, 예측 2026-07-01 ~ 10-31). 평가지표는 RMSLE.

## 폴더 구조
```
data/
  raw/          대회 제공 데이터 (train, test, sample_submission, metaData)
  external/     밖에서 수집한 원자료 (src/fetch_*.py, src/merge_team_data.py) + 출처 기록
    events_master.csv              검증된 행사 275건 (행마다 출처·검증 방법)
    station_master.csv             역 정보 (노선, 개통일, 환승)
    past_ridership_2023_2024_monthly.csv  2023·2024년 7~10월 역·시간대별 승차 (인천교통공사 월보)
    past_ridership_correction_2025.csv    2025 train ÷ 월보 보정비율
    events_reference_unverified.csv, events_2023_2024_reference.csv  참고용 (모델 제외)
    school_list.csv, school_vacation_*.csv  인천 초·중·고 방학 (NEIS)
    ssg_home_games_2025_2026.csv   KBO 공식 경기 일정 (SSG 홈경기, 취소 포함, 행사 검증용)
    ssg_preseason_2025_2026.csv    KBO 시범경기 (문학 14일)
    bucheon_fc_home_games.csv      K리그 공식 경기 일정 (부천FC, 행사 검증용)
    weather_hourly.csv             인천 시간별 날씨 (Open-Meteo)
    gumdan_population_monthly.csv  검단 동별 월별 인구 (행정안전부)
    SOURCES.md                     외부 자료 출처·이용조건·조회일
  derived/      우리 코드가 만든 가공 파일 (src/build_*.py로 언제든 다시 생성)
    calendar_features.csv          날짜별 달력·방학·날씨·인구 피처
    station_events.csv             역·날짜별 행사 표
  MERGE_REPORT.md               팀원 데이터 통합·검증 내역 (2026-11-12)
  processed/    모델 입력 피처 (src/preprocess.py로 생성, 용량이 커서 필요하면 다시 생성)
    train_features.parquet, test_features.parquet
데이터_정리본.md                   팀 공유용 데이터 설명서 (대회 데이터 제외)
share/sincon_추가데이터.zip        팀 공유용 묶음 (짧은 설명서 + external + derived, 대회 데이터 제외)
src/
  calendar_data.py         공휴일·연휴 구분·계절 구간 (공용)
  merge_team_data.py       팀원 자료 검증·통합 (승차량·공식 일정으로 행사 확인)
  fetch_school_vacation.py NEIS 방학 수집 (본인 인증키로 직접 실행)
  fetch_weather.py         날씨 수집
  fetch_population.py      검단 동별 월별 인구 수집 (행정안전부)
  build_calendar.py        달력 피처 생성
  build_event_days.py      역·날짜별 행사 표 생성
  preprocess.py            모델용 피처 생성 (make_dataset: 검증·최종 예측 공용)
  baseline.py              베이스라인 (역·요일·시간별 최근 N주 log1p 평균)
  train_lgbm.py            LightGBM 설정 비교 → 최종 학습 → 제출 파일 생성
  make_kaggle_notebook.py  src/ 코드로 Kaggle 제출 노트북 2개 생성
kaggle/
  sincon_baseline.ipynb    Kaggle 노트북: 베이스라인 (자동 생성, 직접 수정 금지)
  sincon_lgbm.ipynb        Kaggle 노트북: LightGBM (자동 생성, 직접 수정 금지, Kaggle에서 미검증)
outputs/                   결과 폴더 (자세한 설명: outputs/README.md)
  models/                  우리 모델 예측 (lgbm_v3, lgbm_v9, catboost_v9, mlp_v9, ours_v9, baseline, cv_*.json)
  candidates/              제출 후보 (cand_A ~ cand_I)
  archive/                 이전 단계·중복·로그
```

## 실행 방법 (로컬)
Python 3.14.3, 패키지 버전은 `requirements.txt` (pandas 3.1.1, numpy 2.5.4, lightgbm 4.7.0,
scikit-learn 1.9.1). 모든 명령은 프로젝트 루트에서 실행.
베이스라인은 랜덤 요소가 없다. LightGBM 모델은 `random_state=42`로 고정한다.
Kaggle 노트북의 패키지 버전은 로컬과 다를 수 있으므로, 노트북 첫 셀에서 버전을 출력해 기록한다.
```
pip install -r requirements.txt
python src/fetch_weather.py         # (선택) 날씨 다시 받기
python src/fetch_school_vacation.py # (선택) 방학 다시 받기, NEIS 인증키 필요
python src/fetch_population.py      # (선택) 검단 인구 다시 받기
python src/merge_team_data.py --src "<팀원 자료 폴더>"  # (선택) 팀원 자료 검증·통합 → data/external/
python src/build_calendar.py        # data/derived/calendar_features.csv 생성
python src/build_event_days.py      # data/derived/station_events.csv 생성
python src/preprocess.py            # data/processed/{train,test}_features.parquet 생성
python src/baseline.py              # outputs/submission_baseline.csv 생성 (157,194행)
python src/train_lgbm.py            # outputs/submission_lgbm.csv, cv_lgbm.json 생성 (약 6분)
python src/make_kaggle_notebook.py  # kaggle/ 노트북 2개 생성
```

## 초안 제출 (Kaggle 노트북)
1. 코드를 고쳤다면 `python src/make_kaggle_notebook.py`로 노트북을 다시 만든다.
2. kaggle.com/competitions/sincon-2026 → **Code → New Notebook**
   → **File → Import Notebook**에서 `kaggle/sincon_baseline.ipynb`(베이스라인) 또는
   `kaggle/sincon_lgbm.ipynb`(LightGBM) 업로드.
3. 오른쪽 **Input**에 대회 데이터(`sincon-2026`)가 붙어 있는지 확인. 없으면 **Add Input**으로 추가.
   - LightGBM 노트북은 **추가 데이터도 필요**하다: `share/sincon_추가데이터.zip`을 비공개 Kaggle Dataset으로
     올려 두 번째 입력으로 붙인다 (노트북이 `/kaggle/input` 아래에서 파일을 자동으로 찾는다).
4. **Save Version → Save & Run All (Commit)**. 실행이 끝나면 `/kaggle/working/submission.csv` 생성.
5. 버전 화면의 **Output → Submit to Competition**으로 제출하고 Public 점수 확인.
   - 제출은 팀 합산 하루 10회 (한국시간 오전 9시 초기화).

## 최종 제출 (ZIP)
전체 코드 + 입력 데이터 + 최종 예측 CSV를 하나의 ZIP으로 묶어 I-FOSS에 등록한다.
```
Compress-Archive -Path README.md, requirements.txt, src, data, kaggle, outputs -DestinationPath sincon-2026_final.zip -Force
```
→ oss.inha.ac.kr에서 I-FOSS 게시글 작성(ZIP 첨부) → I-KAGGLE에서 그 게시글 선택 (조장, 팀당 1개).

## 데이터 사용 규칙
- **2026-11-11 변경**: 7월 이후 외부 데이터(행사·날씨·학사일정)도 사용한다.
  OT 자료 p.13(예측 기간 날씨 실측·행사 결과 금지)에서 바뀐 기준이다.
- 사용 금지: **2026-07 이후 지하철 승하차 실적** (예측 대상).
- 외부 자료는 `data/external/SOURCES.md`에 이용조건·출처·조회일을 기록한다.
- NEIS 인증키는 발급자 본인만 사용한다 (이용약관 제7조). 코드·제출 ZIP에 넣지 않는다.

## 방법론
### 베이스라인 (`src/baseline.py`)
- 예측값 = 같은 역·요일유형·시간대의 최근 W주 승차 인원을 log1p 공간에서 평균 → expm1.
  공휴일은 일요일로 취급. 선택적으로 작년 같은 시기의 계절 비율(log 공간)을 곱한다.
- 후보 24개(W ∈ {4, 8, 12} × 평균/중앙값 × 요일 3분류/7분류 × 계절 보정 유무)를 RMSLE로 비교.
- 검증 구간 (예측 기간처럼 학습 직후 4개월):
  - fold1: ~2025-06-30 학습 → 2025-07~10 검증 (예측 기간과 같은 계절)
  - fold2: ~2026-02-28 학습 → 2026-03~06 검증
  - 기록이 4주 미만인 신규 개통 역은 검증에서 제외
- 결과: **8주 · log1p 평균 · 요일 7분류 · 계절 보정** 선택, 검증 RMSLE **0.2355**
  (fold1 0.2399, fold2 0.2312). OT 공식 베이스라인 자체 검증은 0.3077.

### 전처리 (`src/preprocess.py`)
- `make_dataset(cutoff, target_start, target_end)`: cutoff까지 학습, 그 뒤 구간(또는 test)을 예측 대상으로
  피처를 만든다. 검증과 최종 예측이 같은 함수를 쓴다.
- 타깃: `y = log1p(boardings)` (RMSLE에 맞춤).
- 피처 42개: 시간(hour, dow, month, 공휴일·연휴, 설·추석 구분 `holiday_type`, 어버이날, 계절 구간 코드),
  역(station_id, line, 환승, 신규 역 개통 후 일수, time_idx), 외부(방학, 대학 학기, 시간별·일별 날씨,
  역별 행사 여부·종류·시작 후 경과 시간, 검단 인구), 타깃 통계(역×요일×시간 등 log 평균, 하차/승차 비율).
- 행사는 `events_master.csv`(검증된 행사)에서 온다. 팀원 자료 통합 후 행사 종류(`event_type`: 야구,
  공연, 부천FC, 맥주축제, 펜타포트, 기타 축제)를 피처로 추가했다.
- 누수 방지: 타깃 통계는 cutoff 이전 데이터로만, 학습 행에는 자기 달을 뺀 값(월 단위 out-of-fold).
  시차 피처 없음. `spike` 행사일은 피처로 쓰지 않음.
  검단 인구는 검단 7개 역에만, 날짜 단위 행사 열은 제외(역 단위 행사만 사용).
### LightGBM (`src/train_lgbm.py`)
검증은 베이스라인과 같은 두 구간을 쓰고, 설정마다 early stopping으로 반복 수를 정한다.
점수는 `outputs/cv_lgbm.json`에도 저장된다. 단계별 결과는 `*_v1`, `*_v2` 파일에 남겨 두었다.

| 단계 | 피처 | 최고 검증 RMSLE | fold1 | fold2 |
|---|---|---|---|---|
| v1 통합 전 | 38 | 0.1902 | 0.2032 | 0.1773 |
| v2 팀원 데이터 통합 (행사 244건) | 42 | **0.1850** | 0.1961 | 0.1739 |
| v3 KBO 시범경기·취소 경기 추가 (행사 275건) — Kaggle Public **0.16087** | 42 | 0.1855 | 0.1987 | 0.1723 |
| v4 행사 종료 시각·인접역 피처 + 최종 3시드 평균 | 47 | 0.1840 | 0.1978 | 0.1702 |
| v5 v4 + 이상치 역·날짜 181개 제외 + 계양·귤현 개통 전 제외 | 47 | 0.1833 | 0.1968 | 0.1698 |
| **v6 v5 + 환승역·외부환승·대학역 열 + 3설정 평균 (현재)** | 50 | **0.1824** | 0.1946 | 0.1701 |

v2와 v3의 차이(0.0005)는 설정 선택의 흔들림 수준이라 새 라벨이 점수를 바꾸지는 않았다.
v4는 행사 종료 시각(`hours_from_end`), 인접역 행사 피처(`adj_event_*`: 아시아드 공연 후 서구청역 등)를 더했고,
최종 모델을 시드 3개(42, 7, 2026) 평균으로 만든다. 설정은 min_data_in_leaf 200 계열 3개만 비교한다.

**v3 (2026-11-12, 1,231초)** — 설정별 결과

| num_leaves | min_data_in_leaf | lambda_l2 | RMSLE | fold1 | fold2 |
|---|---|---|---|---|---|
| **255** | **200** | **5** | **0.1855** | 0.1987 | 0.1723 |
| 127 | 200 | 5 | 0.1856 | 0.1997 | 0.1716 |
| 63 | 50 | 0 | 0.1861 | 0.2012 | 0.1711 |
| 511 | 200 | 10 | 0.1864 | 0.1988 | 0.1740 |
| 127 | 50 | 0 | 0.1875 | 0.2035 | 0.1715 |
| 255 | 50 | 0 | 0.1889 | 0.2049 | 0.1729 |

- 선택: num_leaves 255, min_data_in_leaf 200, lambda_l2 5, learning_rate 0.05, feature/bagging fraction 0.8, seed 42.
- 베이스라인과 섞기(log 공간): 비중 0 → 0.1855, 0.1 → 0.1860, 0.2 → 0.1876, 0.4 → 0.1942.
  섞을수록 나빠져 **LightGBM 단독**을 쓴다.
- 최종 모델: 2026-06-30까지 전체로 1,091회 학습 → `outputs/submission_lgbm.csv`
  (157,194행, ID 순서 일치, 음수·무한값 없음을 코드가 검사).
- 피처 중요도(gain)는 과거 통계(`te_st_dt_h`, `te_st_off_h`)가 대부분이고, 그다음 `station_id`, `hour`,
  `hours_from_event`, `time_idx`, `te_st` 순.
- 예측 확인: 9/28은 평일 수준(47.9만), 7/17은 공휴일 수준(28.0만), 문학경기장 공연일 9/12·9/13은 17~23시
  승차가 평소의 9.4배·6.1배, 취소 경기일(8/5, 8/6, 8/22)은 평소의 1.6~2.3배.
- **Kaggle Public 점수 (2026-11-12 제출, v3): 0.16087.** 검증 RMSLE(0.1855)와 다른 구간·행이라 직접 비교는 안 되고, 같은 날 팀 제출 중 가장 낮았다 (팀원 최고 0.16215, 이전 GitHub 제출 0.19527, 대회 베이스라인 노트북 0.28264).
- **한계**: 검증은 학습 데이터 안의 두 구간이라 Public·Private 점수와 다를 수 있다.
  외부 피처가 점수에 얼마나 기여하는지는 피처를 하나씩 빼는 비교를 하지 않아 모른다.
  행사 시각이 시작 시각 기준이라 이른 시간에 시작하는 페스티벌(8/29, 10/24 등)의 저녁 승차를 약하게
  예측한다 (평소의 5배 안팎, 과거 공연은 61~162배). 자세한 내용은 `data/MERGE_REPORT.md`.

## 제출 체크리스트
- [x] 예측 CSV: `example_id, boardings`, **157,194행**, ID 누락·중복 없음, 0 이상 유한수 (`outputs/submission_lgbm.csv`, 코드가 자동 검사)
- [x] Kaggle에 제출해 Public 점수 확인 (v3, 0.16087). v4 이후는 제출 전
- [ ] 전체 코드 + 입력 데이터 + 최종 출력 포함 (ZIP)
- [ ] I-FOSS 게시글 등록 → I-KAGGLE에서 게시글 선택 (조장, 팀당 1개)
- [ ] 마감 2026-11-13(금) 21:00

## 남은 작업
- 행사 종료 시각(`end_hour`) 피처 추가 (공연·페스티벌의 저녁 승차 예측 보강, `data/MERGE_REPORT.md` 참고)
- 외부 피처 기여도 확인 (피처 그룹별로 빼고 검증 점수 비교)
- 최종 ZIP 만들기 전에 `data/processed/`(재생성 가능)를 포함할지 정하고, README·SOURCES.md 최종 점검

### v5 (이상치 제외)
행사 목록으로 설명되지 않는 급증 역·날짜 181개(`data/derived/outlier_days.csv`, 같은 역·시각·평일/휴일 ±28일 중앙값 3배 이상)와 검단 연장 개통(2025-06-28) 전 계양(3110)·귤현(3111) 기록을 학습에서 뺐다. 검증 대상 행은 그대로다. 개선 폭은 작다(0.1840→0.1833). 이 목록은 승차량으로 만든 것이라 피처가 아니라 학습 데이터 정리에만 쓴다.

### v7 (CatBoost 추가)
`src/train_catboost.py`: CatBoost(depth 8, lr 0.06, 800회, 시드 2개)를 v6와 log 공간 0.2 비중으로 섞어 `submission_lgbm_cat.csv`(v7)를 만든다.
검증에서 0.1958→0.1950, 0.1714→0.1707 (CatBoost 단독은 0.205/0.178).
시험했지만 채택하지 않은 것: 행사 관람객 수 피처(7~10월 모사 구간 0.1948→0.2001로 악화, 야구 관람객 자료 없음), 같은 계절 행 가중치(0.1698 그대로).

### 시험했지만 채택하지 않은 것: 기준선 위 잔차 LightGBM (`src/train_residual.py`)
직전 8주 (역, 요일, 시간) 평균을 피처로 넣고 그 위의 잔차를 학습. 3~6월 구간 검증: 단독 0.1822(기존 LightGBM 0.1714), 두 예측을 섞어도 이득 없음(0.1714→0.1714~0.1739). 7~10월 모사 구간은 학습 기간이 너무 짧아 시험하지 못했다.

### 시험했지만 채택하지 않은 것: 역별 성장률 보정
최근 8주 대 1년 전 같은 기간의 역별 log 평균 차이(성장률)로 예측 수준을 보정. 3~6월 구간 검증: 모델 잔차와의 상관 0.24, 기울기 0.08~0.11, 교차 적합 보정 후 RMSLE 0.1714→0.1716~0.1718(개선 없음). 7~10월 모사 구간은 1년 전 기록이 없어 시험하지 못했다.

### 보조 모델 추가 시험 (임베딩 MLP, 일 합계×시간대 비율)
- `src/train_mlp.py` 임베딩 MLP(역·시간·요일·월·노선 임베딩 + 숫자 피처, 기준값 te_st_dt_h 위의 잔차): 단독 0.2139 / 0.1932(LightGBM 0.1958 / 0.1714). LightGBM과 log 공간 0.1 비중으로 섞으면 0.1958→0.1955, 0.1714→0.1708. 규제를 강화한 변형은 더 나빴다. 최종 모델 v8 = v6 0.72 + CatBoost 0.18 + MLP 0.10 (`outputs/submission_lgbm_cat_mlp.csv`).
- `src/train_twostage.py` 일 평균 수준(1단계) + 시간대 편차(2단계) 분해: 단독 0.2091 / 0.1793. 섞은 이득이 한 구간에서만 있고(0.1714→0.1706) 다른 구간은 악화(0.1958→0.1961)라 채택하지 않았다.
- 지하철 정시표: 인천교통공사 1호선 첫차·막차 표를 확인했으나 평일·휴일 차이가 1분 안팎이고 역마다 고정된 값이라 `station_id`가 이미 담고 있는 정보여서 시험하지 않았다. (2·7호선 시각표는 구하지 못함)

### v9 (날씨 항목 확장)
Open-Meteo에서 습도·구름량·일사량(+체감온도·풍속·날씨코드)을 추가로 받아(`python src/fetch_weather.py extra` → `data/external/weather_extra_hourly.csv`, CC BY 4.0, 조회 2026-11-12) 부분집합별로 검증했다. 습도·구름량·일사량 3개를 넣었을 때만 안정적으로 좋아졌다(두 구간 평균 0.1831→0.1814; 체감온도·풍속·날씨코드는 효과 없거나 악화). 피처 53개, LightGBM 3설정 평균 검증 RMSLE 0.1824→0.1808. 이 모델과 CatBoost·MLP를 다시 학습해 v9로 만들었다.
시험했지만 채택하지 않은 것: 야구 경기별 관람객 수(공식 경기별 기록을 자동으로 구하지 못함), 대학 학기(이미 `univ_semester`로 반영돼 있음).

### 작년 같은 요일 앵커 (`src/yoy_anchor.py`)
(역, 날짜, 시간)마다 364·371·357일 전 같은 역·시간의 log 승차를 0.5:0.25:0.25로 평균하고 역별 성장분(최근 8주 대 1년 전 8주)을 더한 예측. 공휴일 구분이 어긋나는 lag는 버리고, 1년 전 기록이 없는 역(검단 신설)은 제외. 학습이 없는 방식이다.
3~6월 구간 검증: 앵커 단독 0.212, LightGBM과 0.15 비중 혼합 0.1681→0.1659 (4월 0.1929→0.1875 등 4개월 중 3개월 개선). 7~10월 모사 구간은 1년 전 기록이 없어 검증 불가. 예측 대상 행의 93.5%에 앵커 값이 있다.

### 앵커 스태커 (`src/stack_anchor.py`)
fold2(학습 ~2026-02, 예측 3~6월)의 모델 예측 pb와 작년 앵커(행사·공휴일·이상치 게이트, 3lag·7lag 두 종류)로 작은 LightGBM이 pb의 잔차(y-pb)를 학습하고, 예측 대상에는 기준 예측에 이 보정을 shrink 0.7로 더한다.
시간 순서 검증(앞 달로 학습→뒤 달 평가): 5~6월 0.1620→0.1578, 4월 0.1929→0.1834, 6월 0.1614→0.1531. 이득 일부(5~6월)는 앵커 없이도 생겨 최근 라벨로 모델 편향을 고친 효과일 수 있고, 앵커가 있어야 생기는 이득은 4월이다. 7~10월에는 1년 전 기록이 없어 검증하지 못했다.

## 최종 제출 파일 (cand_K, Public 0.151321)
`src/make_final.py`가 `outputs/models/`의 우리 모델 예측과 `final/input_team_*.csv`(팀원 제출 파일)로부터 `outputs/final_K_0.151321.csv`를 다시 만든다 (원본과 log 거리 4e-16). 구성은 스크립트 첫 부분 주석 참고. 팀원 쪽 파일은 팀원 코드로 재현해야 한다.

## 후처리 (팀원 Dohyeong 코드, `src/postprocess/`, `src/apply_postprocess.py`)
섞은 파일에 적용: (1) 축제 보정 — 모델이 모르는 2026 7~10월 소규모 축제 5종(송도해변·계양 가을꽃·부평풍물·능허대·드림파크)의 시간대별 로그 증가를 더함, (2) 명절 보정 — 2026 추석 9/24~27과 8/15(토)를 대회 train.csv의 과거 명절(2025 추석·설, 2026 설) 실적과 비교해 보정(역별 차이 0.5배).
확인: K는 추석 9/24~26 일 합계를 일요일의 0.95/0.77/0.95배로 예측했지만 실제 과거 명절은 0.73/0.57/0.75(2025 추석), 0.73/0.67/0.85(2026 설)였다. 보정 후 0.77/0.63/0.80으로 과거 패턴에 가까워진다.
팀원 파일 `final/team_datasets_README.md`, `final/team_blend_weights_OPT.json`, `data/external/festival_list_final_team.csv`에 근거 기록. 팀원 보고 Public: 섞기 0.151361 → 축제 보정 약 0.1503 → 명절 보정 0.148837 → 역별 차이 0.5배 0.148399 (역별 0.5배는 팀원이 점수를 보고 고른 값이라 일부 낙관적일 수 있음).
재현 순서: `python src/make_final.py` (K) → `python src/apply_postprocess.py outputs/final_K_0.151321.csv outputs/final_KP_postproc.csv`.

## 검단 개통 보정 (`src/geomdan_adjust.py`, 팀원 분석 메모 `final/team_geomdan_note.md`)
검단 신설 3개 역(3107·3108·3109)과 계양(3110)의 7~10월 평일 평균 수준을 `2026-06 실제 × 성숙역 계절(7월 -0.069, 8월 -0.084, 9월 -0.004, 10월 +0.004) × 입주 세대 증가(탄력성 0.5, 검단 3개 역만)`로 잡은 목표와 비교해, 로그 차이의 0.5배를 그 역·그 달 모든 행에 더한다.
팀원 원본 스크립트는 받지 못해 메모대로 재구현했고, 팀원 최고본(OPT)에 적용했을 때 예측 평일 평균(검단호수공원 1,155·1,120·1,259·1,323 등)과 보정 범위(검단호수공원 +0.05~+0.09 등)가 메모와 일치함을 확인했다. 세대 증가율의 분모 17,500은 메모의 "10월까지 +30% 안팎"에 맞춘 가정이다.
순서: `make_final.py` → `apply_postprocess.py` → `geomdan_adjust.py`.

## SSG 경기 종료 시각 보정 (`src/fit_ssg_profile.py`, `src/ssg_adjust.py`, 데이터 `data/external/ssg_munhak_game_times.csv`)
KBO 게임센터(koreabaseball.com)에서 문학 홈경기 159경기(2025-03~2026-10)의 개시·종료·관중을 가져왔다(조회 2026-11-13, 승하차 실적 아님). 모델은 종료를 시작+3.2시간으로 추정해 실제 종료 시간대(rel 0, 1)의 문학경기장 승차를 +0.67, +0.52(log1p) 과소 예측했고(두 검증 구간 모두 같은 방향), 경기 단위 leave-one-out으로 RMSE가 0.581에서 0.529로 줄었다. 2026-07~10월 35경기 136행에 0.7배로 더한다.
순서: `make_final.py` → `apply_postprocess.py` → `geomdan_adjust.py` → `ssg_adjust.py`.

## 우리 모델 혼합 비율 재조정 (`src/tune_ours_weights.py`)
검증 두 구간에서 v9 피처로 LightGBM(3설정 평균)·CatBoost·MLP·XGBoost를 학습해 예측을 저장하고(`outputs/tune_ours/`), log1p 공간 최적 비율을 구했다 (Public 점수 미사용). 단독 RMSLE는 LightGBM 0.1931/0.1685, XGBoost 0.1983/0.1697, CatBoost 0.2044/0.1758, MLP 0.2201/0.2039.
기존 0.72/0.18/0.10(LightGBM/CatBoost/MLP)은 0.1937/0.1696으로 LightGBM 단독보다 나빴다. 최적은 LightGBM 0.80, XGBoost 0.13, CatBoost 0.06, MLP 0 (0.1930/0.1681). MLP는 두 구간 교차에서도 0이었다. 적용: LightGBM 0.80 + XGBoost 0.12 + CatBoost 0.08.

## 역 주변 버스 이용량 보정 (`src/bus_adjust.py`, 데이터 `data/external/station_bus_month_2024_2026_v2.csv`)
팀원 제공 파일: 71개 역 반경 300·500m 버스 정류장의 월별 승하차(2024-01~2026-09, 인천·경기). 지하철 승하차가 아니라 버스 자료. 출처·라이선스·조회일은 전달받지 못해 **팀원에게 확인해 기록해야 한다**(규칙: 외부 자료는 출처·조회일 기록).
2026-04~06 평균 대비 7~9월 버스 일평균의 로그 변화에서 역 간 평균을 뺀 값 dc에 k=0.25를 곱해 그 달 모든 행에 더한다 (10월은 자료가 없어 그대로). 검증(두 구간 교차): 역×월 잔차와 상관 0.60~0.65, 전체 역 기준 0.1681→0.1650, 0.1948→0.1934. 단 검단 3역·계양·귤현을 빼면 이득이 사라진다(0.1668, 0.1950).

## 추가 데이터 검색 결과 (2026-11-13, `data/external/events_found_2026_search.csv`)
삼산체육관: 대한항공 홈은 계양체육관, KOVO컵은 여수(10/11~25), 리그 개막 10/31 → 7~10월 삼산체육관 배구 경기 없음. 송도컨벤시아: 세계한상대회 9/28~30 확인했으나 2025년 센트럴파크·국제업무지구에 평일 컨벤션형 급증이 없어 보정 근거 없음. 대학 축제(인천대·인하대·연세대 송도 등) 2026 날짜와 공연 종료 시각은 찾지 못함. 보정 후보는 만들지 않음.

## 추가 데이터 수집 결과 (2026-11-13)
KBO 게임센터에서 SSG 문학 159경기의 점수·연장 여부 등(`data/external/ssg_munhak_game_results.csv`)을 받아 종료 시각 보정 후 남은 문학경기장 오차를 검증했으나 이득이 약 0.0001이라 보정에 넣지 않았다. 인천 생활인구·상권 월별 자료, 7~10월 입주 물량(송도 0가구), 1·2호선 운영 변경 공지는 쓸 수 있는 자료를 찾지 못했다.

### SSG 점수 차 보정 (`src/fit_ssg_margin.py`, `ssg_adjust.py ... margin`)
종료 시각 보정 후 남은 문학경기장 잔차를 |점수 차|로 보정(rel -1: c -0.357, d +0.089 / rel 1: c +0.564, d -0.141 등, `data/derived/ssg_margin_profile.csv`). 접전이면 종료 후 승차 집중, 큰 점수 차면 종료 전 이탈. 경기 단위 leave-one-out RMSE 0.5212 → 0.5139. 후보: `cand_AJ_OPT4_ssg_margin.csv`.

## 검바위 급증의 원인 (2026-11-13)
검바위(3208) 12시 승차가 평일 4~5일 연속 평소(약 130명)의 3~5배(+450명)가 되는 기간이 2025-04-28~05-01, 07-03~09, 10-16~21, 12-17~23, 2026-04-27~30에 있고, 하루 합계는 변하지 않는다. 다른 역은 같은 기간에 변화가 없다 → 인근 학교 시험일(일찍 하교)로 추정. 2025와 2026 4월이 364일(같은 요일)로 정확히 일치해 예측 구간 시험일을 364일 이동한 2026-07-02·03·06·07·08, 10-15·16·19·20으로 추정했다(학교 일정으로 확인하지 못함). 보정 `src/exam_adjust.py`: 그 날 12시 예측 +450 × 0.6.
전 역을 다시 훑었을 때 같은 유형의 반복 구간은 검바위뿐이었다 (부천종합운동장 3~4월, 인천대공원 4월 등은 봄 행사로 7~10월과 무관).
