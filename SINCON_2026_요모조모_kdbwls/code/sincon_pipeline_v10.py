"""SINCON 2026 - 인천 지하철 승차 인원 예측 파이프라인 v7 (v6 + 2023·2024 월보 역x시간 월 계절 모양·시간대 비중)
(v3: 날씨·시간대별 비·경기 관중·행사·검단 입주 + v4: 인천대공원 계절·행사, 환승역 축제, 노이즈역 분리 학습, 과거 월보 계절성)

실행:  python sincon_pipeline.py --data_dir ./data --mode cv      # 검증만
       python sincon_pipeline.py --data_dir ./data --mode submit  # 검증 + 최종 학습 + submission.csv

Kaggle Notebook / Colab: 이 파일 내용을 셀에 붙여 넣고 DATA_DIR만 바꿔 main()을 호출해도 됩니다.
필요 파일: train.csv, test.csv, metaData.csv, holidays_kr_2025_2026.csv,
           external/ 아래: weather_incheon_2025_2026.csv, rain_blocks_incheon.csv (Open-Meteo),
           kbo_munhak_2025_2026.csv, kbo_munhak_crowd.csv (KBO), events_2025_2026.csv (행사), geomdan_movein.csv (검단 입주),
           park_3226_features.csv, transfer_events_2025_2026.csv, past_monthly_2023_2024.csv (scripts/build_v4_external.py로 생성),
           ops_daily_features.csv, ops_incident_hourly.csv (scripts/build_ops_features.py로 생성),
           school_calendar_2025_2026.csv (NEIS 학사일정 방학 비율·인천대 학기, 팀 추가 데이터 calendar_features.csv에서 추출),
           monthly_prior_2023_2024.csv (인천교통공사 월보 2023·2024 역x시간 월 계절 모양, 팀 sincon_splits.zip에서 계산)

규칙(변경 후): 예측 기간의 승차 데이터만 쓰지 않으면 외부 데이터 사용 가능.
승차 기반 특징(최근 수준, 상대팀 인기도)은 기준일 이전 데이터로만 계산합니다.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42
EXT_DIR = Path("external")


# ----------------------------------------------------------------------------
# 1. 달력 특징
# ----------------------------------------------------------------------------
def make_calendar(holiday_csv, start="2025-01-01", end="2026-10-31"):
    hol = pd.read_csv(holiday_csv, parse_dates=["date"])
    cal = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    cal["month"] = cal["date"].dt.month
    cal["dow"] = cal["date"].dt.dayofweek
    cal["doy"] = cal["date"].dt.dayofyear
    cal = cal.merge(hol[["date", "type"]].rename(columns={"type": "htype"}), on="date", how="left")
    cal["is_holiday"] = cal["htype"].notna().astype(int)
    cal["is_myeongjeol"] = cal["htype"].isin(["seollal", "chuseok"]).astype(int)
    cal["is_labor_day_2025"] = (cal["date"] == "2025-05-01").astype(int)
    off = ((cal["dow"] >= 5) | (cal["is_holiday"] == 1)).astype(int)
    cal["is_offday"] = off
    cal["day_type"] = np.select([cal["dow"].eq(6) | cal["is_holiday"].eq(1), cal["dow"].eq(5)], [2, 1], 0)
    cal["is_before_offday"] = ((off == 0) & (off.shift(-1, fill_value=0) == 1)).astype(int)
    cal["is_after_offday"] = ((off == 0) & (off.shift(1, fill_value=0) == 1)).astype(int)
    cal["is_bridge_day"] = ((off == 0) & (off.shift(1, fill_value=0) == 1) & (off.shift(-1, fill_value=0) == 1)).astype(int)
    block = (off != off.shift()).cumsum()
    cal["offday_block_len"] = off.groupby(block).transform("size") * off
    cal["offday_block_pos"] = (off.groupby(block).cumcount() + 1) * off
    # 명절 연휴 전후 평일 (귀성·귀경)
    mj = cal["is_myeongjeol"]
    cal["near_myeongjeol"] = ((mj.shift(1, fill_value=0) + mj.shift(-1, fill_value=0) + mj.shift(2, fill_value=0)
                               + mj.shift(-2, fill_value=0)) > 0).astype(int) * (1 - mj)
    return cal.drop(columns=["htype"])


# ----------------------------------------------------------------------------
# 2. 기준일 이전 데이터로 만드는 '최근 수준' 특징
# ----------------------------------------------------------------------------
def recent_profiles(hist, cutoff, weeks=8):
    """기준일 직전 weeks주 동안의 log1p 평균. 공휴일은 day_type=2(일요일 패턴)로 묶임."""
    start = pd.Timestamp(cutoff) - pd.Timedelta(weeks=weeks) + pd.Timedelta(days=1)
    w = hist[(hist["date"] >= start) & (hist["date"] <= cutoff)]
    p_dt = w.groupby(["station_id", "hour", "day_type"])["y"].mean().rename(f"lvl_dt_{weeks}w")
    p_dow = w[w["is_holiday"] == 0].groupby(["station_id", "hour", "dow"])["y"].mean().rename(f"lvl_dow_{weeks}w")
    p_st = w[w["day_type"] == 0].groupby("station_id")["y"].mean().rename(f"lvl_st_{weeks}w")
    return p_dt, p_dow, p_st


def add_profiles(df, hist, cutoff):
    out = df
    for wk in (8, 26):
        p_dt, p_dow, p_st = recent_profiles(hist, cutoff, wk)
        out = out.merge(p_dt, on=["station_id", "hour", "day_type"], how="left")
        out = out.merge(p_dow, on=["station_id", "hour", "dow"], how="left")
        out = out.merge(p_st, on="station_id", how="left")
    # 짧은 이력 역(26주가 안 되는 신규역)은 8주 값으로 채움
    for a, b in [("lvl_dt_26w", "lvl_dt_8w"), ("lvl_dow_26w", "lvl_dow_8w"), ("lvl_st_26w", "lvl_st_8w")]:
        out[a] = out[a].fillna(out[b])
    out["lvl_dow_8w"] = out["lvl_dow_8w"].fillna(out["lvl_dt_8w"])
    out["lvl_dow_26w"] = out["lvl_dow_26w"].fillna(out["lvl_dt_26w"])
    return out


# ----------------------------------------------------------------------------
# 3. 모델
# ----------------------------------------------------------------------------
FEATS = ["station_id", "line", "hour", "dow", "day_type", "month", "doy", "is_holiday", "is_myeongjeol",
         "is_labor_day_2025", "is_before_offday", "is_after_offday", "is_bridge_day", "offday_block_len",
         "offday_block_pos", "near_myeongjeol"]
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED, num_threads=0)


LEARNER = "lgb"   # "lgb" 또는 "xgb" (같은 피처·같은 잔차 목표, 학습기만 바꿈)
XGB_PARAMS = dict(objective="reg:squarederror", eta=0.05, max_depth=9, min_child_weight=40, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", max_cat_to_onehot=1, max_bin=256, nthread=0)


XGB_OBJ = "log_resid"
PER_STATION = False
SPLIT_OFFDAY = False
PER_STATION_PARAMS = dict(num_leaves=31, min_data_in_leaf=20, learning_rate=0.05)
PER_STATION_ROUNDS = 400
LGB_OBJ = "regression"   # "poisson" / "tweedie" 면 원 단위 학습
TWEEDIE_P = 1.2


def fit_predict_xgb(Xtr, ytr, w, Xpr, n_rounds, obj=None, bm=None):
    import xgboost as xgb
    def prep(X):
        X = X.copy()
        for c in X.columns:   # 범주형은 station_id만 (full에서 이미 category), 나머지는 실수
            if c != "station_id" and not np.issubdtype(X[c].dtype, np.number):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X
    dtr = xgb.DMatrix(prep(Xtr), ytr, weight=w, enable_categorical=True, base_margin=None if bm is None else bm[0])
    dpr = xgb.DMatrix(prep(Xpr), enable_categorical=True, base_margin=None if bm is None else bm[1])
    p = 0
    for i in range(N_SEEDS):
        bst = xgb.train({**XGB_PARAMS, **({"objective": obj} if obj else {}), "seed": SEED + i}, dtr, num_boost_round=n_rounds)
        p = p + bst.predict(dpr) / N_SEEDS
    return p


def fit_predict_lgb(train_df, pred_df, n_rounds, valid_df=None):
    """잔차 학습: 목표 = log1p(y) - 같은 역·시간·요일유형의 26주 수준.
    학습 행의 수준은 '그 행 이전'이 아니라 학습 구간 전체에서 계산되므로, 모델은 달력에 따른 '편차'를 배웁니다."""
    F = FEAT_SUBSET or active_feats()
    if TRAIN_MONTHS:
        train_df = train_df[train_df["date"] > train_df["date"].max() - pd.DateOffset(months=TRAIN_MONTHS)]
    Xtr = train_df[F]
    ytr = train_df["y"] - train_df["base_tr"]
    w = None
    if HALF_LIFE:
        age = (train_df["date"].max() - train_df["date"]).dt.days.values
        w = 0.5 ** (age / HALF_LIFE)
    if LEARNER == "xgb":
        if XGB_OBJ == "log_resid":     # log1p 공간 잔차(기본): 목표 = log1p(y) - 기준
            return None, fit_predict_xgb(Xtr, ytr, w, pred_df[F], n_rounds) + pred_df["base"].values
        y_cnt = np.expm1(train_df["y"].values)
        if XGB_OBJ == "poisson":        # 원 단위 포아송, 기준(역·시간·요일) 수준을 시작점으로
            bm_tr = np.log(np.maximum(np.expm1(train_df["base_tr"].values), 0.5))
            bm_pr = np.log(np.maximum(np.expm1(pred_df["base"].values), 0.5))
            mu = fit_predict_xgb(Xtr, y_cnt, w, pred_df[F], n_rounds, obj="count:poisson", bm=(bm_tr, bm_pr))
            return None, np.log1p(mu)
        if XGB_OBJ == "raw_mse":        # 원 단위 제곱오차 (RMSLE와 맞지 않는 기준선)
            mu = fit_predict_xgb(Xtr, y_cnt, w, pred_df[F], n_rounds, obj="reg:squarederror")
            return None, np.log1p(np.clip(mu, 0, None))
    if LGB_OBJ in ("poisson", "tweedie"):   # 원 단위 포아송/트위디, 기준 수준을 시작점(init_score)으로
        bm_tr = np.log(np.maximum(np.expm1(train_df["base_tr"].values), 0.5))
        bm_pr = np.log(np.maximum(np.expm1(pred_df["base"].values), 0.5))
        dtr = lgb.Dataset(Xtr, np.expm1(train_df["y"].values), weight=w, init_score=bm_tr,
                          categorical_feature=[c for c in ["station_id"] if c in F], free_raw_data=False)
        prm = {**LGB_PARAMS, "objective": LGB_OBJ, **({"tweedie_variance_power": TWEEDIE_P} if LGB_OBJ == "tweedie" else {})}
        raw = 0
        for i in range(N_SEEDS):
            model = lgb.train({**prm, "seed": SEED + i}, dtr, num_boost_round=n_rounds)
            raw = raw + model.predict(pred_df[F], raw_score=True) / N_SEEDS
        return model, np.log1p(np.exp(raw + bm_pr))
    if SPLIT_OFFDAY:   # 출근일 / 쉬는 날을 따로 학습 (자전거 대회의 회원·비회원 분리와 같은 발상)
        off_tr = (train_df["dow"] >= 5) | (train_df["is_holiday"] == 1); off_pr = (pred_df["dow"] >= 5) | (pred_df["is_holiday"] == 1)
        p = np.zeros(len(pred_df))
        for mt, mp in ((off_tr.to_numpy(), off_pr.to_numpy()), (~off_tr.to_numpy(), ~off_pr.to_numpy())):
            d1 = lgb.Dataset(Xtr[mt], ytr[mt], weight=None if w is None else w[mt], categorical_feature=[c for c in ["station_id"] if c in F], free_raw_data=False)
            for i in range(N_SEEDS):
                p[mp] += lgb.train({**LGB_PARAMS, "seed": SEED + i}, d1, num_boost_round=n_rounds).predict(pred_df.loc[mp, F]) / N_SEEDS
        return None, p + pred_df["base"].values
    if PER_STATION:   # 역마다 따로 학습 (station_id 제외, 작은 트리)
        F2 = [c for c in F if c != "station_id"]; p = np.zeros(len(pred_df))
        prm = {**LGB_PARAMS, **PER_STATION_PARAMS}
        for sid in pred_df["station_id"].unique():
            mt = (train_df["station_id"] == sid).to_numpy(); mp = (pred_df["station_id"] == sid).to_numpy()
            if mt.sum() < 200:
                continue
            d1 = lgb.Dataset(train_df.loc[mt, F2], ytr[mt], weight=None if w is None else w[mt], free_raw_data=False)
            for i in range(N_SEEDS):
                p[mp] += lgb.train({**prm, "seed": SEED + i}, d1, num_boost_round=PER_STATION_ROUNDS).predict(pred_df.loc[mp, F2]) / N_SEEDS
        return None, p + pred_df["base"].values
    dtr = lgb.Dataset(Xtr, ytr, weight=w, categorical_feature=[c for c in ["station_id"] if c in F], free_raw_data=False)
    # 시드 여러 개로 학습해 평균 (feature_fraction·bagging 무작위성으로 생기는 흔들림을 줄임)
    p = 0
    for i in range(N_SEEDS):
        model = lgb.train({**LGB_PARAMS, "seed": SEED + i}, dtr, num_boost_round=n_rounds)
        p = p + model.predict(pred_df[F]) / N_SEEDS
    return model, p + pred_df["base"].values


def rmsle_log(pred_log, true_log):
    return float(np.sqrt(np.mean((np.clip(pred_log, 0, None) - true_log) ** 2)))


# ----------------------------------------------------------------------------
# 4. 한 fold 실행 (cutoff까지 학습, 그 뒤 기간 예측)
# ----------------------------------------------------------------------------
def run_fold(full, cutoff, horizon_end, n_rounds=600, verbose=True):
    cutoff = pd.Timestamp(cutoff)
    hist = full[full["date"] <= cutoff].copy()
    fut = full[(full["date"] > cutoff) & (full["date"] <= horizon_end)].copy()
    return predict_period(hist, fut, cutoff, n_rounds, verbose)


def active_feats():
    return (FEATS + (WEATHER_COLS if USE_WEATHER else []) + (GAME_COLS if USE_GAMES else []) + (RAIN_COLS if USE_RAIN else [])
            + (CROWD_COLS if USE_CROWD else []) + (EVENT_COLS if USE_EVENTS else []) + (GEOM_COLS if USE_GEOM else [])
            + (PARK_COLS if USE_PARK else []) + (XFER_COLS if USE_XFER else [])
            + (OPS_COLS if USE_OPS else []) + (RANK_COLS if USE_RANK else []) + (RANK_HOUR_COLS if USE_RANK_HOUR else [])
            + (VAC_COLS if USE_VAC else []) + (UNIV_COLS if USE_UNIV else []) + (MP_COLS if USE_MP else [])
            + (LY_COLS if USE_LY else []) + EXTRA_COLS)


def predict_period(hist, fut, cutoff, n_rounds, verbose=True):
    pop = opponent_popularity(hist)
    for df in (hist, fut):
        df["munhak_opp_pop"] = np.where(df["munhak_sched"] == 1, df["opponent"].map(pop).astype(float), np.nan)
        df.loc[(df["munhak_sched"] == 1) & df["munhak_opp_pop"].isna(), "munhak_opp_pop"] = 1.0
    fut = add_profiles(fut, hist, cutoff)
    if USE_RANK or USE_RANK_HOUR:
        r, hr = rank_features(hist)
        hist, fut = attach_ranks(hist, r, hr), attach_ranks(fut, r, hr)
    # 학습 데이터의 기준 수준: 학습 구간 전체의 역·시간·요일유형 평균(계절 편차를 모델이 배우도록)
    base_tr = hist.groupby(["station_id", "hour", "day_type"])["y"].mean().rename("base_tr")
    hist = hist.merge(base_tr, on=["station_id", "hour", "day_type"], how="left")
    fut = fut.merge(base_tr, on=["station_id", "hour", "day_type"], how="left")
    if USE_LY:
        hist, fut = last_year_features(hist, hist), last_year_features(hist, fut)
    for df_ in (hist, fut):
        df_["t_days"] = (df_["date"] - pd.Timestamp("2025-01-01")).dt.days
    if any(c in FAC_COLS for c in EXTRA_COLS + (FEAT_SUBSET or [])):
        hist, fut = facility_features(hist), facility_features(fut)
    if any(c in STYPE_COLS for c in EXTRA_COLS + (FEAT_SUBSET or [])):
        hist, fut = station_type_features(hist, hist), station_type_features(hist, fut)
    if any(c in SPATIAL_COLS for c in EXTRA_COLS + (FEAT_SUBSET or [])):
        hist, fut = spatial_lag_features(hist, hist), spatial_lag_features(hist, fut)

    preds = {}
    preds["v0_ot_baseline"] = fut["lvl_dow_8w"].values                  # OT 베이스라인과 같은 방식
    preds["v1_daytype_8w"] = fut["lvl_dt_8w"].values                    # 공휴일=일요일 패턴
    # LightGBM: 학습 구간 평균 기준 편차를 배운 뒤, 최근 26주 수준 + (최근 수준 - 학습 평균) 보정
    # 26주 수준이 없는 신규역은 8주 수준 사용
    fut["base"] = fut["base_tr"]
    _, p_lgb_raw = fit_predict_lgb(hist, fut, n_rounds)
    if NOISE_SPLIT:
        # 정상 역은 노이즈 역(경기장·공원 등 급증이 잦은 역)을 뺀 데이터로 따로 학습한 모델로 예측
        nh = ~hist["station_id"].astype(int).isin(NOISE_STATIONS)
        nf = ~fut["station_id"].astype(int).isin(NOISE_STATIONS).to_numpy()
        _, p_clean = fit_predict_lgb(hist[nh], fut[nf], n_rounds)
        p_lgb_raw = p_lgb_raw.copy()
        p_lgb_raw[nf] = p_clean
    preds["v2_lgb_trainlevel"] = p_lgb_raw
    # 최근 수준으로 이동: 평일 기준 역 수준 차이(최근 8주 - 학습 전체)를 더함
    st_shift = (fut["lvl_st_8w"] - hist[hist["day_type"] == 0].groupby("station_id")["y"].mean()
                .reindex(fut["station_id"]).values)
    preds["v3_lgb_recentshift"] = p_lgb_raw + st_shift.fillna(0).values
    preds["v4_blend"] = 0.5 * preds["v1_daytype_8w"] + 0.5 * preds["v3_lgb_recentshift"]
    preds["v5_lgb_avg"] = 0.5 * preds["v2_lgb_trainlevel"] + 0.5 * preds["v3_lgb_recentshift"]
    # 검단 3개 역: 기준일 이후 늘어난 입주 세대(로그)의 절반만큼 올림 (트리 모델은 학습 범위 밖 증가를 못 따라감, 검증 B로 0.5 선택)
    hh_c = hist.loc[hist["date"] == hist["date"].max(), "geomdan_hh_log"].max()
    g = fut["station_id"].astype(int).isin(GEOMDAN).to_numpy()
    preds["v6_final"] = preds["v5_lgb_avg"] + np.where(g, 0.5 * (fut["geomdan_hh_log"].to_numpy() - hh_c), 0)   # 최종 사용
    # 과거 월보(2023·2024) 역별 7~10월 계절 모양을 더함 (학습 구간에 7~10월이 없을 때만 의미가 있음)
    if PRIOR_BETA:
        preds["v6_prior"] = preds["v6_final"] + PRIOR_BETA * fut["prior_si"].fillna(0).to_numpy()

    scores = {}
    if "y" in fut and fut["y"].notna().all():
        for k, p in preds.items():
            scores[k] = rmsle_log(np.nan_to_num(p, nan=fut["y"].mean()), fut["y"].values)
        if verbose:
            print(f"  cutoff {cutoff.date()} -> {fut['date'].min().date()}~{fut['date'].max().date()} "
                  f"({fut['date'].nunique()}일, {fut['station_id'].nunique()}역)")
            for k, v in scores.items():
                print(f"    {k:22s} RMSLE {v:.5f}")
    return fut, preds, scores


# ----------------------------------------------------------------------------
# 5. main
# ----------------------------------------------------------------------------
WEATHER_COLS = ["precip_mm", "precip_hours", "tmax"]
GAME_COLS = ["munhak_sched", "munhak_game", "munhak_ngames", "munhak_start", "munhak_rel_hour", "munhak_opp_pop"]
RAIN_COLS = ["rain_now", "rain_prev", "rain_06_09", "rain_17_20"]
CROWD_COLS = ["munhak_crowd_log"]
EVENT_COLS = ["ev_kind", "ev_start", "ev_rel_hour"]
GEOM_COLS = ["geomdan_hh_log"]
USE_WEATHER, USE_GAMES, USE_RAIN, USE_CROWD, USE_EVENTS, USE_GEOM = True, True, True, True, True, True
PARK_COLS = ["park_foliage_w", "park_foliage_days_from_peak", "park_cherry_w", "park_event_scale", "park_event_active",
             "park_event_exit", "park_waterpark_open", "park_waterpark_hour"]
XFER_COLS = ["xf_kind", "xf_active", "xf_rel_hour"]
# 실험 결과(검증 A·B·C, 3시드 평균)로 고른 설정: 인천대공원 피처만 켬.
# 환승역 축제·노이즈역 분리·과거 월보 계절성은 점수가 좋아지지 않아 꺼 둠 (켜려면 True / PRIOR_BETA=0.25)
USE_PARK, USE_XFER = True, False
NOISE_SPLIT = False
NOISE_STATIONS = [3127, 3754, 3226, 3209, 3122, 3210, 3139]
PRIOR_BETA = 0.0
N_SEEDS = 3
OPS_COLS = ["ops_days_since_open", "ops_l1_ext_open", "ops_l1_trial_run", "ops_is_terminal", "ops_headway_am_plan",
            "ops_headway_off_plan", "ops_inc_hour", "ops_inc_recovery_hour", "ops_inc_primary"]
RANK_COLS = ["rank_board_recent", "rank_pct_recent", "rank_board_all", "rank_board_weekday", "rank_board_offday",
             "rank_alight_recent", "rank_tier", "share_board_recent", "offday_weekday_ratio", "board_growth_log_8w", "line_seq"]
RANK_HOUR_COLS = ["rank_board_hour"]
# 실험 결과(검증 A·B·C, 3시드 평균): 운영 개요는 1호선 검단연장 개통·시운전 두 열만, 역 순위는 역 단위만 사용
OPS_COLS = ["ops_l1_ext_open", "ops_l1_trial_run"]
USE_OPS, USE_RANK, USE_RANK_HOUR = True, True, False
# 방학: 인천 초·중·고 557곳 중 방학인 학교 비율(NEIS 학사일정), 인천대 학기 중 여부(인천대 학사일정)
VAC_COLS = ["vac_elem", "vac_middle", "vac_high", "vac_all"]
UNIV_COLS = ["univ_semester"]
# 실험(검증 A·B·C, 3시드): 초·중·고 방학 비율은 B·C를 나쁘게 해서 끄고, 인천대 학기만 사용 (검단 3역 제외)
USE_VAC, USE_UNIV = False, True
VAC_SKIP_GEOMDAN = True
# 인천교통공사 월보 2023·2024 (팀 분할 자료 sincon_splits.zip): 역x시간 월 계절 모양, 역 월 계절 모양, 시간대 비중
# 실험(3시드): A 0.2479→0.2418, C 0.1739→0.1716, B 0.1663→0.1672 → 7~10월 검증 기준으로 역x시간·역 계절 모양 2열만 사용
MP_COLS = ["mp_sh", "mp_st"]
USE_MP = True
# 작년 같은 시기(364일 전 ±7일, 같은 평일/휴일 구분) 같은 역·시간대 수준 - 학습 평균 (기준일 이전 기록만 사용)
LY_COLS = ["ly_dev", "ly_st_dev"]
USE_LY = True
EXTRA_COLS = []          # 실험: ["t_days"], 공간 피처 SPATIAL_COLS 등
TRAIN_MONTHS = None      # 실험: 최근 N개월만 학습 (None = 전체)
HALF_LIFE = None         # 실험: 최근 데이터 가중치 반감기(일) (None = 같은 가중치)
FEAT_SUBSET = None       # 실험: 모듈 모델용 피처 목록 (None = 전체)
SPATIAL_COLS = ["own_lag123", "nb_lag123", "line_lag123"]
# 역 주변 대형 시설 개장·폐장 (뉴스·공공기관 자료, external/facility_changes_2025_2026.csv)
FAC_DROP = set()
EVENTS_FILE = "events_2025_2026.csv"
FAC_COLS = ["fac_retail_open_d", "fac_retail_close_d", "fac_school_open_d"]


def facility_features(df):
    fc = pd.read_csv(EXT_DIR / "facility_changes_2025_2026.csv", parse_dates=["date"])
    fc = fc[~fc.station_id.isin(FAC_DROP)]
    d = df.assign(sid=df["station_id"].astype(int))
    for kind, change, col in (("retail", "open", "fac_retail_open_d"), ("retail", "close", "fac_retail_close_d"), ("school", "open", "fac_school_open_d")):
        ev = fc[(fc.kind == kind) & (fc.change == change)].groupby("sid" if False else "station_id")["date"].min()
        start = d["sid"].map(ev)
        days = (d["date"] - start).dt.days
        d[col] = np.where(days >= 0, np.minimum(days, 180), -1)      # 변화 후 경과 일수(180일 상한), 변화 전·해당 없음 = -1
    return d.drop(columns=["sid"])


# 역 성격(주거·업무·상권) 지표: 기준일 이전 승하차로 계산 (평일 기준)
STYPE_COLS = ["st_am_board_sh", "st_pm_board_sh", "st_am_alight_ratio", "st_pm_alight_ratio", "st_mid_sh", "st_night_sh", "st_offday_ratio", "st_board_alight"]


def station_type_features(hist, df):
    h = hist.assign(sid=hist["station_id"].astype(int))
    wd = h[h["day_type"] == 0]
    B = wd.groupby(["sid", "hour"])[["boardings", "alightings"]].mean().unstack()
    b, a = B["boardings"], B["alightings"]; bt = b.sum(1); at = a.sum(1)
    am, pm, mid, night = [6, 7, 8], [17, 18, 19], [10, 11, 12, 13, 14, 15], [21, 22, 23]
    f = pd.DataFrame({
        "st_am_board_sh": b[am].sum(1) / bt,                       # 출근 시간 승차 비중 (주거지일수록 큼)
        "st_pm_board_sh": b[pm].sum(1) / bt,                       # 퇴근 시간 승차 비중 (업무지구일수록 큼)
        "st_am_alight_ratio": np.log((a[am].sum(1) + 1) / (b[am].sum(1) + 1)),   # 출근 시간 하차/승차 (업무지구 +)
        "st_pm_alight_ratio": np.log((a[pm].sum(1) + 1) / (b[pm].sum(1) + 1)),   # 퇴근 시간 하차/승차 (주거지 +)
        "st_mid_sh": b[mid].sum(1) / bt,                           # 낮 시간 비중 (상권·학교)
        "st_night_sh": b[night].sum(1) / bt,                       # 밤 시간 비중 (유흥·상권)
        "st_board_alight": np.log(bt / at),
    })
    day = h.groupby(["sid", "date", "day_type"])["boardings"].sum().reset_index()
    f["st_offday_ratio"] = np.log(day[day.day_type > 0].groupby("sid").boardings.mean() / day[day.day_type == 0].groupby("sid").boardings.mean())
    f = f.reset_index().rename(columns={"index": "sid"})
    d = df.assign(sid=df["station_id"].astype(int)).drop(columns=[c for c in STYPE_COLS if c in df.columns])
    return d.merge(f, on="sid", how="left").drop(columns=["sid"])


def spatial_lag_features(hist, df, lag=123):
    """역 하루 수준(log, 학습 평균 대비)을 28일 이동평균으로 만든 뒤 lag일 전 값을 붙임: 자기 역, 같은 노선 인접역(±2), 노선 전체.
    lag=123이면 예측 기간 끝(기준일+123일)까지 기준일 이전 기록만 씀."""
    h = hist.assign(sid=hist["station_id"].astype(int))
    day = h.groupby(["sid", "date"])["y"].mean().unstack(0)
    day = day.reindex(pd.date_range(day.index.min(), day.index.max()))
    dev = day - day.mean()
    roll = dev.rolling(28, min_periods=7).mean()
    roll.index = roll.index + pd.Timedelta(days=lag)
    line = h.groupby("sid")["line"].first()
    nb = pd.DataFrame(index=roll.index)
    for s_ in roll.columns:
        same = sorted(line[line == line[s_]].index)
        i = same.index(s_); nbs = [x for x in same[max(0, i - 2): i + 3] if x != s_]
        nb[s_] = roll[nbs].mean(axis=1)
    ln = pd.DataFrame({s_: roll[line[line == line[s_]].index].mean(axis=1) for s_ in roll.columns})
    def st(x, name): return x.stack().rename(name).reset_index().rename(columns={"level_0": "date", "level_1": "sid"})
    feats = st(roll, "own_lag123").merge(st(nb, "nb_lag123"), on=["date", "sid"], how="outer").merge(st(ln, "line_lag123"), on=["date", "sid"], how="outer")
    d = df.assign(sid=df["station_id"].astype(int)).drop(columns=[c for c in SPATIAL_COLS if c in df.columns])
    return d.merge(feats, on=["date", "sid"], how="left").drop(columns=["sid"])


def last_year_features(hist, df):
    h = hist[["station_id", "hour", "date", "day_type", "y"]].copy()
    h["sid"] = h["station_id"].astype(int); h["off"] = (h["day_type"] > 0).astype(int)
    out = []
    idx = pd.date_range(h["date"].min(), h["date"].max())
    for off in (0, 1):
        x = h[h["off"] == off].pivot_table(index="date", columns=["sid", "hour"], values="y").reindex(idx)
        r = x.rolling(15, center=True, min_periods=2).mean()
        st = h[h["off"] == off].groupby(["date", "sid"])["y"].mean().unstack().reindex(idx).rolling(15, center=True, min_periods=2).mean()
        r.index = r.index + pd.Timedelta(days=364); st.index = st.index + pd.Timedelta(days=364)
        a = r.stack(["sid", "hour"], future_stack=True).rename("ly").reset_index().rename(columns={"level_0": "date", "index": "date"})
        b = st.stack(future_stack=True).rename("ly_st").reset_index().rename(columns={"level_0": "date", "index": "date"})
        a = a.merge(b, on=["date", "sid"], how="left"); a["off"] = off
        out.append(a.dropna(subset=["ly"]))
    ly = pd.concat(out)
    d = df.assign(sid=df["station_id"].astype(int), off=(df["day_type"] > 0).astype(int))
    d = d.drop(columns=[c for c in ("ly", "ly_st", "ly_dev", "ly_st_dev") if c in d.columns]).merge(ly, on=["date", "sid", "hour", "off"], how="left")
    stbase = hist.assign(sid=hist["station_id"].astype(int)).groupby("sid")["y"].mean().rename("st_base")
    d = d.merge(stbase, on="sid", how="left")
    d["ly_dev"] = d["ly"] - d["base_tr"]; d["ly_st_dev"] = d["ly_st"] - d["st_base"]
    return d.drop(columns=["sid", "off", "ly", "ly_st", "st_base"])
GEOMDAN = [3107, 3108, 3109]
MUNHAK = 3127


def game_daily(games_csv):
    g = pd.read_csv(games_csv, parse_dates=["date"])
    g["start_hour"] = g["time"].str.slice(0, 2).astype(int) + g["time"].str.slice(3, 5).astype(int) / 60
    g["played"] = ~g["remark"].str.contains("취소|사정")
    d = g.groupby("date").agg(munhak_sched=("opponent", "size"), opponent=("opponent", "first"),
                              series=("series", "first"), sched_start=("start_hour", "min"))
    p = g[g["played"]].groupby("date").agg(munhak_ngames=("opponent", "size"), played_start=("start_hour", "min"))
    d = d.join(p, how="left")
    d["munhak_sched"] = 1
    d["munhak_ngames"] = d["munhak_ngames"].fillna(0)
    d["munhak_game"] = (d["munhak_ngames"] > 0).astype(int)
    d["munhak_start"] = d["played_start"].fillna(d["sched_start"])
    return d[["munhak_sched", "munhak_game", "munhak_ngames", "munhak_start", "opponent", "series"]].reset_index()


def add_external(full, weather_csv, games_csv):
    w = pd.read_csv(weather_csv, parse_dates=["date"])
    full = full.merge(w[["date"] + WEATHER_COLS], on="date", how="left")
    g = game_daily(games_csv)
    full = full.merge(g, on="date", how="left")
    is_m = full["station_id"].astype(int).eq(MUNHAK)
    for c in ["munhak_sched", "munhak_game", "munhak_ngames"]:
        full[c] = np.where(is_m, full[c].fillna(0), 0)
    full["munhak_start"] = np.where(is_m, full["munhak_start"], np.nan)
    # 경기 시작 후 몇 시간째인지 (경기 끝나고 귀가하는 시간대를 직접 표시)
    full["munhak_rel_hour"] = np.where(full["munhak_sched"] == 1, full["hour"] - full["munhak_start"], -99)
    full["opponent"] = np.where(is_m, full["opponent"], np.nan)
    return full


def add_external_v3(full):
    hour = full["hour"]
    # 1) 시간대별 비: 그 시각이 속한 구간(출근·낮·퇴근·밤)의 강수량과 바로 앞 구간 강수량
    r = pd.read_csv(EXT_DIR / "rain_blocks_incheon.csv", parse_dates=["date"])
    full = full.merge(r, on="date", how="left")
    blk = ["rain_06_09", "rain_10_16", "rain_17_20", "rain_21_23"]
    full[blk] = full[blk].fillna(0)
    b = np.select([hour <= 9, hour <= 16, hour <= 20], [0, 1, 2], 3)
    arr = full[blk].to_numpy()
    full["rain_now"] = arr[np.arange(len(full)), b]
    full["rain_prev"] = np.where(b > 0, arr[np.arange(len(full)), np.maximum(b - 1, 0)], 0)
    # 2) 문학경기장 경기 관중 수 (KBO 일자별 관중 현황, 더블헤더는 합산, 포스트시즌은 매진 23,000 가정)
    c = pd.read_csv(EXT_DIR / "kbo_munhak_crowd.csv", parse_dates=["date"]).groupby("date")["crowd"].sum()
    is_m = full["station_id"].astype(int).eq(MUNHAK)
    crowd = full["date"].map(c)
    crowd = crowd.where(crowd.notna(), np.where((full["series"] == "postseason") & (full["munhak_game"] == 1), 23000, np.nan))
    crowd = np.where(full["munhak_game"] == 1, crowd, 0)
    full["munhak_crowd_log"] = np.where(is_m, np.log1p(pd.Series(crowd).astype(float)), 0)
    # 3) 행사 (부천FC·송도 축제·문학/아시아드 공연): 해당 역에만 표시
    e = pd.read_csv(EXT_DIR / EVENTS_FILE, parse_dates=["date"])
    e["kind_code"] = e["kind"].map({"football": 1, "festival": 2, "concert": 3})
    e = e.sort_values("start_hour").drop_duplicates(["date", "station_id"])
    full = full.merge(e[["date", "station_id", "kind_code", "start_hour"]].rename(
        columns={"kind_code": "ev_kind", "start_hour": "ev_start"}).assign(station_id=lambda d: d.station_id.astype(int)),
        left_on=["date", full["station_id"].astype(int)], right_on=["date", "station_id"], how="left",
        suffixes=("", "_ev")).drop(columns=["station_id_ev", "key_1"], errors="ignore")
    full["ev_kind"] = full["ev_kind"].fillna(0)
    full["ev_rel_hour"] = np.where(full["ev_kind"] > 0, full["hour"] - full["ev_start"], -99)
    # 4) 검단 입주: 단지별 입주 시작 후 60일 동안 선형으로 입주한다고 보고 누적 세대 수 (검단 3개 역에만)
    g = pd.read_csv(EXT_DIR / "geomdan_movein.csv", parse_dates=["start_date"])
    dates = pd.Series(sorted(full["date"].unique()))
    cum = sum(np.clip((dates - s).dt.days / 60, 0, 1) * h for s, h in zip(g["start_date"], g["households"]))
    hh = pd.Series(cum.values, index=dates)
    full["geomdan_hh_log"] = np.where(full["station_id"].astype(int).isin(GEOMDAN), np.log1p(full["date"].map(hh)), 0)
    return full


def add_external_v4(full):
    sid = full["station_id"].astype(int)
    # 1) 인천대공원(3226) 단풍·벚꽃 가중치, 시민의 날, 물놀이장
    pk = pd.read_csv(EXT_DIR / "park_3226_features.csv", parse_dates=["date"])
    full = full.assign(_sid=sid).merge(pk.rename(columns={"station_id": "_sid"}), on=["date", "_sid", "hour"], how="left")
    pc = [c for c in pk.columns if c.startswith("park_")]
    full[pc] = full[pc].fillna(0)
    full.loc[full["_sid"] != 3226, "park_foliage_days_from_peak"] = 30
    # 2) 환승역 축제(계양 국화·워터축제, 부평풍물대축제, 드림파크 국화축제) + 부천FC 경기 인접역(까치울·춘의)
    xe = pd.read_csv(EXT_DIR / "transfer_events_2025_2026.csv", parse_dates=["date"])
    xe["xf_kind"] = xe["kind"].map({"fest": 1, "football": 2})
    xe = xe.sort_values("start_hour").drop_duplicates(["date", "station_id"])
    full = full.merge(xe[["date", "station_id", "xf_kind", "start_hour", "end_hour"]].rename(columns={"station_id": "_sid"}),
                      on=["date", "_sid"], how="left")
    full["xf_kind"] = full["xf_kind"].fillna(0)
    full["xf_active"] = ((full["xf_kind"] > 0) & (full["hour"] >= full["start_hour"]) & (full["hour"] <= full["end_hour"] + 1)).astype(int)
    full["xf_rel_hour"] = np.where(full["xf_kind"] > 0, full["hour"] - full["start_hour"], -99)
    full = full.drop(columns=["start_hour", "end_hour"])
    # 3) 과거 월보 계절 모양: 역별 (월 일평균 로그 - 그 해 7~10월 평균) 의 2023·2024 평균
    pm = pd.read_csv(EXT_DIR / "past_monthly_2023_2024.csv")
    pm["l"] = np.log(pm["boardings_total"] / pm["days"])
    pm["si"] = pm["l"] - pm.groupby(["year", "station_id"])["l"].transform("mean")
    si = pm.groupby(["station_id", "month"])["si"].mean().rename("prior_si").reset_index().rename(columns={"station_id": "_sid"})
    full = full.merge(si, on=["_sid", "month"], how="left")
    return full.drop(columns=["_sid"])


def add_ops(full):
    """운영 개요: 검단연장 개통·시운전 기간, 계획 배차, 종착역 여부, 운행 장애 시간대 (문서 상수로 재구성)."""
    od = pd.read_csv(EXT_DIR / "ops_daily_features.csv", parse_dates=["date"]).rename(columns={"station_id": "_sid"})
    oi = pd.read_csv(EXT_DIR / "ops_incident_hourly.csv", parse_dates=["date"]).rename(columns={"station_id": "_sid"})
    full = full.assign(_sid=full["station_id"].astype(int)).merge(od, on=["date", "_sid"], how="left")
    full = full.merge(oi.drop(columns=["ops_inc_delay_min"]), on=["date", "_sid", "hour"], how="left")
    full[["ops_inc_hour", "ops_inc_recovery_hour", "ops_inc_primary"]] = full[["ops_inc_hour", "ops_inc_recovery_hour", "ops_inc_primary"]].fillna(0)
    return full.drop(columns=["_sid"])


def rank_features(hist, weeks=8):
    """노선별 역 이용 순위 (기준일 이전 승하차 기록만 사용). 같은 노선 역끼리 순위, 1 = 승차 최다."""
    h = hist.assign(sid=hist["station_id"].astype(int), off=(hist["day_type"] > 0).astype(int))
    d = h.groupby(["sid", "line", "date", "off"], observed=True)[["boardings", "alightings"]].sum().reset_index()
    end = d["date"].max()
    rec = d[d["date"] > end - pd.Timedelta(weeks=weeks)]
    prv = d[(d["date"] <= end - pd.Timedelta(weeks=weeks)) & (d["date"] > end - pd.Timedelta(weeks=2 * weeks))]
    g = pd.DataFrame({
        "b_rec": rec.groupby("sid")["boardings"].mean(), "a_rec": rec.groupby("sid")["alightings"].mean(),
        "b_all": d.groupby("sid")["boardings"].mean(),
        "b_wd": rec[rec["off"] == 0].groupby("sid")["boardings"].mean(), "b_off": rec[rec["off"] == 1].groupby("sid")["boardings"].mean(),
        "b_prv": prv.groupby("sid")["boardings"].mean(), "n": d.groupby("sid")["date"].nunique()})
    g["line"] = d.groupby("sid")["line"].first()
    g.loc[g["n"] < 28, ["b_rec", "a_rec", "b_all", "b_wd", "b_off"]] = np.nan   # 이력 28일 미만은 순위 없음
    r = pd.DataFrame(index=g.index)
    for c, src in [("rank_board_recent", "b_rec"), ("rank_board_all", "b_all"), ("rank_board_weekday", "b_wd"),
                   ("rank_board_offday", "b_off"), ("rank_alight_recent", "a_rec")]:
        r[c] = g.groupby("line")[src].rank(ascending=False)
    nline = g.groupby("line")["b_rec"].transform("count")
    r["rank_pct_recent"] = (r["rank_board_recent"] - 1) / (nline - 1)
    r["rank_tier"] = pd.cut(r["rank_pct_recent"], [-0.01, 0.15, 1 / 3, 2 / 3, 1.01], labels=False)
    r["share_board_recent"] = g["b_rec"] / g.groupby("line")["b_rec"].transform("sum")
    r["offday_weekday_ratio"] = g["b_off"] / g["b_wd"]
    r["board_growth_log_8w"] = np.log(g["b_rec"] / g["b_prv"])
    r["line_seq"] = g.groupby("line").cumcount() + 1
    # 시간대별 순위 (최근 8주, 평일/휴일 × 시간)
    hr = h[h["date"] > end - pd.Timedelta(weeks=weeks)].groupby(["sid", "line", "off", "hour"], observed=True)["boardings"].mean().rename("bh").reset_index()
    hr["rank_board_hour"] = hr.groupby(["line", "off", "hour"])["bh"].rank(ascending=False)
    return r.reset_index().rename(columns={"index": "sid"}), hr[["sid", "off", "hour", "rank_board_hour"]]


def attach_ranks(df, r, hr):
    df = df.drop(columns=[c for c in RANK_COLS + RANK_HOUR_COLS if c in df.columns])
    df = df.assign(sid=df["station_id"].astype(int), off=(df["day_type"] > 0).astype(int))
    df = df.merge(r, on="sid", how="left").merge(hr, on=["sid", "off", "hour"], how="left")
    return df.drop(columns=["sid", "off"])


def opponent_popularity(hist, k=5):
    """기준일 이전 문학경기장역 정규시즌 경기일 승차로 상대팀 인기도 계산 (1.0 = 보통)."""
    m = hist[(hist["station_id"].astype(int) == MUNHAK) & (hist["munhak_game"] == 1) & (hist["series"] == "regular")]
    if m.empty:
        return pd.Series(dtype=float)
    d = m.groupby("date").agg(y=("boardings", "sum"), opp=("opponent", "first"), off=("is_offday", "first"),
                              st=("munhak_start", "first"))
    d["y"] = np.log(d["y"])
    d["slot"] = d["off"].astype(str) + (d["st"] < 17).astype(str)
    d["r"] = d["y"] - d.groupby("slot")["y"].transform("median")
    s = d.groupby("opp")["r"].agg(["mean", "size"])
    return np.exp(s["mean"] * s["size"] / (s["size"] + k))


def load(data_dir, holiday_csv):
    d = Path(data_dir)
    tr = pd.read_csv(d / "train.csv", parse_dates=["date"])
    te = pd.read_csv(d / "test.csv", parse_dates=["date"])
    md = pd.read_csv(d / "metaData.csv")
    cal = make_calendar(holiday_csv)
    tr["y"] = np.log1p(tr["boardings"])
    full = pd.concat([tr, te], ignore_index=True)
    full = full.merge(cal, on="date", how="left").merge(
        md[["station_id", "canonical_line"]].rename(columns={"canonical_line": "line"}), on="station_id", how="left")
    full = add_external(full, EXT_DIR / "weather_incheon_2025_2026.csv", EXT_DIR / "kbo_munhak_2025_2026.csv")
    full = add_external_v3(full)
    full = add_external_v4(full)
    full = add_ops(full)
    mp = pd.read_csv(EXT_DIR / "monthly_prior_2023_2024.csv")
    full = full.assign(_sid=full["station_id"].astype(int)).merge(mp.rename(columns={"station_id": "_sid"}), on=["_sid", "hour", "month"], how="left").drop(columns=["_sid"])
    sc = pd.read_csv(EXT_DIR / "school_calendar_2025_2026.csv", parse_dates=["date"])
    full = full.merge(sc[["date"] + VAC_COLS + UNIV_COLS], on="date", how="left")
    if VAC_SKIP_GEOMDAN:
        # 검단 3역은 2025-06-28 개통 직후가 여름방학이라, 개통 초기 증가를 방학 효과로 오인함 → 방학 값을 0으로 고정
        g = full["station_id"].astype(int).isin(GEOMDAN)
        full.loc[g, VAC_COLS + UNIV_COLS] = 0
    full["station_id"] = full["station_id"].astype("category")
    return full, te


def run_fold_random(full, frac=0.3, start="2025-07-01", end="2025-12-31", seed=SEED, n_rounds=600, verbose=True):
    """검증 C: 2025년 하반기 날짜 중 30%를 무작위로 빼서 예측 (Public/Private처럼 무작위 날짜).
    축제·단풍·물놀이장처럼 1년에 한 번 오는 일을 다른 날로 배운 뒤 맞히는지 보는 용도. v2(학습 평균 기준)만 의미 있음."""
    days = pd.Series(sorted(full.loc[(full["date"] >= start) & (full["date"] <= end), "date"].unique()))
    hold = set(days.sample(frac=frac, random_state=seed))
    m = full["date"].isin(hold)
    hist, fut = full[~m].copy(), full[m].copy()
    fut, preds, _ = predict_period(hist, fut, full["date"].max(), n_rounds, verbose=False)
    sc = rmsle_log(preds["v2_lgb_trainlevel"], fut["y"].values)
    if verbose:
        print(f"  무작위 {len(hold)}일 holdout: v2_lgb_trainlevel RMSLE {sc:.5f}")
    return fut, preds, {"v2_lgb_trainlevel": sc}


def main(data_dir="data", holiday_csv="holidays_kr_2025_2026.csv", mode="cv", out="submission.csv"):
    full, te = load(data_dir, holiday_csv)
    train_part = full[full["example_id"].isna()]
    print("[검증 A] 2025-06-30까지 보고 2025년 7~10월 예측 (같은 계절, 작년 이력 없음)")
    run_fold(train_part, "2025-06-30", pd.Timestamp("2025-10-31"))
    print("[검증 B] 2026-02-28까지 보고 2026년 3~6월 예측 (가장 최근, 작년 같은 달 이력 있음)")
    run_fold(train_part, "2026-02-28", pd.Timestamp("2026-06-30"))

    if mode == "submit":
        hist = train_part.copy()
        fut = full[full["example_id"].notna()].copy()
        fut, preds, _ = predict_period(hist, fut, pd.Timestamp("2026-06-30"), 600, verbose=False)
        p = np.expm1(np.clip(preds["v6_final"], 0, None))  # v10: v9ly + 실험용 설정(학습 기간·최근 가중치·추세·공간 피처)
        sub = pd.DataFrame({"example_id": fut["example_id"].values, "boardings": p})
        sub = te[["example_id"]].merge(sub, on="example_id", how="left")
        assert len(sub) == 157194 and sub["example_id"].is_unique
        assert sub["boardings"].notna().all() and np.isfinite(sub["boardings"]).all() and (sub["boardings"] >= 0).all()
        sub.to_csv(out, index=False)
        print(f"저장: {out} ({len(sub)}행)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--holidays", default="holidays_kr_2025_2026.csv")
    ap.add_argument("--mode", default="cv", choices=["cv", "submit"])
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    main(a.data_dir, a.holidays, a.mode, a.out)
