"""
export_processed_data.py
'sincon_추가데이터2' 전면 반영 정제 및 마스터 피처셋 생성 스크립트

반영된 핵심 개선사항 (설명서.md 및 추가데이터2 검증 완료):
1. 2026-09-28 정상 평일 반영 (추석 연휴 9/24~27 및 대체공휴일 없음)
2. 2026-07-17 제헌절 공휴일(18년만의 재지정) 정확한 반영
3. 문학경기장 공연 6일 신규 추가 (8/29~30, 9/12~13, 10/24~25)
4. 근거 없는 허위 행사일(8/28, 9/9, 9/25, 9/26) 배제
5. 문학 시범경기 14일 및 공식 취소 경기 17일 정밀 반영
6. K리그 공식 관중 수 및 검증된 275개 events_master.csv 전면 연동
7. past_ridership_2023_2024_monthly 기반의 월별 역사별 형상(hist_month_shape) 산출
8. 도메인 Prior(prior_log_b) 자동 산출 및 최신화
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'sincon-2026'
EXTRA2_DIR = BASE_DIR / 'sincon_추가데이터2'
OUT_DIR = BASE_DIR / 'data_processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("[1/5] 'sincon_추가데이터2' 및 대회 기본 데이터 로드 중...")

train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw = pd.read_csv(DATA_DIR / 'test.csv')
meta_raw = pd.read_csv(DATA_DIR / 'metaData.csv')

cal_raw = pd.read_csv(EXTRA2_DIR / 'derived' / 'calendar_features.csv')
events_raw = pd.read_csv(EXTRA2_DIR / 'external' / 'events_master.csv')
station_master = pd.read_csv(EXTRA2_DIR / 'external' / 'station_master.csv')
weather_hourly = pd.read_csv(EXTRA2_DIR / 'external' / 'weather_hourly.csv')
past_ridership = pd.read_csv(EXTRA2_DIR / 'external' / 'past_ridership_2023_2024_monthly.csv')

print(f"  - Train: {train_raw.shape}")
print(f"  - Test:  {test_raw.shape}")
print(f"  - Verified Events: {events_raw.shape[0]}건")
print(f"  - Stations: {station_master.shape[0]}개 역")
print(f"  - Calendar: {cal_raw.shape[0]}일")

# ------------------------------------------------------------------------------
# 1. 캘린더 피처 마스터 정제
# ------------------------------------------------------------------------------
print("\n[2/5] 캘린더 피처 마스터 정제 중...")
cal = cal_raw.copy()
cal_dt = pd.to_datetime(cal['date'])
cal['month'] = cal_dt.dt.month
cal['day'] = cal_dt.dt.day
cal['weekday'] = cal['dow']
cal['dayofyear'] = cal_dt.dt.day_of_year

# 필수 캘린더 컬럼 유지 및 저장
cal_save_cols = [
    'date', 'month', 'day', 'weekday', 'dayofyear',
    'is_holiday', 'is_parents_day', 'holiday_type', 'eff_weekday', 'eff_is_wknd',
    'seasonal_period_code', 'offday_run_len', 'is_bridge', 'before_long_offday', 'after_long_offday',
    'univ_semester', 'vac_elem', 'vac_middle', 'vac_high', 'vac_all',
    'precip_day_mm', 'temp_max_c', 'temp_min_c', 'snow_day_cm', 'gumdan_pop_adj'
]
cal = cal[cal_save_cols].copy()
cal.to_csv(OUT_DIR / 'calendar_master.csv', index=False, encoding='utf-8-sig')

# ------------------------------------------------------------------------------
# 2. 이벤트 & 역 메타 마스터 정제
# ------------------------------------------------------------------------------
print("[3/5] 이벤트 & 역 메타 정제 중...")

# events_master는 이미 검증 완료된 275건
events = events_raw[['date', 'station_id', 'venue', 'event_name', 'ev_key', 'ev_start', 'ev_att']].copy()
events = events.drop_duplicates(subset=['date', 'station_id'], keep='first')
events.to_csv(OUT_DIR / 'events_master.csv', index=False, encoding='utf-8-sig')

ev_key_list = sorted(events['ev_key'].dropna().unique())
ev_key_codes = {k: i + 1 for i, k in enumerate(ev_key_list)}
print(f"  - 이벤트 키 종류: {len(ev_key_list)}개 -> {ev_key_list}")

# 역 메타 마스터 및 출퇴근 비율 계산
g_commute = train_raw.groupby(['station_id', 'hour'])['boardings'].mean().unstack()
rush_ratio_map = (g_commute[[7, 8, 9]].sum(axis=1) / g_commute[[18, 19, 20]].sum(axis=1)).to_dict()
station_master['rush_ratio'] = station_master['station_id'].map(rush_ratio_map).fillna(2.0)
station_master.to_csv(OUT_DIR / 'station_master.csv', index=False, encoding='utf-8-sig')

# ------------------------------------------------------------------------------
# 3. Train / Test Enriched 기본 피처 생성
# ------------------------------------------------------------------------------
print("[4/5] Enriched 기본 피처 생성 중...")
night_ext_dates = [
    '2025-01-29', '2025-01-30', '2025-10-06', '2025-10-07',
    '2026-02-17', '2026-02-18', '2026-09-26', '2026-09-27'
]

def enrich(df):
    out = df.merge(cal, on='date', how='left')
    out = out.merge(
        station_master[[
            'station_id', 'canonical_line', 'open_date', 'is_transfer', 'is_ext_transfer',
            'is_geomdan_new', 'is_univ_station', 'rush_ratio'
        ]],
        on='station_id', how='left'
    )
    out = out.merge(weather_hourly[['date', 'hour', 'precip_mm', 'temp_c', 'snow_cm']], on=['date', 'hour'], how='left')
    out = out.merge(events[['date', 'station_id', 'ev_key', 'ev_start', 'ev_att']], on=['date', 'station_id'], how='left')
    
    out['ev_key'] = out['ev_key'].fillna('')
    out['ev_code'] = out['ev_key'].map(ev_key_codes).fillna(0).astype(int)
    out['ev_hours_from_start'] = np.where(out['ev_code'] > 0, out['hour'] - out['ev_start'], -99.0)
    out['ev_att_log'] = np.log1p(out['ev_att'].fillna(0.0))
    
    # 1. 행사 퇴장/입장 가우시안 피크
    exit_diff = out['hour'] - (out['ev_start'] + 3.0)
    out['ev_exit_peak'] = np.where(out['ev_code'] > 0, np.exp(-0.5 * (exit_diff / 0.9) ** 2) * out['ev_att_log'], 0.0)
    entry_diff = out['hour'] - (out['ev_start'] - 1.2)
    out['ev_entry_peak'] = np.where(out['ev_code'] > 0, np.exp(-0.5 * (entry_diff / 0.8) ** 2) * out['ev_att_log'], 0.0)
    
    # 2. 날씨 x 주말 / 출퇴근 상호작용
    out['rain_on_wknd'] = out['precip_mm'] * out['eff_is_wknd']
    out['heat_on_wknd'] = np.maximum(0.0, out['temp_c'] - 32.0) * out['eff_is_wknd']
    is_rush = out['hour'].isin([7, 8, 9, 18, 19]).astype(int)
    out['rain_on_rush'] = out['precip_mm'] * is_rush * (1 - out['eff_is_wknd'])
    
    # 3. 심야 연장 운행일 피처
    out['is_night_extended'] = (out['date'].isin(night_ext_dates) & (out['hour'] >= 23)).astype(int)
    
    out['days_since_open'] = (pd.to_datetime(out['date']) - pd.to_datetime(out['open_date'])).dt.days
    out['sin_doy'] = np.sin(2 * np.pi * out['dayofyear'] / 365.25)
    out['cos_doy'] = np.cos(2 * np.pi * out['dayofyear'] / 365.25)
    out['is_rain'] = (out['precip_mm'] > 0.1).astype(int)
    
    if 'boardings' in out.columns:
        out['log_b'] = np.log1p(out['boardings'])
        out['log_a'] = np.log1p(out['alightings'])
    return out

train_df = enrich(train_raw)
test_df = enrich(test_raw)

# 평상시 클린 데이터 정의 (이상치/공휴일 배제)
def is_clean_row(df):
    return (~df['holiday_type'].isin([2, 3, 4])) & (df['ev_key'] == '') & \
        ~((df['station_id'] == 3122) & (df['is_parents_day'] == 1))

train_df['clean'] = is_clean_row(train_df)

# 과거 2023-2024 월보 기반 월별 형상(hist_month_shape)
hcols = [f'{h:02d}시' for h in range(6, 24)]
mh = past_ridership.melt(id_vars=['연도', '월', '일수', 'station_id'], value_vars=hcols, var_name='hcol', value_name='tot')
mh['hour'] = mh['hcol'].str[:2].astype(int)
mh['lavg'] = np.log1p(mh['tot'] / mh['일수'])
mh['shape'] = mh['lavg'] - mh.groupby(['연도', 'station_id', 'hour'])['lavg'].transform('mean')

def hist_shape_for(year):
    s = mh[mh['연도'] < year].groupby(['station_id', '월', 'hour'])['shape'].mean()
    return s.rename('hist_month_shape').reset_index().rename(columns={'월': 'month'})

# ------------------------------------------------------------------------------
# 4. 시계열 블록 구축 (과거 8주/4주 무누수 기준 프로파일)
# ------------------------------------------------------------------------------
print("[5/5] 시계열 블록 (B1, B2, B3, V, T) 구축 중...")

def profile(win, keys, col, min_n=3):
    g = win.groupby(keys)[col].agg(['mean', 'count'])
    return g['mean'].where(g['count'] >= min_n)

def lookup(prof, df, keys):
    return prof.reindex(pd.MultiIndex.from_arrays([df[k] for k in keys])).to_numpy()

geomdan_shift = [3110, 3111]

def build_block(target, origin, block_name):
    origin_dt = pd.Timestamp(origin)
    hist = train_df[(train_df['date'] <= origin) & train_df['clean']]
    w8 = hist[hist['date'] >= (origin_dt - pd.Timedelta(days=55)).strftime('%Y-%m-%d')]
    w4 = hist[hist['date'] >= (origin_dt - pd.Timedelta(days=27)).strftime('%Y-%m-%d')]
    res = target.copy()
    res['block'] = block_name
    
    k_ew = ['station_id', 'eff_weekday', 'hour']
    k_wk = ['station_id', 'eff_is_wknd', 'hour']

    b_ew = lookup(profile(w8, k_ew, 'log_b'), res, k_ew)
    b_wk = lookup(profile(w8, k_wk, 'log_b', 5), res, k_wk)
    res['base_log_b'] = np.where(np.isnan(b_ew), b_wk, b_ew)
    res['base_wk_log_b'] = b_wk
    
    r_ew = lookup(profile(w4, k_ew, 'log_b', 2), res, k_ew)
    r_wk = lookup(profile(w4, k_wk, 'log_b', 3), res, k_wk)
    res['recent4_log_b'] = np.where(np.isnan(r_ew), r_wk, r_ew)
    res['trend_4w'] = res['recent4_log_b'] - res['base_log_b']
    
    res['base_log_a'] = lookup(profile(w8, k_ew, 'log_a'), res, k_ew)
    res['base_ab_diff'] = res['base_log_a'] - res['base_log_b']
    res['base_dow_rel'] = res['base_log_b'] - res['base_wk_log_b']
    
    st_day = w8.groupby('station_id')['log_b'].mean()
    res['station_level'] = res['station_id'].map(st_day)

    wb = weather_hourly[(weather_hourly['date'] >= w8['date'].min()) & (weather_hourly['date'] <= origin)]
    res['temp_vs_base'] = res['temp_c'] - wb['temp_c'].mean()

    res['days_ahead'] = (pd.to_datetime(res['date']) - origin_dt).dt.days
    pop_o = cal.loc[cal['date'] == origin, 'gumdan_pop_adj'].iloc[0]
    res['gumdan_growth'] = np.log(res['gumdan_pop_adj'] / pop_o) * res['is_geomdan_new']
    res['regime_shift'] = (res['station_id'].isin(geomdan_shift) & (origin < '2025-06-28')
                           & (res['date'] >= '2025-06-28')).astype(int)

    year = int(res['date'].str[:4].iloc[0]) if res['date'].str[:4].nunique() == 1 else None
    if year is not None and res['month'].between(7, 10).all():
        res = res.merge(hist_shape_for(year), on=['station_id', 'month', 'hour'], how='left')
    else:
        res['hist_month_shape'] = np.nan
    return res

def split(df, a, b):
    return df[(df['date'] >= a) & (df['date'] <= b)].copy()

B1 = build_block(split(train_df, '2025-03-01', '2025-06-30'), '2025-02-28', 'B1_2025_03_06')
B2 = build_block(split(train_df, '2025-07-01', '2025-10-31'), '2025-06-30', 'B2_2025_07_10')
B3 = build_block(split(train_df, '2025-11-01', '2026-02-28'), '2025-10-31', 'B3_2025_11_2026_02')
V  = build_block(split(train_df, '2026-03-01', '2026-06-30'), '2026-02-28', 'V_2026_03_06')
T  = build_block(test_df.copy(), '2026-06-30', 'T_2026_07_10')

# Parquet & CSV 동시 저장
full_train = pd.concat([B1, B2, B3, V], ignore_index=True)

print(f"\n  - Full Train Shape: {full_train.shape}")
print(f"  - Test Shape:       {T.shape}")

full_train.to_parquet(BASE_DIR / 'train_features.parquet', index=False)
T.to_parquet(BASE_DIR / 'test_features.parquet', index=False)

full_train.to_csv(OUT_DIR / 'train_features.csv', index=False)
T.to_csv(OUT_DIR / 'test_features.csv', index=False)

print(f"[저장 완료] {BASE_DIR / 'train_features.parquet'} ({len(full_train):,}행)")
print(f"[저장 완료] {BASE_DIR / 'test_features.parquet'} ({len(T):,}행)")

# ------------------------------------------------------------------------------
# 5. 최신화된 Domain Prior (submission_prior.csv) 산출
# ------------------------------------------------------------------------------
print("\n[6/5] 'sincon_추가데이터2' 기반 최신 Domain Prior 산출 중...")

def compute_prior_series(blk, origin, ly_base, ly_target):
    origin_dt = pd.Timestamp(origin)
    hist_all = train_df[(train_df['date'] <= origin)]
    hist = hist_all[hist_all['clean']]
    k_wk = ['station_id', 'eff_is_wknd', 'hour']
    
    ly_b = hist[(hist['date'] >= ly_base[0]) & (hist['date'] <= ly_base[1])]
    ly_t = hist[(hist['date'] >= ly_target[0]) & (hist['date'] <= ly_target[1])]
    rb = ly_b.groupby(k_wk)['log_b'].mean()
    rt = ly_t.groupby(['station_id', 'seasonal_period_code', 'eff_is_wknd', 'hour'])['log_b'].mean()
    sd = (rt - rb.reindex(pd.MultiIndex.from_arrays([rt.index.get_level_values(i) for i in (0, 2, 3)])).to_numpy())
    
    geomdan_new = [3107, 3108, 3109]
    avg_sd = sd[~sd.index.get_level_values(0).isin(geomdan_new + geomdan_shift)].groupby(level=[1, 2, 3]).mean()
    d = sd.reindex(pd.MultiIndex.from_arrays([blk['station_id'], blk['seasonal_period_code'], blk['eff_is_wknd'], blk['hour']])).to_numpy()
    d_avg = avg_sd.reindex(pd.MultiIndex.from_arrays([blk['seasonal_period_code'], blk['eff_is_wknd'], blk['hour']])).to_numpy()
    use_avg = blk['station_id'].isin(geomdan_new + geomdan_shift).to_numpy() | np.isnan(d)
    sdelta = np.nan_to_num(np.where(use_avg, d_avg, d), nan=0.0)

    gtrend = np.where(
        blk['is_geomdan_new'] == 1,
        0.65 * blk['trend_4w'] + np.where(blk['station_id'] == 3107, 0.025, 0.012) / 30 * blk['days_ahead'].clip(0, 150),
        0.20 * blk['trend_4w']
    )
    prior = blk['base_log_b'].to_numpy() + 0.90 * sdelta + np.nan_to_num(gtrend)

    ev_delta = np.zeros(len(blk))
    lunar = hist_all[hist_all['holiday_type'].isin([3, 4])]
    sun = hist[hist['weekday'] == 6].groupby(['station_id', 'hour'])['log_b'].mean()
    lp = lunar.groupby(['station_id', 'holiday_type', 'hour'])['log_b'].mean()
    ld = (lp - sun.reindex(pd.MultiIndex.from_arrays([lp.index.get_level_values(0), lp.index.get_level_values(2)])).to_numpy())
    m = blk['holiday_type'].isin([3, 4]).to_numpy()
    if m.any():
        ev_delta[m] = np.nan_to_num(ld.reindex(pd.MultiIndex.from_arrays(
            [blk.loc[m, 'station_id'], blk.loc[m, 'holiday_type'], blk.loc[m, 'hour']])).to_numpy(), nan=-0.25)
            
    lab = hist_all[hist_all['holiday_type'] == 2].groupby(['station_id', 'hour'])['log_b'].mean()
    mayw = hist[(hist['month'] == 5) & (hist['eff_is_wknd'] == 0)].groupby(['station_id', 'hour'])['log_b'].mean()
    m = (blk['holiday_type'] == 2).to_numpy()
    if m.any():
        ev_delta[m] = np.nan_to_num((lab - mayw).reindex(pd.MultiIndex.from_arrays(
            [blk.loc[m, 'station_id'], blk.loc[m, 'hour']])).to_numpy(), nan=-0.35)
            
    evh = hist_all[hist_all['ev_key'] != '']
    norm = hist.groupby(k_wk)['log_b'].mean()
    e1 = evh.groupby(['station_id', 'ev_key', 'eff_is_wknd', 'hour'])['log_b'].mean()
    e1 = e1 - norm.reindex(pd.MultiIndex.from_arrays([e1.index.get_level_values(i) for i in (0, 2, 3)])).to_numpy()
    e2 = e1.groupby(level=[0, 1, 3]).mean()
    m = (blk['ev_key'] != '').to_numpy()
    if m.any():
        sub = blk.loc[m]
        v1 = e1.reindex(pd.MultiIndex.from_arrays([sub['station_id'], sub['ev_key'], sub['eff_is_wknd'], sub['hour']])).to_numpy()
        v2 = e2.reindex(pd.MultiIndex.from_arrays([sub['station_id'], sub['ev_key'], sub['hour']])).to_numpy()
        ev_delta[m] = np.nan_to_num(np.where(np.isnan(v1), v2, v1), nan=0.0)
        
    return np.clip(prior + ev_delta, 0, None)

prior_test_log = compute_prior_series(T, '2026-06-30', ('2025-05-06', '2025-06-30'), ('2025-07-01', '2025-10-31'))
prior_boardings = np.clip(np.expm1(prior_test_log), 0, None)

sub_prior = pd.DataFrame({
    'example_id': T['example_id'],
    'boardings': prior_boardings
})
sub_prior.to_csv(BASE_DIR / 'submission_prior.csv', index=False)
print(f"[저장 완료] {BASE_DIR / 'submission_prior.csv'} ({len(sub_prior):,}행)")
print("\n[성공] 'sincon_추가데이터2' 기반 전처리 및 데이터 추출이 완료되었습니다!")
print("=" * 70)
