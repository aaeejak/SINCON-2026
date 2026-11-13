"""
SINCON 2026 - 인천 지하철 승차인원 예측 파이프라인 (누수 제거 + 실측 외부 데이터 반영)

주요 특징:
1. 시간대 누수 없는 순방향 롤링 블록 구조 (Origin-based 4M Forecasting)
   - B1: ~2025-02-28 -> 2025-03~06
   - B2: ~2025-06-30 -> 2025-07~10
   - B3: ~2025-10-31 -> 2025-11~2026-02
   - V : ~2026-02-28 -> 2026-03~06 (검증 블록)
   - T : ~2026-06-30 -> 2026-07~10 (테스트 블록)
2. 7월 이후 외부 데이터(날씨 실측, 경기/축제 관중 수, 행사 일정 변경, 공휴일 변동) 전면 활용
3. 도메인 Prior(전년 동기 계절/학사/명절 보정) + LightGBM 직접 모델 최적 가중 블렌딩
"""

from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'sincon-2026'
EXTRA_DIR = BASE_DIR / 'sincon_추가데이터' / 'sincon_추가데이터'
NEW_DIR = BASE_DIR / 'SINCON_데이터_CSV'
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR

train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw = pd.read_csv(DATA_DIR / 'test.csv')
meta = pd.read_csv(DATA_DIR / 'metaData.csv')
cal_extra = pd.read_csv(EXTRA_DIR / 'derived' / 'calendar_features.csv')
weather_hourly = pd.read_csv(EXTRA_DIR / 'external' / 'weather_hourly.csv')
ssg = pd.read_csv(EXTRA_DIR / 'external' / 'ssg_home_games_2025_2026.csv')
events_new = pd.read_csv(NEW_DIR / '04_행사_일정' / '전체_2023-2026.csv', encoding='utf-8-sig')
monthly_hist = pd.read_csv(NEW_DIR / '03_과거3개년_7-10월_승차' / '월별_역·시간대_합계.csv', encoding='utf-8-sig')
station_cls = pd.read_csv(NEW_DIR / '02_환승역_증가일·외부요인' / '전체_역_구분.csv', encoding='utf-8-sig')
open_dates = pd.read_csv(NEW_DIR / '05_역_개통일' / 'station_open_dates.csv', encoding='utf-8-sig')

print(f"[로드] train {train_raw.shape}, test {test_raw.shape}")

# ------------------------------------------------------------------------------
# 1. 캘린더 (공휴일 재정비: 2026-07-17 제헌절 반영)
# ------------------------------------------------------------------------------
h_general = [
    '2025-01-01', '2025-03-01', '2025-03-03', '2025-05-05', '2025-05-06',
    '2025-06-03', '2025-06-06', '2025-08-15', '2025-10-03', '2025-10-08',
    '2025-10-09', '2025-12-25',
    '2026-01-01', '2026-03-01', '2026-03-02', '2026-05-05', '2026-05-24',
    '2026-05-25', '2026-06-03', '2026-06-06',
    '2026-07-17',  # 제헌절 공휴일
    '2026-08-15', '2026-08-17', '2026-10-03', '2026-10-05', '2026-10-09'
]
h_labor = ['2025-05-01', '2026-05-01']
h_lunar_sub = ['2025-01-27', '2025-01-28', '2025-01-30', '2025-10-05', '2025-10-07',
               '2026-02-16', '2026-02-18', '2026-09-24', '2026-09-26']
h_lunar_main = ['2025-01-29', '2025-10-06', '2026-02-17', '2026-09-25']
parents_days = ['2025-05-08', '2026-05-08']

cal = pd.DataFrame({'dt': pd.date_range('2025-01-01', '2026-10-31', freq='D')})
cal['date'] = cal['dt'].dt.strftime('%Y-%m-%d')
cal['month'] = cal['dt'].dt.month
cal['day'] = cal['dt'].dt.day
cal['weekday'] = cal['dt'].dt.day_of_week
cal['dayofyear'] = cal['dt'].dt.day_of_year
cal['holiday_type'] = 0
cal.loc[cal['date'].isin(h_general), 'holiday_type'] = 1
cal.loc[cal['date'].isin(h_labor), 'holiday_type'] = 2
cal.loc[cal['date'].isin(h_lunar_sub), 'holiday_type'] = 3
cal.loc[cal['date'].isin(h_lunar_main), 'holiday_type'] = 4
cal['is_holiday'] = cal['holiday_type'].isin([1, 3, 4]).astype(int)
cal['is_parents_day'] = cal['date'].isin(parents_days).astype(int)
cal['eff_weekday'] = np.where(cal['is_holiday'] == 1, 6, cal['weekday'])
cal['eff_is_wknd'] = (cal['eff_weekday'] >= 5).astype(int)

off = (cal['eff_is_wknd'] == 1).to_numpy()
run_len = np.zeros(len(cal), dtype=int)
i = 0
while i < len(cal):
    if off[i]:
        j = i
        while j < len(cal) and off[j]:
            j += 1
        run_len[i:j] = j - i
        i = j
    else:
        i += 1
cal['offday_run_len'] = run_len
prev_off = np.r_[False, off[:-1]]
next_off = np.r_[off[1:], False]
prev_run = np.r_[0, run_len[:-1]]
next_run = np.r_[run_len[1:], 0]
cal['is_bridge'] = ((~off) & prev_off & next_off).astype(int)
cal['before_long_offday'] = ((~off) & (next_run >= 3)).astype(int)
cal['after_long_offday'] = ((~off) & (prev_run >= 3)).astype(int)


def seasonal_period(m, d):
    if m == 6:
        return 6 if d <= 19 else 7
    if m == 7:
        return 8 if d <= 19 else 9
    if m == 8:
        return 9 if d <= 16 else 10
    return {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 9: 11, 10: 12, 11: 13, 12: 14}[m]


cal['seasonal_period_code'] = [seasonal_period(m, d) for m, d in zip(cal['month'], cal['day'])]
cal = cal.merge(cal_extra[['date', 'univ_semester', 'vac_elem', 'vac_middle', 'vac_high', 'vac_all',
                           'precip_day_mm', 'temp_max_c', 'temp_min_c', 'snow_day_cm', 'gumdan_pop_adj']],
                on='date', how='left')

# ------------------------------------------------------------------------------
# 2. 행사 테이블 (실측 관중 수 포함)
# ------------------------------------------------------------------------------
def parse_start(s, default):
    if not isinstance(s, str) or ':' not in s:
        return default
    hh, mm = s.split('~')[0].strip().split(':')[:2]
    return int(hh) + int(mm) / 60.0


ev_rows = []
for _, r in ssg[ssg['played'] == 1].iterrows():
    st = parse_start(r['start_time'], 18.5)
    key = 'bb_14' if st < 15 else 'bb_17' if st < 17.5 else 'bb_18' if st < 18.25 else 'bb_1830'
    att = float(r['crowd']) if 'crowd' in r and pd.notna(r['crowd']) else np.nan
    ev_rows.append((r['date'], 3127, key, st, att, 1))

venue_sid = {'문학경기장 공연': 3127, '참고: 아시아드 공연': 3209, '부천FC 홈경기': 3754, '송도달빛축제공원': 3139}
for _, r in events_new.iterrows():
    g, name = r['구분'], str(r['행사명'])
    if g not in venue_sid or '[취소]' in name:
        continue
    att = pd.to_numeric(str(r['규모/관중']).replace(',', ''), errors='coerce')
    if g == '부천FC 홈경기':
        st = parse_start(r['시작시간'], 19.0)
        key = 'fc_day' if st < 15 else 'fc_1630' if st < 18 else 'fc_night'
        prio = 1
    elif g == '송도달빛축제공원':
        st = parse_start(r['시작시간'], 12.0)
        key = 'pentaport' if '펜타포트' in name else 'beer' if '맥주' in name else 'fest'
        prio = 1
    else:
        st = parse_start(r['시작시간'], 18.0)
        key = 'concert'
        prio = 0
    ev_rows.append((r['date'], venue_sid[g], key, st, att, prio))

events = pd.DataFrame(ev_rows, columns=['date', 'station_id', 'ev_key', 'ev_start', 'ev_att', 'prio'])
events = events[(events['date'] >= '2025-01-01') & (events['date'] <= '2026-10-31')]
events = events.sort_values('prio').drop_duplicates(['date', 'station_id'], keep='first').drop(columns='prio')
ev_key_codes = {k: i + 1 for i, k in enumerate(sorted(events['ev_key'].unique()))}
print(f"[행사] {len(events)}건: {events.groupby(['station_id', 'ev_key']).size().to_dict()}")

# ------------------------------------------------------------------------------
# 3. 역 메타 피처 결합
# ------------------------------------------------------------------------------
meta_feat = meta[['station_id', 'canonical_line']].copy()
meta_feat = meta_feat.merge(open_dates[['station_id', 'open_date']], on='station_id', how='left')
cls = station_cls[['station_id', '구분', '환승 유형']].rename(columns={'구분': 'cls', '환승 유형': 'xfer'})
meta_feat = meta_feat.merge(cls, on='station_id', how='left')
meta_feat['is_transfer'] = (meta_feat['cls'] == '환승역').astype(int)
meta_feat['is_ext_transfer'] = meta_feat['xfer'].astype(str).str.contains('외부').astype(int)
meta_feat = meta_feat.drop(columns=['cls', 'xfer'])
geomdan_new = [3107, 3108, 3109]
geomdan_shift = [3110, 3111]
meta_feat['is_geomdan_new'] = meta_feat['station_id'].isin(geomdan_new).astype(int)
meta_feat['is_univ_station'] = meta_feat['station_id'].isin([3136, 3133, 3115, 3134]).astype(int)


def enrich(df):
    out = df.merge(cal.drop(columns='dt'), on='date', how='left')
    out = out.merge(meta_feat, on='station_id', how='left')
    out = out.merge(weather_hourly, on=['date', 'hour'], how='left')
    out = out.merge(events, on=['date', 'station_id'], how='left')
    out['ev_key'] = out['ev_key'].fillna('')
    out['ev_code'] = out['ev_key'].map(ev_key_codes).fillna(0).astype(int)
    out['ev_hours_from_start'] = np.where(out['ev_code'] > 0, out['hour'] - out['ev_start'], -99)
    out['ev_att_log'] = np.log1p(out['ev_att'].fillna(0))
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


def is_clean_row(df):
    return (~df['holiday_type'].isin([2, 3, 4])) & (df['ev_key'] == '') & \
        ~((df['station_id'] == 3122) & (df['is_parents_day'] == 1))


train_df['clean'] = is_clean_row(train_df)

# ------------------------------------------------------------------------------
# 4. 과거 3개년 7~10월 월별 형태(shape) 피처
# ------------------------------------------------------------------------------
hcols = [f'{h:02d}시' for h in range(6, 24)]
mh = monthly_hist.melt(id_vars=['연도', '월', '일수', 'station_id'], value_vars=hcols, var_name='hcol', value_name='tot')
mh['hour'] = mh['hcol'].str[:2].astype(int)
mh['lavg'] = np.log1p(mh['tot'] / mh['일수'])
mh['shape'] = mh['lavg'] - mh.groupby(['연도', 'station_id', 'hour'])['lavg'].transform('mean')


def hist_shape_for(year):
    s = mh[mh['연도'] < year].groupby(['station_id', '월', 'hour'])['shape'].mean()
    return s.rename('hist_month_shape').reset_index().rename(columns={'월': 'month'})


# ------------------------------------------------------------------------------
# 5. 블록 구성: origin 이전 데이터로 base 프로파일 -> 이후 4개월 타깃
# ------------------------------------------------------------------------------
def profile(win, keys, col, min_n=3):
    g = win.groupby(keys)[col].agg(['mean', 'count'])
    return g['mean'].where(g['count'] >= min_n)


def lookup(prof, df, keys):
    return prof.reindex(pd.MultiIndex.from_arrays([df[k] for k in keys])).to_numpy()


def build_block(target, origin):
    origin_dt = pd.Timestamp(origin)
    hist = train_df[(train_df['date'] <= origin) & train_df['clean']]
    w8 = hist[hist['date'] >= (origin_dt - pd.Timedelta(days=55)).strftime('%Y-%m-%d')]
    w4 = hist[hist['date'] >= (origin_dt - pd.Timedelta(days=27)).strftime('%Y-%m-%d')]
    res = target.copy()
    k_ew, k_wk = ['station_id', 'eff_weekday', 'hour'], ['station_id', 'eff_is_wknd', 'hour']

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


B1 = build_block(split(train_df, '2025-03-01', '2025-06-30'), '2025-02-28')
B2 = build_block(split(train_df, '2025-07-01', '2025-10-31'), '2025-06-30')
B3 = build_block(split(train_df, '2025-11-01', '2026-02-28'), '2025-10-31')
V = build_block(split(train_df, '2026-03-01', '2026-06-30'), '2026-02-28')
T = build_block(test_df.copy(), '2026-06-30')
for n, b in [('B1', B1), ('B2', B2), ('B3', B3), ('V', V), ('T', T)]:
    print(f"[블록 {n}] rows={len(b):,}  base NaN={b['base_log_b'].isna().mean():.4f}  "
          f"hist_shape non-null={b['hist_month_shape'].notna().mean():.2f}")
assert T['base_log_b'].notna().all(), "테스트 base 결측"

# ------------------------------------------------------------------------------
# 6. 도메인 Prior (V, T 전용)
# ------------------------------------------------------------------------------
def domain_prior(blk, origin, ly_base, ly_target):
    hist_all = train_df[train_df['date'] <= origin]
    hist = hist_all[hist_all['clean']]
    k_wk = ['station_id', 'eff_is_wknd', 'hour']
    ly_b = hist[(hist['date'] >= ly_base[0]) & (hist['date'] <= ly_base[1])]
    ly_t = hist[(hist['date'] >= ly_target[0]) & (hist['date'] <= ly_target[1])]
    rb = ly_b.groupby(k_wk)['log_b'].mean()
    rt = ly_t.groupby(['station_id', 'seasonal_period_code', 'eff_is_wknd', 'hour'])['log_b'].mean()
    sd = (rt - rb.reindex(pd.MultiIndex.from_arrays(
        [rt.index.get_level_values(i) for i in (0, 2, 3)])).to_numpy())
    avg_sd = sd[~sd.index.get_level_values(0).isin(geomdan_new + geomdan_shift)].groupby(level=[1, 2, 3]).mean()
    d = sd.reindex(pd.MultiIndex.from_arrays(
        [blk['station_id'], blk['seasonal_period_code'], blk['eff_is_wknd'], blk['hour']])).to_numpy()
    d_avg = avg_sd.reindex(pd.MultiIndex.from_arrays(
        [blk['seasonal_period_code'], blk['eff_is_wknd'], blk['hour']])).to_numpy()
    use_avg = blk['station_id'].isin(geomdan_new + geomdan_shift).to_numpy() | np.isnan(d)
    sdelta = np.nan_to_num(np.where(use_avg, d_avg, d), nan=0.0)

    gtrend = np.where(blk['is_geomdan_new'] == 1,
                      0.65 * blk['trend_4w'] + np.where(blk['station_id'] == 3107, 0.025, 0.012) / 30 * blk['days_ahead'].clip(0, 150),
                      0.20 * blk['trend_4w'])
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


V['prior_log_b'] = domain_prior(V, '2026-02-28', ('2025-01-05', '2025-02-28'), ('2025-03-01', '2025-06-30'))
T['prior_log_b'] = domain_prior(T, '2026-06-30', ('2025-05-05', '2025-06-30'), ('2025-07-01', '2025-10-31'))

# ------------------------------------------------------------------------------
# 7. 직접 모델 (LightGBM)
# ------------------------------------------------------------------------------
cat_cols = ['station_id', 'canonical_line', 'hour', 'weekday', 'eff_weekday', 'holiday_type',
            'seasonal_period_code', 'month', 'ev_code']
num_cols = ['base_log_b', 'base_wk_log_b', 'recent4_log_b', 'trend_4w', 'base_log_a', 'base_ab_diff',
            'base_dow_rel', 'station_level', 'days_ahead', 'gumdan_growth', 'regime_shift', 'hist_month_shape',
            'is_holiday', 'eff_is_wknd', 'offday_run_len', 'is_bridge', 'before_long_offday', 'after_long_offday',
            'univ_semester', 'vac_elem', 'vac_middle', 'vac_high', 'vac_all', 'is_univ_station',
            'precip_mm', 'temp_c', 'snow_cm', 'is_rain', 'precip_day_mm', 'temp_max_c', 'temp_min_c',
            'snow_day_cm', 'temp_vs_base', 'ev_start', 'ev_hours_from_start', 'ev_att_log',
            'is_geomdan_new', 'days_since_open', 'is_transfer', 'is_ext_transfer', 'sin_doy', 'cos_doy', 'day']
feats = cat_cols + num_cols


def rmsle_log(pred_log, actual):
    return float(np.sqrt(np.mean((np.clip(pred_log, 0, None) - np.log1p(actual)) ** 2)))


params = dict(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=127,
              min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, n_jobs=-1)


def fit(blocks, rounds=None, valid=None, seed=42):
    tr = pd.concat(blocks, ignore_index=True)
    tr = tr[tr['base_log_b'].notna()]
    dtr = lgb.Dataset(tr[feats], tr['log_b'] - tr['base_log_b'], categorical_feature=cat_cols)
    p = dict(params, seed=seed)
    if valid is not None:
        dva = lgb.Dataset(valid[feats], valid['log_b'] - valid['base_log_b'], categorical_feature=cat_cols, reference=dtr)
        return lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(p, dtr, rounds)


SEEDS = [42, 7, 2026]
val_models = [fit([B1, B2, B3], valid=V, seed=s) for s in SEEDS]
best_rounds = int(np.mean([m.best_iteration for m in val_models]))
V['direct_log_b'] = V['base_log_b'] + np.mean([m.predict(V[feats], num_iteration=m.best_iteration) for m in val_models], axis=0)

y_v = V['boardings'].to_numpy()
sc_base = rmsle_log(V['base_log_b'].to_numpy(), y_v)
sc_prior = rmsle_log(V['prior_log_b'].to_numpy(), y_v)
sc_direct = rmsle_log(V['direct_log_b'].to_numpy(), y_v)
blend_scores = {w: rmsle_log(w * V['direct_log_b'] + (1 - w) * V['prior_log_b'], y_v) for w in np.round(np.linspace(0, 1, 11), 2)}
best_w = min(blend_scores, key=blend_scores.get)

print("\n" + "=" * 70)
print(f"[검증 V: 2026-03~06]  (학습 B1+B2+B3, 조기종료 평균 {best_rounds}회)")
print(f"  base(최근 8주 프로파일)      : {sc_base:.5f}")
print(f"  도메인 Prior                 : {sc_prior:.5f}")
print(f"  직접 LightGBM                : {sc_direct:.5f}")
print(f"  블렌드 (직접 {best_w:.1f} + Prior {1 - best_w:.1f}) : {blend_scores[best_w]:.5f}")
print("  블렌드 곡선:", {k: round(v, 5) for k, v in blend_scores.items()})
print("=" * 70)

# ------------------------------------------------------------------------------
# 8. 최종 학습 (B1+B2+B3+V) -> 테스트 추론 및 제출 파일 생성
# ------------------------------------------------------------------------------
final_rounds = int(best_rounds * 1.1)
final_models = [fit([B1, B2, B3, V], rounds=final_rounds, seed=s) for s in SEEDS]
T['direct_log_b'] = T['base_log_b'] + np.mean([m.predict(T[feats]) for m in final_models], axis=0)
T['final_log_b'] = best_w * T['direct_log_b'] + (1 - best_w) * T['prior_log_b']

sub = T[['example_id']].copy()
sub['boardings'] = np.clip(np.expm1(T['final_log_b'].to_numpy()), 0, None)
assert len(sub) == len(test_raw) == 157194 and sub['boardings'].notna().all()
sub = test_raw[['example_id']].merge(sub, on='example_id', how='left')
OUT_DIR.mkdir(parents=True, exist_ok=True)
sub.to_csv(OUT_DIR / 'submission.csv', index=False)

for nm, col in [('submission_direct.csv', 'direct_log_b'), ('submission_prior.csv', 'prior_log_b')]:
    s2 = T[['example_id']].copy()
    s2['boardings'] = np.clip(np.expm1(T[col].to_numpy()), 0, None)
    test_raw[['example_id']].merge(s2, on='example_id').to_csv(OUT_DIR / nm, index=False)

daily = T.assign(pred=sub.set_index('example_id').loc[T['example_id'], 'boardings'].to_numpy()).groupby('date')['pred'].sum()
print(f"\n[테스트 sanity] 7/10(금) {daily['2026-07-10']:,.0f} / 7/17(제헌절) {daily['2026-07-17']:,.0f} / "
      f"7/19(일) {daily['2026-07-19']:,.0f} / 9/25(추석) {daily['2026-09-25']:,.0f} / 9/28(월) {daily['2026-09-28']:,.0f}")
print(f"[저장 완료] {OUT_DIR / 'submission.csv'} ({len(sub):,}행)")
