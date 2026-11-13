"""모델 학습용 피처를 만든다 (SINCON 2026).

핵심 함수: make_dataset(cutoff, target_start, target_end)
    - cutoff까지의 train을 "학습 데이터", 그 이후 구간(또는 test)을 "예측 대상"으로 피처를 만든다.
    - 검증할 때는 cutoff를 바꿔 같은 함수를 부르면 되므로, 검증과 최종 예측이 같은 절차를 따른다.

누수 방지 원칙
    1. 승차량으로 만든 통계(타깃 인코딩)는 cutoff 이전 데이터로만 계산한다.
    2. 학습 행에는 "자기 달을 뺀" 통계(월 단위 out-of-fold)를 붙여, 자기 정답을 보지 않게 한다.
    3. 시차 피처(어제 승차량 등)는 쓰지 않는다. 예측 기간에는 값이 없기 때문.
    4. station_events의 spike 행(승차량으로 찾은 행사일)은 피처로 쓰지 않는다.

피처 그룹
    - 시간: hour, dow, month, is_offday, 공휴일·연휴 관련 (calendar_features)
    - 역: station_id(범주), line, multi_line, days_since_open(신규 개통 역만), time_idx(경과일)
    - 외부: 방학 비율, 대학 학기, 날씨(그 시간·직전 3시간), 행사(역별), 검단 인구
    - 타깃 통계: 역×요일×시간 / 역×휴일여부×시간 / 역×시간 log 평균, 하차/승차 비율

사용법 (프로젝트 루트에서):
    python src/preprocess.py            # 최종용(학습 ~2026-06-30, 예측 test) 피처를 data/processed/에 저장
"""
from pathlib import Path

import numpy as np
import pandas as pd

def _find(marker, local):
    """로컬 폴더가 있으면 그것을, 없으면 Kaggle 입력(/kaggle/input)에서 marker 파일이 있는 폴더를 쓴다."""
    if (local / marker).exists():
        return local
    hits = sorted(Path("/kaggle/input").glob(f"**/{marker}"))
    return hits[0].parent if hits else local


RAW = _find("train.csv", Path("data/raw"))
EXT = _find("weather_hourly.csv", Path("data/external"))
DER = _find("calendar_features.csv", Path("data/derived"))
OUT = Path("data/processed")

USE_WX_EXTRA = True
TARGET = "y"  # log1p(boardings)
# 날짜 단위 행사 열은 모든 역에 붙으므로 빼고, 역 단위 행사(event_*)만 쓴다.
# vac_estimated는 데이터 설명용 표시라 피처가 아니다.
# eff_weekday·eff_is_wknd는 daytype·is_offday와 같은 정보라 뺀다.
CAL_DROP = ["holiday_name", "ssg_home_game", "ssg_start_hour", "pentaport", "vac_estimated",
            "eff_weekday", "eff_is_wknd"]
# 행사 종류 코드 (공연·축제는 경기와 끝나는 시각이 달라 종류를 따로 알려 준다)
EVENT_TYPES = {"bb": 1, "concert": 2, "fc": 3, "beer": 4, "pentaport": 5, "fest": 6}
EVENT_EXACT = {"bb_pre": 8, "bb_cancel": 9}  # 시범경기, 취소 경기는 야구와 따로 구분


def event_type_code(key):
    return EVENT_EXACT.get(key) or next((v for p, v in EVENT_TYPES.items() if key.startswith(p)), 7)
# 검단 인구가 영향을 주는 역: 인천1호선 연장 3개 역 + 인천2호선 검단 구간 4개 역
GUMDAN_STATIONS = [3107, 3108, 3109, 3201, 3202, 3203, 3204]


# ------------------------------------------------------------------ 불러오기
def load_raw():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
    test = pd.read_csv(RAW / "test.csv", parse_dates=["date"])
    meta = pd.read_csv(RAW / "metaData.csv", parse_dates=["first_date", "last_date"])
    return train, test, meta


def load_external():
    cal = pd.read_csv(DER / "calendar_features.csv", parse_dates=["date"]).drop(columns=CAL_DROP)
    wx = pd.read_csv(EXT / "weather_hourly.csv", parse_dates=["date"])
    wx = wx.sort_values(["date", "hour"]).reset_index(drop=True)
    wx["precip_3h_mm"] = wx.precip_mm.rolling(3, min_periods=1).sum()  # 그 시간 포함 직전 3시간
    extra = EXT / "weather_extra_hourly.csv"
    wcols = ["precip_mm", "precip_3h_mm", "temp_c"]
    if USE_WX_EXTRA and extra.exists():  # 습도·구름량·일사량
        ex = pd.read_csv(extra, parse_dates=["date"])
        wx = wx.merge(ex, on=["date", "hour"], how="left")
        wcols += ["humidity", "cloud_pct", "radiation"]  # 검증에서 이득이 있던 3개만 (체감온도·풍속·날씨코드는 효과 없음)
    ev = pd.read_csv(DER / "station_events.csv", parse_dates=["date"])
    ev = ev[ev.source != "spike"].copy()  # 원칙 4
    ev["event_type"] = ev.ev_key.fillna("").map(event_type_code)
    return cal, wx[["date", "hour"] + wcols], ev


# ------------------------------------------------------------------ 피처
def add_static(df, meta):
    m = meta.copy()
    m["multi_line"] = m.groupby("station_name").station_id.transform("size").gt(1).astype(int)
    df = df.merge(m[["station_id", "canonical_line", "multi_line", "first_date"]], on="station_id", how="left")
    # 기존 역의 first_date는 실제 개통일이 아니라 데이터 시작일(2025-01-01)이라,
    # 그대로 쓰면 모든 역의 '시간 흐름' 지표가 된다 → 데이터 기간 중 개통한 역에만 적용
    new_station = df.first_date > pd.Timestamp("2025-01-01")
    df["days_since_open"] = np.where(new_station, (df.date - df.first_date).dt.days.clip(lower=0), -1)
    # 시간 흐름(전반적인 수요 추세). 트리 모델은 학습 범위 밖 값을 '가장 최근 수준'으로 다룬다.
    df["time_idx"] = (df.date - pd.Timestamp("2025-01-01")).dt.days
    df = df.rename(columns={"canonical_line": "line"}).drop(columns="first_date")
    # 환승역·외부 노선 환승·대학역 표시 (역 정보 파일, 없으면 0)
    sm_path = EXT / "station_master.csv"
    cols = ["is_transfer", "is_ext_transfer", "is_univ_station"]
    if sm_path.exists():
        df = df.merge(pd.read_csv(sm_path)[["station_id"] + cols], on="station_id", how="left")
    for c in cols:
        df[c] = df[c].fillna(0).astype(int) if c in df else 0
    return df


def neighbor_pairs(meta):
    """같은 노선에서 바로 옆 역 (station_id가 노선 안에서 연속). 행사장 인접역으로 승차가 번진다
    (예: 아시아드 공연 후 서구청역 22~23시가 평소의 18~37배)."""
    rows = []
    for _, g in meta.sort_values("station_id").groupby("canonical_line"):
        ids = g.station_id.tolist()
        for a, b in zip(ids, ids[1:]):
            rows += [(a, b), (b, a)]
    return pd.DataFrame(rows, columns=["station_id", "nb_id"])


def add_calendar(df, cal, wx, ev, meta):
    df = df.merge(cal, on="date", how="left")
    df["month"] = df.date.dt.month
    # 검단 인구는 계속 늘기만 하는 값이라 모든 역에 붙이면 '시간 흐름' 대용이 된다 → 검단 역에만
    df["gumdan_pop_adj"] = np.where(df.station_id.isin(GUMDAN_STATIONS), df.gumdan_pop_adj, 0.0)
    df = df.merge(wx, on=["date", "hour"], how="left")

    # 역별 행사: 그날 행사 여부, 시작 시각 기준 몇 시간 지났는지 (경기 종료 무렵 승차가 몰림)
    e = ev.groupby(["date", "station_id"]).agg(event_any=("source", "size"),
                                                event_start_hour=("start_hour", "max"),
                                                event_end_hour=("ev_end", "max"),
                                                event_type=("event_type", "min")).reset_index()
    e["event_any"] = 1
    df = df.merge(e, on=["date", "station_id"], how="left")
    df["event_any"] = df.event_any.fillna(0).astype(int)
    df["event_type"] = df.event_type.fillna(0).astype(int)
    df["event_start_hour"] = df.event_start_hour.fillna(-1).astype(int)
    df["event_end_hour"] = df.event_end_hour.fillna(-1.0)
    known = (df.event_any == 1) & (df.event_start_hour >= 0)
    df["hours_from_event"] = np.where(known, df.hour - df.event_start_hour, -99)
    # 종료 시각 기준: 공연·페스티벌은 시작이 이르고 길어서 종료 기준이 승차 시각을 더 잘 설명한다
    known_end = (df.event_any == 1) & (df.event_end_hour >= 0)
    df["hours_from_end"] = np.where(known_end, df.hour - np.floor(df.event_end_hour), -99)

    # 인접역 행사 (같은 노선 바로 옆 역)
    nb = neighbor_pairs(meta).merge(
        e.rename(columns={"station_id": "nb_id"})[["date", "nb_id", "event_type", "event_end_hour"]], on="nb_id")
    nb = nb.groupby(["date", "station_id"]).agg(adj_event_type=("event_type", "min"),
                                                 adj_end_hour=("event_end_hour", "max")).reset_index()
    df = df.merge(nb, on=["date", "station_id"], how="left")
    df["adj_event_any"] = df.adj_event_type.notna().astype(int)
    df["adj_event_type"] = df.adj_event_type.fillna(0).astype(int)
    adj_known = (df.adj_event_any == 1) & df.adj_end_hour.notna()
    df["adj_hours_from_end"] = np.where(adj_known, df.hour - np.floor(df.adj_end_hour.fillna(0)), -99)
    df = df.drop(columns="adj_end_hour")
    return df


def daytype(df):
    """요일 7분류, 공휴일은 일요일(6)로."""
    return np.where(df.is_holiday == 1, 6, df.dow)


TE_KEYS = {
    "te_st_dt_h": ["station_id", "daytype", "hour"],
    "te_st_off_h": ["station_id", "is_offday", "hour"],
    "te_st_h": ["station_id", "hour"],
    "te_st": ["station_id"],
}


def _stats(hist):
    """hist(로그 승차·하차 포함)로 그룹별 평균을 만든다."""
    out = {}
    for name, keys in TE_KEYS.items():
        out[name] = hist.groupby(keys).y.mean().rename(name)
    out["alight_ratio"] = (hist.groupby(["station_id", "hour"]).ya.mean()
                           - hist.groupby(["station_id", "hour"]).y.mean()).rename("alight_ratio")
    return out


def _join_stats(df, stats):
    for name, s in stats.items():
        keys = TE_KEYS.get(name, ["station_id", "hour"])
        df = df.join(s, on=keys)
    # 빈 조합은 더 넓은 그룹 값으로 채움
    df["te_st_dt_h"] = df.te_st_dt_h.fillna(df.te_st_off_h).fillna(df.te_st_h)
    df["te_st_off_h"] = df.te_st_off_h.fillna(df.te_st_h)
    df["te_st_h"] = df.te_st_h.fillna(df.te_st)
    return df


def add_target_stats(train_part, target_part):
    """학습 행: 월 단위 out-of-fold 통계 / 예측 행: 학습 구간 전체 통계."""
    month = train_part.date.dt.to_period("M")
    pieces = []
    for m in month.unique():
        mask = month == m
        pieces.append(_join_stats(train_part[mask].copy(), _stats(train_part[~mask])))
    train_out = pd.concat(pieces).loc[train_part.index]
    target_out = _join_stats(target_part.copy(), _stats(train_part))
    return train_out, target_out


# ------------------------------------------------------------------ 데이터셋
def make_dataset(cutoff="2026-06-30", target_start=None, target_end=None, data=None, clean=True):
    """cutoff까지 학습, (target_start ~ target_end) 구간을 예측 대상으로 하는 피처를 만든다.

    target_start가 None이면 test.csv가 예측 대상 (최종 예측).
    반환: (train_df, target_df, feature_cols)  — train_df에는 y(로그 타깃)가 있다.
    """
    train, test, meta = data or load_raw()
    cal, wx, ev = load_external()
    cutoff = pd.Timestamp(cutoff)

    hist = train[train.date <= cutoff].copy()
    if clean:
        # 행사 목록으로 설명되지 않는 이상치 역·날짜는 학습에서 뺀다 (검증/예측 대상 행은 그대로)
        od = pd.read_csv(DER / "outlier_days.csv", parse_dates=["date"])[["station_id", "date"]].assign(_o=1)
        hist = hist.merge(od, on=["station_id", "date"], how="left")
        hist = hist[hist._o.isna()].drop(columns="_o")
        if cutoff >= pd.Timestamp("2025-09-01"):
            # 검단 연장(2025-06-28 개통)으로 수요 체계가 바뀐 계양·귤현의 개통 전 기록은 뺀다
            hist = hist[~(hist.station_id.isin([3110, 3111]) & (hist.date < pd.Timestamp("2025-06-28")))]
    if target_start is None:
        target = test.copy()
    else:
        target = train[(train.date >= target_start) & (train.date <= target_end)].copy()
        # 검증 대상 역은 cutoff 시점에 기록이 있는 역만 (예측 기간과 같은 조건)
        target = target[target.station_id.isin(hist.station_id.unique())]

    frames = []
    for part in (hist, target):
        part = add_static(part, meta)
        part = add_calendar(part, cal, wx, ev, meta)
        part["daytype"] = daytype(part)
        frames.append(part)
    hist, target = frames

    hist["y"] = np.log1p(hist.boardings)
    hist["ya"] = np.log1p(hist.alightings)
    hist, target = add_target_stats(hist.reset_index(drop=True), target.reset_index(drop=True))
    if "boardings" in target:
        target["y"] = np.log1p(target.boardings)

    feats = feature_columns(hist)
    return hist, target, feats


def feature_columns(df):
    drop = {"date", "boardings", "alightings", "y", "ya", "example_id"}
    return [c for c in df.columns if c not in drop]


CATEGORICAL = ["station_id", "line"]


def main():
    tr, te, feats = make_dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(OUT / "train_features.parquet", index=False)
    te.to_parquet(OUT / "test_features.parquet", index=False)
    print(f"학습 {tr.shape}, 예측 {te.shape}, 피처 {len(feats)}개")
    print(feats)


if __name__ == "__main__":
    main()
