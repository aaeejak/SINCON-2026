"""SSG 문학 홈경기 종료 시간대 보정 (문학경기장 3127).
과거 경기(2025-03~2026-06)의 '같은 요일 비경기일 대비 로그 상승폭'을 종료 시각 기준 시간대(rel=-2..2)별로
관중(z)·종료 분(f=종료시각의 분/60)·7~10월·주말로 회귀 → 2026 경기의 목표 상승폭.
예측 파일의 같은 값(예측 비경기일 대비)과의 차이를 K배 더한다.
검증(경기 단위 leave-one-out, 검증 A·B 예측): 상승폭 RMSE A 1.003→0.73, B 0.758→0.60 (K 0.5~0.7).
종료 시각·관중: KBO 게임센터 경기 기록 (external/ssg_munhak_game_times.csv, nayun 수집)."""
import numpy as np, pandas as pd, sys
ST, RELS, K = 3127, range(-2, 3), 0.6
COLS = ["z", "f", "summer", "wk"]
g = pd.read_csv("external/ssg_munhak_game_times.csv", parse_dates=["date"]).drop_duplicates("date").set_index("date")
H = set(pd.to_datetime(pd.read_csv("holidays_kr_2025_2026.csv").date))
ev = pd.read_csv("external/events_2025_2026.csv", parse_dates=["date"]); EVD = set(ev[ev.station_id.isin([3127, 3128])].date)
sched = pd.read_csv("external/kbo_munhak_2025_2026.csv", parse_dates=["date"]); SCHED = set(sched.date)
OKF = {pd.Timestamp("2026-09-26"), pd.Timestamp("2026-09-27")}

def uplift(x):
    """x: date, hour, l (log1p) for station 3127. 경기일별 rel 시간대 상승폭."""
    non = x[~x.date.isin(set(g.index) | SCHED | H | EVD | OKF)]
    out = []
    for dd in sorted(set(x.date) & set(g.index)):
        r = g.loc[dd]
        ref = non[(non.date.dt.dayofweek == dd.dayofweek) & ((non.date - dd).abs().dt.days <= 35)].groupby("hour").l.mean()
        day = x[x.date == dd].set_index("hour").l
        for rel in RELS:
            h = int(np.floor(r.end_h)) + rel
            if h in day.index and h in ref.index:
                out.append((dd, rel, h, day[h] - ref[h], (r.crowd - 16568) / 5000, r.end_h % 1,
                            float(dd.month in (7, 8, 9, 10)), float(dd.dayofweek >= 5)))
    return pd.DataFrame(out, columns=["date", "rel", "hour", "u", "z", "f", "summer", "wk"])

def coefs():
    d = pd.read_csv("data/train.csv", parse_dates=["date"]); d = d[d.station_id == ST]; d["l"] = np.log1p(d.boardings)
    A = uplift(d)
    return {rel: np.linalg.lstsq(np.c_[np.ones((A.rel == rel).sum()), A.loc[A.rel == rel, COLS].values],
                                 A.loc[A.rel == rel, "u"].values, rcond=None)[0] for rel in RELS}

def adjust(sub, test, k=K):
    W = coefs()
    t = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left"); t["date"] = pd.to_datetime(t.date)
    t["lp"] = np.log1p(t.boardings)
    x = t[t.station_id == ST][["date", "hour", "lp"]].rename(columns={"lp": "l"})
    P = uplift(x)
    P["t"] = [W[r.rel] @ np.r_[1, [getattr(r, c) for c in COLS]] for r in P.itertuples()]
    P["c"] = (k * (P.t - P.u)).clip(-1.5, 1.5)
    m = t[["date", "station_id", "hour"]].merge(P[["date", "hour", "c"]].assign(station_id=ST), how="left").c.fillna(0).values
    t["lp"] = t.lp + m
    t["boardings"] = np.expm1(t.lp.clip(lower=0))
    return sub[["example_id"]].merge(t[["example_id", "boardings"]], on="example_id", how="left"), P

if __name__ == "__main__":
    te = pd.read_csv("data/test.csv"); sub = pd.read_csv(sys.argv[1])
    out, P = adjust(sub, te, float(sys.argv[3]) if len(sys.argv) > 3 else K)
    assert out.boardings.notna().all() and len(out) == len(sub)
    out.to_csv(sys.argv[2], index=False)
    print("경기", P.date.nunique(), "행", len(P), "rel별 평균 보정", P.groupby("rel").c.mean().round(3).to_dict(), "저장", sys.argv[2])
