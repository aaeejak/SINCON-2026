"""명절 연휴 보정: 과거 명절(실제)의 '일요일 대비 역x시간 로그 차'와 예측의 같은 값을 비교해 차이를 더한다."""
import numpy as np, pandas as pd
d = pd.read_csv("data/train.csv", parse_dates=["date"]); d["l"] = np.log1p(d.boardings)
def sundays(lo, hi, excl):
    return [s for s in pd.date_range(lo, hi) if s.dayofweek == 6 and not (excl[0] <= s <= excl[1])]
def rel_actual(day, lo, hi, excl):
    ref = d[d.date.isin(sundays(lo, hi, excl))].groupby(["station_id", "hour"]).l.mean()
    x = d[d.date == pd.Timestamp(day)].set_index(["station_id", "hour"]).l
    return (x - ref).rename(day)
X = lambda s: (pd.Timestamp(s[0]), pd.Timestamp(s[1]))
# 과거 명절 (연휴 첫날 / 명절 당일 / 마지막날 / 다음 일요일)
ANALOG = {
  "chuseok25": dict(w=0.5, lo="2025-09-01", hi="2025-11-02", excl=X(("2025-10-02","2025-10-12")), eve="2025-10-05", main="2025-10-06", after="2025-10-07", sun="2025-10-12"),
  "seol26":    dict(w=0.25, lo="2026-01-04", hi="2026-03-15", excl=X(("2026-02-13","2026-02-22")), eve="2026-02-16", main="2026-02-17", after="2026-02-18", sun="2026-02-22"),
  "seol25":    dict(w=0.25, lo="2025-01-05", hi="2025-03-02", excl=X(("2025-01-24","2025-02-02")), eve="2025-01-28", main="2025-01-29", after="2025-01-30", sun="2025-02-02"),
}
def target(roles=("eve","main","after","sun"), use=None):
    use = use or ANALOG
    out = {}
    for r in roles:
        vs = pd.concat({k: rel_actual(a[r], a["lo"], a["hi"], a["excl"]) for k, a in use.items()}, axis=1)
        w = pd.Series({k: a["w"] for k, a in use.items()})
        ok = vs.notna()
        out[r] = (vs.fillna(0) * w).sum(axis=1) / (ok * w).sum(axis=1)   # 자료 있는 명절만 가중 평균 (검단 신설역)
    return out

# 토요일 공휴일 (2025-03-01 삼일절, 2026-06-06 현충일): 일요일 대비 +0.21, +0.20. 예측 2026-08-15(토)는 +0.06뿐
ANALOG_SAT = {
  "sam25": dict(w=0.5, lo="2025-02-09", hi="2025-03-30", excl=X(("2025-03-01","2025-03-03")), sathol="2025-03-01"),
  "hyun26": dict(w=0.5, lo="2026-05-10", hi="2026-06-28", excl=X(("2026-06-06","2026-06-07")), sathol="2026-06-06"),
}
ROLE_2026 = {"2026-09-24": "eve", "2026-09-25": "main", "2026-09-26": "after", "2026-09-27": "sun"}
SHRINK = {"eve": 0.85, "main": 0.85, "after": 0.85, "sun": 0.5, "sathol": 0.6}
SUN_2026 = [pd.Timestamp(x) for x in ("2026-09-06", "2026-09-13", "2026-09-20", "2026-10-18", "2026-10-25")]

def adjust(sub, test, station_weight=1.0, include_sat=True, scale=1.0):
    """sub: example_id, boardings (제출 파일). 추석 4일 역x시간 로그 보정."""
    T = target(); T.update(target(("sathol",), ANALOG_SAT))
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"]); k["lp"] = np.log1p(k["boardings"])
    refs = {"chuseok": SUN_2026, "aug": [pd.Timestamp(x) for x in ("2026-07-26", "2026-08-02", "2026-08-09", "2026-08-23", "2026-08-30")]}
    for day, r in {**ROLE_2026, **({"2026-08-15": "sathol"} if include_sat else {})}.items():
        refp = k[k.date.isin(refs["aug" if r == "sathol" else "chuseok"])].groupby(["station_id", "hour"]).lp.mean()
        m = k.date == pd.Timestamp(day)
        x = k[m].set_index(["station_id", "hour"])
        c = (T[r] - (x.lp - refp)).reindex(x.index)
        hm = c.groupby(level="hour").transform("mean")          # 시간대 평균
        c = (hm + station_weight * (c - hm)).fillna(0).clip(-0.8, 0.4).to_numpy()
        k.loc[m, "lp"] = x.lp.to_numpy() + min(1.0, SHRINK[r] * scale) * c
    k["boardings"] = np.expm1(k["lp"].clip(lower=0))
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left")

if __name__ == "__main__":
    import sys
    te = pd.read_csv("data/test.csv"); sub = pd.read_csv(sys.argv[1])
    out = adjust(sub, te); assert out.boardings.notna().all() and len(out) == len(sub)
    out.to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
