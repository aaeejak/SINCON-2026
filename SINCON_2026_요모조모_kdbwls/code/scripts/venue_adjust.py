"""대형 공연장·축제장 주변 역 보정: 과거 같은 장소 행사일의 역x시간 '같은 요일 평소 대비 로그 차'(실제, 여러 날 평균)
− 예측 파일의 2026 행사일 같은 값, 에 배율을 곱해 더한다. 행사장 역 자체와 인접역(관객이 나눠 타는 역)을 함께 본다."""
import sys, numpy as np, pandas as pd
EV = pd.read_csv("external/events_2025_2026.csv", parse_dates=["date"])
H = set(pd.to_datetime(pd.read_csv("holidays_kr_2025_2026.csv").date))
KBO = set(pd.read_csv("external/kbo_munhak_2025_2026.csv", parse_dates=["date"]).date)
EVD = set(EV.date)
GROUPS = {  # 이름: (행사 역, 함께 볼 역, 2026 대상 날 제외 목록, SSG 경기일을 기준일에서 뺄지)
    "아시아드 공연": (3209, [3209, 3210, 3208], [], False),
    "문학 공연": (3127, [3127, 3128, 3126], ["2026-08-29", "2026-08-30"], True),   # 낮 시작 페스티벌은 제외
    "송도 축제": (3139, [3138], [], False),   # 행사장 역(3139)은 모델이 축제별로 이미 학습 → 인접역만
}
SHRINK_VENUE, SHRINK_NB = 0.4, 0.7   # 행사장 역은 모델에 행사 피처가 있어 약하게, 인접역은 피처가 없어 강하게
def rel(df, st, x, ex_kbo=True):
    ref = [t for t in pd.date_range(x - pd.Timedelta(days=35), x + pd.Timedelta(days=35))
           if t.dayofweek == x.dayofweek and t != x and t not in EVD and t not in H and (not ex_kbo or t not in KBO)]
    s = df[df.station_id == st]
    r = s[s.date.isin(ref)].groupby("hour").lp.mean(); v = s[s.date == x].set_index("hour").lp
    return (v - r) if len(r) else None
def adjust(sub, test, verbose=False, okf=True):
    d = pd.read_csv("data/train.csv", parse_dates=["date"]); d["lp"] = np.log1p(d.boardings)
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"]); k["lp"] = np.log1p(k["boardings"]); base = k.copy()
    for name, (vst, sts, skip, exk) in GROUPS.items():
        e = EV[(EV.station_id == vst) & (EV.kind != "football")]
        past = sorted(set(e[e.date <= "2026-06-30"].date)); fut = [x for x in sorted(set(e[e.date >= "2026-07-01"].date)) if str(x.date()) not in skip]
        for st in sts:
            T = pd.concat([r for x in past if (r := rel(d, st, x, exk)) is not None], axis=1).mean(axis=1)
            sh = SHRINK_VENUE if st == vst else SHRINK_NB
            for x in fut:
                cur = rel(base, st, x, exk)
                if cur is None: continue
                c = (T - cur).reindex(cur.index).fillna(0).clip(-0.8, 2.5) * sh
                m = ((k.station_id == st) & (k.date == x)).to_numpy()
                k.loc[m, "lp"] += k.loc[m, "hour"].map(c).fillna(0).to_numpy()
                if verbose: print(name, st, x.date(), "저녁(19~23시) 평균 보정", round(float(c.loc[19:23].mean()), 2), "최대", round(float(c.max()), 2))
    # 행사 목록에 없던 문학 행사: OBJET K-POP FESTA 2026 (9/26~27, 13시 시작). 모델이 모르는 행사라
    # 과거 문학 공연일 상승폭(T)을 12시 이후에만 0.5배 그대로 더한다 (추석 보정과 겹치지 않게 '차이'가 아니라 '상승폭'을 더함)
    e = EV[(EV.station_id == 3127) & (EV.kind == "concert") & (EV.date <= "2026-06-30")]
    for st in ((3127, 3128, 3126) if okf else ()):
        T = pd.concat([r for x in sorted(set(e.date)) if (r := rel(d, st, x, True)) is not None], axis=1).mean(axis=1)
        T = T.where(T.index >= 12, 0).clip(lower=0) * 0.5
        for x in pd.to_datetime(["2026-09-26", "2026-09-27"]):
            m = ((k.station_id == st) & (k.date == x)).to_numpy()
            k.loc[m, "lp"] += k.loc[m, "hour"].map(T).fillna(0).to_numpy()
            if verbose: print("OKF", st, x.date(), "최대", round(float(T.max()), 2))
    k["boardings"] = np.expm1(k.lp.clip(lower=0))
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left")
if __name__ == "__main__":
    te = pd.read_csv("data/test.csv"); out = adjust(pd.read_csv(sys.argv[1]), te, verbose=True)
    assert out.boardings.notna().all() and len(out) == 157194
    out.to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
