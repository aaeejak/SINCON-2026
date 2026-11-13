"""취소된 문학 행사 되돌리기.
- OKF 2026 (9/26~27): 9/15 연기·취소 발표 → venue_adjust의 OKF 상승분(okf_delta) 전부 제거
- GHOSTIVAL 2026 (10/24~25): 10/21 전면 취소 발표 → 3127·3128·3126의 그날 예측을 같은 요일 평소 날(±28일, 행사·공휴일 제외) 예측 평균으로 교체
사용: python scripts/cancel_adjust.py 입력 출력 [okf를 이미 반만 가진 파일이면 okf_scale]"""
import sys, numpy as np, pandas as pd
EXCL = pd.to_datetime(["2026-09-26", "2026-09-27", "2026-10-03", "2026-10-24", "2026-10-25", "2026-09-12", "2026-09-13", "2026-08-29", "2026-08-30", "2026-09-24", "2026-09-25", "2026-10-09", "2026-10-05"])
def adjust(sub, te, okf_scale=0.0):
    okf = 0.0
    if okf_scale:
        L = lambda f: np.log1p(te[["example_id"]].merge(pd.read_csv(f), on="example_id").boardings.values)
        okf = L("submission_OPT4_final_geo_okf.csv") - L("submission_OPT4_final_geo.csv")
    t = te[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id"); t["date"] = pd.to_datetime(t.date)
    t["l"] = np.log1p(t.boardings.values) - okf_scale * okf
    for day in ["2026-10-24", "2026-10-25"]:
        D = pd.Timestamp(day)
        for st in (3127, 3128, 3126):
            x = t[t.station_id == st]
            ref = x[(x.date.dt.dayofweek == D.dayofweek) & ((x.date - D).abs().dt.days.between(1, 28)) & ~x.date.isin(EXCL)].groupby("hour").l.mean()
            m = (t.station_id == st) & (t.date == D)
            t.loc[m, "l"] = t.loc[m, "hour"].map(ref).values
    t["boardings"] = np.expm1(t.l.clip(lower=0))
    return sub[["example_id"]].merge(t[["example_id", "boardings"]], on="example_id")
if __name__ == "__main__":
    te = pd.read_csv("data/test.csv"); out = adjust(pd.read_csv(sys.argv[1]), te, float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
    assert out.boardings.notna().all() and len(out) == 157194; out.to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
