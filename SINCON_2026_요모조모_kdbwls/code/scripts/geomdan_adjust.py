"""검단 3개 역(검단호수공원·신검단중앙·아라)과 계양 보정.
기준: 2026-06 실제 평일 하루 평균 × 성숙역 계절(2025년 6월 대비 각 달) × 입주 세대 증가(탄력성 0.5).
예측 파일의 월 평일 평균과의 로그 차이에 배율(0.5)을 곱해 그 역·그 달 전체(모든 시간)에 더한다."""
import sys, numpy as np, pandas as pd
SEAS = {7: -0.069, 8: -0.084, 9: -0.004, 10: 0.004}          # 성숙역 평일 수준, 2025-06 대비 (중앙값)
GROW = {7: 0.037, 8: 0.10, 9: 0.135, 10: 0.16}               # 입주 세대 증가 × 0.5 (60일·120일 입주 가정 평균)
STATIONS = {3107: GROW, 3108: GROW, 3109: GROW, 3110: {7: -0.005, 8: -0.007, 9: -0.01, 10: -0.012}}  # 계양: 연 −5% 추세
SHRINK = 0.5
def adjust(sub, test, verbose=False):
    H = set(pd.to_datetime(pd.read_csv("holidays_kr_2025_2026.csv").date))
    d = pd.read_csv("data/train.csv", parse_dates=["date"])
    d = d[(d.date >= "2026-06-01") & (d.date.dt.dayofweek < 5) & ~d.date.isin(H)]
    jun = d.groupby(["station_id", "date"]).boardings.sum().groupby("station_id").mean()
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left"); k["date"] = pd.to_datetime(k["date"])
    wd = (k.date.dt.dayofweek < 5) & ~k.date.isin(H); lp = np.log1p(k.boardings.to_numpy())
    for st, grow in STATIONS.items():
        for m in (7, 8, 9, 10):
            sel = (k.station_id == st) & (k.date.dt.month == m)
            pred = k[sel & wd].groupby("date").boardings.sum().mean()
            c = SHRINK * (np.log(jun[st]) + SEAS[m] + grow[m] - np.log(pred))
            lp[sel.to_numpy()] += c
            if verbose: print(st, m, "6월 실제", round(jun[st]), "예측", round(pred), "목표", round(jun[st] * np.exp(SEAS[m] + grow[m])), "보정", round(c, 3))
    k["boardings"] = np.expm1(lp)
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left")
if __name__ == "__main__":
    te = pd.read_csv("data/test.csv"); out = adjust(pd.read_csv(sys.argv[1]), te, verbose=True)
    assert out.boardings.notna().all() and len(out) == 157194
    out.to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
