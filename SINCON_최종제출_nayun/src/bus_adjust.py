"""역 주변 버스 이용량 변화 보정 (팀원 제공 data/external/station_bus_month_2024_2026_v2.csv, 반경 500m 버스 월별 승하차 일평균).

보정 = k × dc, dc = [log(해당 월 버스 일평균 / 2026-04~06 평균) 의 역 간 평균을 뺀 값].
7~9월 2026 실제 버스 자료가 있는 달만 적용하고 10월(자료 없음)은 그대로 둔다. 역 평균을 빼므로 전체 계절·수준 변화는 건드리지 않고 '다른 역보다 버스가 더 늘거나 준 정도'만 반영한다.
검증(두 구간 교차, 역×월 잔차 ~ 버스 변화): 상관 0.60~0.65, 기울기 0.23~0.31. 이득의 대부분은 검단·계양·귤현(팀 검단 보정과 겹침)에서 나왔고 그 외 66개 역에서는 불안정.
사용법: python src/bus_adjust.py 입력.csv 출력.csv [k=0.25] [exclude5]   (exclude5: 검단 3역·계양·귤현 제외)
"""
import sys

import numpy as np
import pandas as pd

ANCHOR = ["2026-04", "2026-05", "2026-06"]
MONTHS = {7: "2026-07", 8: "2026-08", 9: "2026-09"}
SPECIAL = [3107, 3108, 3109, 3110, 3111]


def corrections():
    d = pd.read_csv("data/external/station_bus_month_2024_2026_v2.csv")
    b = d[d.radius_m == 500].set_index(["station_id", "ym"]).bus_total_per_day
    base = b.unstack("ym")[ANCHOR].replace(0, np.nan).mean(axis=1)
    out = {}
    for m, ym in MONTHS.items():
        s = b.xs(ym, level="ym").replace(0, np.nan)
        dl = np.log(s / base)
        out[m] = (dl - dl.mean()).fillna(0.0)
    return out


def adjust(sub, test, k=0.25, exclude5=False):
    c = corrections()
    t = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    t["date"] = pd.to_datetime(t["date"])
    lp = np.log1p(t.boardings.to_numpy(dtype=float))
    for m, dc in c.items():
        sel = (t.date.dt.month == m).to_numpy()
        corr = t.station_id.map(dc).fillna(0.0).to_numpy()
        if exclude5:
            corr = np.where(t.station_id.isin(SPECIAL), 0.0, corr)
        lp[sel] += k * corr[sel]
    t["boardings"] = np.expm1(lp)
    return sub[["example_id"]].merge(t[["example_id", "boardings"]], on="example_id", how="left")


def main():
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    k = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "exclude5" else 0.25
    out = adjust(sub, te, k, exclude5="exclude5" in sys.argv)
    assert out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"저장 {sys.argv[2]} (k={k})")


if __name__ == "__main__":
    main()
