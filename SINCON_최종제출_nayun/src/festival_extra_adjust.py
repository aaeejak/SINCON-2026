"""팀원 축제 보정표(src/postprocess/event_adjust.py RULES)에서 빠진 실제 개최일을 0.5배로 채운다.

- 부평풍물대축제 제30회: 2026-10-02(금)~10-04(일) 3일 (팀원 표는 10/3·4만). 10/2에 10/3·4 보정값의 0.5배.
- 드림파크 국화축제: 2026-10-16(금)~10-25(일) 10일 (팀원 표는 주말 4일만). 2025년 금요일(10/24)에도 1.26배 증가가 있어 금요일 10/16·10/23에 주말 보정값의 0.5배.
금요일 효과 크기(0.5배)는 가정이다. 개최일 출처는 data/external/SOURCES.md.
사용법: python src/festival_extra_adjust.py 입력.csv 출력.csv [pungmul] [chrysanthemum]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocess"))
import event_adjust as E  # noqa: E402

EXTRA = {"pungmul": ("부평풍물대축제", ["2026-10-02"]), "chrysanthemum": ("드림파크 국화축제", ["2026-10-16", "2026-10-23"])}


def adjust(sub, test, which, factor=0.5):
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"])
    lp = np.log1p(k.boardings.to_numpy(dtype=float)).copy()
    n = 0
    for key in which:
        prefix, days = EXTRA[key]
        for name, st, _, prof in [r for r in E.RULES if r[0].startswith(prefix)]:
            m = ((k.station_id == st) & k.date.isin(pd.to_datetime(days)) & k.hour.isin(list(prof))).to_numpy()
            a = k.loc[m, "hour"].map(prof).to_numpy()
            lp[m] += factor * np.log1p(E.trend_of(name) * np.expm1(a))
            n += int(m.sum())
    k["boardings"] = np.expm1(lp).clip(0)
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left"), n


def main():
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    out, n = adjust(sub, te, [w for w in sys.argv[3:] if w in EXTRA])
    assert len(out) == len(te) and out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
