"""SSG 경기 점수 차 보정 계수: 종료 시각 보정(ssg_end_profile) 후 남은 잔차를 |점수 차|로 설명.

rel(= 시간 - 종료 시각의 정각) -1, 0, 1, 2 별로 r2 = r - (a_rel + b_rel*관중 z) ~ c_rel + d_rel*|점수 차| 를 능형 회귀(d에만 벌점)로 맞춘다.
검증(78경기, 경기 단위 leave-one-out): 경기일 문학경기장 RMSE 0.5212 → 0.5139.
입력: data/external/ssg_munhak_game_results.csv, data/derived/ssg_end_profile.csv 출력: data/derived/ssg_margin_profile.csv (약 4분)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_ssg_profile as F  # noqa: E402


def main():
    d = pd.read_csv("data/external/ssg_munhak_game_results.csv", parse_dates=["date"])
    R = F.residuals().merge(d[["date", "crowd", "end_h", "margin"]], on="date")
    R["rel"] = (R.hour - np.floor(R.end_h)).astype(int)
    R["z"] = (R.crowd - F.Z_MEAN) / F.Z_SD
    R["absm"] = R.margin.abs()
    prof = pd.read_csv("data/derived/ssg_end_profile.csv").set_index("rel")
    out = []
    for k in (-1, 0, 1, 2):
        a = R[R.rel == k]
        r2 = a.r.to_numpy() - (prof.loc[k, "a"] + prof.loc[k, "b"] * a.z.to_numpy())
        X = np.c_[np.ones(len(a)), a.absm.to_numpy()]
        w = np.linalg.solve(X.T @ X + np.diag([0.0, 3.0]), X.T @ r2)
        out.append(dict(rel=k, c=w[0], d=w[1], n=len(a)))
    pd.DataFrame(out).round(4).to_csv("data/derived/ssg_margin_profile.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
