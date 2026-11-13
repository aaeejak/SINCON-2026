"""SSG 문학 홈경기 종료 시각 보정 (문학경기장 3127, 2026-07~10월 홈경기 35경기).

모델은 경기 종료를 '시작 + 3.2시간'으로 추정해서, 실제 종료 시각과 어긋나면 종료 시간대의 승차를 과소·과대 예측한다.
KBO 게임센터(koreabaseball.com)의 경기별 실제 종료 시각·관중 수(data/external/ssg_munhak_game_times.csv)로,
rel = 시간 - 종료 시각의 정각 이 -1, 0, 1, 2인 행에 (a_rel + b_rel * 관중 z) * shrink 를 log1p 공간에서 더한다.
계수는 data/derived/ssg_end_profile.csv (src/fit_ssg_profile.py가 검증 2개 구간의 잔차로 학습; rel 0은 +0.67, rel 1은 +0.52).
검증(78경기, 경기 단위 leave-one-out): 경기일 문학경기장 RMSE 0.581 → 0.529.
사용법: python src/ssg_adjust.py 입력.csv 출력.csv [shrink=0.7] [margin]   (margin: 점수 차 보정 추가, src/fit_ssg_margin.py)
"""
import sys

import numpy as np
import pandas as pd

Z_MEAN, Z_SD = 16568.0, 5000.0


def adjust(sub, test, shrink=0.7, margin=False):
    prof = pd.read_csv("data/derived/ssg_end_profile.csv").set_index("rel")
    mp = pd.read_csv("data/derived/ssg_margin_profile.csv").set_index("rel") if margin else None
    res = pd.read_csv("data/external/ssg_munhak_game_results.csv", parse_dates=["date"]).drop_duplicates("date").set_index("date") if margin else None
    g = pd.read_csv("data/external/ssg_munhak_game_times.csv", parse_dates=["date"])
    g = g[g.date >= "2026-07-01"].drop_duplicates("date")
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"])
    k["lp"] = np.log1p(k.boardings)
    n = 0
    for r in g.itertuples():
        for rel in prof.index:
            h = int(np.floor(r.end_h)) + int(rel)
            m = ((k.station_id == 3127) & (k.date == r.date) & (k.hour == h)).to_numpy()
            if m.any():
                add = prof.loc[rel, "a"] + prof.loc[rel, "b"] * (r.crowd - Z_MEAN) / Z_SD
                if margin and rel in mp.index:  # 점수 차 보정: 접전이면 종료 후 승차 집중, 큰 점수 차면 종료 전 이탈
                    add += mp.loc[rel, "c"] + mp.loc[rel, "d"] * abs(res.loc[r.date, "margin"])
                k.loc[m, "lp"] += shrink * add
                n += int(m.sum())
    k["boardings"] = np.expm1(k.lp).clip(lower=0)
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left"), n


def main():
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    sh = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "margin" else 0.7
    out, n = adjust(sub, te, sh, margin="margin" in sys.argv)
    assert out.boardings.notna().all() and len(out) == len(sub)
    out.to_csv(sys.argv[2], index=False)
    print(f"보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
