"""작은 보정 두 가지 (근거가 약해 시험용).

ftuned : F_v10_stack 안의 우리 몫을 검증 B로 정한 비율(LGBM 0.80 + XGB 0.12 + Cat 0.08)로 바꾼 F와의 차이를
         팀 블렌드에서 F 비중(OPT json 0.405)만큼 더한다: x + 0.405·(log F_tuned − log F). 검증 B에서 우리 몫 0.1696 → 0.1682.
         gb_v6_orig 안의 실제 F 비중은 모르므로 OPT 값을 쓴 근사.
hangul : 2026-10-09(금, 한글날, 3일 연휴 첫날) 전 역 수준을 log -0.035.
         예측은 같은 요일 평일의 0.632배인데, 과거 금요일 공휴일(2025-08-15 0.589, 2025-10-03 0.569)보다 높다 → 차이(-0.07)의 절반.
v3swap : 팀 블렌드의 nayun_v3 자리(OPT json 비중 0.085)를 우리 검증 최적 혼합(ours_tuned = 0.80·LGBM v9 + 0.12·XGB v9 + 0.08·Cat v9)으로 바꾼 근사.
         x + 0.085·(log ours_tuned − log lgbm_v3). 검증 B: v3 0.1723 → ours_tuned 0.1682. gb 안의 실제 v3 비중은 모름.
사용법: python src/misc_adjust.py 입력.csv 출력.csv [ftuned] [hangul] [v3swap]
"""
import sys

import numpy as np
import pandas as pd

F_W = 0.405


def main():
    te = pd.read_csv("data/raw/test.csv", parse_dates=["date"])
    L = lambda p: np.log1p(pd.read_csv(p).set_index("example_id").boardings.reindex(te.example_id).to_numpy())
    x = L(sys.argv[1]).copy()
    opts = set(sys.argv[3:])
    if "ftuned" in opts:
        x += F_W * (L("outputs/archive/candidates_unscored/cand_F_tuned.csv") - L("outputs/archive/candidates_unscored/cand_F_v10_stack.csv"))
    if "v3swap" in opts:
        M = "outputs/models/"
        tuned = 0.80 * L(M + "lgbm_v9.csv") + 0.12 * L(M + "xgb_v9.csv") + 0.08 * L(M + "catboost_v9.csv")
        x += 0.085 * (tuned - L(M + "lgbm_v3.csv"))
    if "hangul" in opts:
        x[(te.date == "2026-10-09").to_numpy()] -= 0.035
    out = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(x).clip(0)})
    assert len(out) == 157194 and np.isfinite(out.boardings).all()
    out.to_csv(sys.argv[2], index=False)
    print("저장", sys.argv[2])


if __name__ == "__main__":
    main()
