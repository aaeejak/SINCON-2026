"""F(= cand_F_v10_stack)의 'ours_v9' 자리를 XGBoost를 섞은 걸로 바꾼 F를 만든다 (팀 블렌드에서 F 자리만 교체용).

ours_x = 0.7 * ours_v9 + 0.3 * xgb_v9      (ours_v9 = 0.72 LightGBM + 0.18 CatBoost + 0.10 MLP)
v9파이널 = 0.30*ours_x + 0.25*tb + 0.15*b4 + 0.10*v2 + 0.05*(5fold, post, mlp, lstm)
D = 0.85*v9파이널 + 0.15*v5data → F_x = D + 0.7*앵커 스태커 보정 (src/stack_anchor.py)
입력: outputs/models/{ours_v9,xgb_v9}.csv, team_inputs/input_team_*.csv   사용법: python src/make_F_xgb.py [출력 csv] [tuned]   (tuned: ours = 0.80 LightGBM + 0.12 XGBoost + 0.08 CatBoost)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST = pd.read_csv("data/raw/test.csv", usecols=["example_id"]).example_id
M, F = Path("outputs/models"), Path("team_inputs")
L = lambda p: np.log1p(pd.read_csv(p).set_index("example_id").boardings.reindex(TEST).to_numpy())


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/candidates/cand_F_xgb.csv"
    if len(sys.argv) > 2 and sys.argv[2] == "tuned":   # 검증 구간으로 정한 비율 (src/tune_ours_weights.py): MLP는 뺌
        ours = 0.80 * L(M / "lgbm_v9.csv") + 0.12 * L(M / "xgb_v9.csv") + 0.08 * L(M / "catboost_v9.csv")
    else:
        ours = 0.7 * L(M / "ours_v9.csv") + 0.3 * L(M / "xgb_v9.csv")
    v9 = (0.30 * ours + 0.25 * L(F / "input_team_blend.csv") + 0.15 * L(F / "input_team_blend4.csv")
          + 0.10 * L(F / "input_team_v2.csv") + 0.05 * L(F / "input_team_5fold.csv") + 0.05 * L(F / "input_team_post.csv")
          + 0.05 * L(F / "input_team_mlp.csv") + 0.05 * L(F / "input_team_lstm.csv"))
    D = 0.85 * v9 + 0.15 * L(F / "input_team_v5data.csv")
    tmp = Path("outputs/_tmp_Dx.csv")
    pd.DataFrame({"example_id": TEST, "boardings": np.expm1(D).clip(0)}).to_csv(tmp, index=False)
    subprocess.run([sys.executable, "src/stack_anchor.py", str(tmp), out_path, "0.7"], check=True)
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
