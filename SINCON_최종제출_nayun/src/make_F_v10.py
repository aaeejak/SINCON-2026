"""F_v10_stack(팀 블렌드 OPT의 F 자리, `cand_F_v10_stack.csv`)을 다시 만든다 (약 3분).

ours_v9 = 0.72·lgbm_v9 + 0.18·catboost_v9 + 0.10·mlp_v9
v9파이널 = 0.30·ours_v9 + 0.25·blend + 0.15·blend4 + 0.10·v2 + 0.05·(5fold, post, mlp, lstm)   (팀원 모델 예측: team_inputs/)
D = 0.85·v9파이널 + 0.15·v5data  →  F = D + 0.7·앵커 스태커 보정 (src/stack_anchor.py)
저장본과 최대 차이(log) 3.6e-15 확인 (2026-11-13). 'tuned' 판(cand_F_tuned)은 src/make_F_xgb.py [출력] tuned.
사용법: python src/make_F_v10.py [출력=outputs/archive/candidates_unscored/cand_F_v10_stack.csv]
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
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/archive/candidates_unscored/cand_F_v10_stack.csv"
    ours = 0.72 * L(M / "lgbm_v9.csv") + 0.18 * L(M / "catboost_v9.csv") + 0.10 * L(M / "mlp_v9.csv")
    v9 = (0.30 * ours + 0.25 * L(F / "input_team_blend.csv") + 0.15 * L(F / "input_team_blend4.csv")
          + 0.10 * L(F / "input_team_v2.csv") + 0.05 * L(F / "input_team_5fold.csv") + 0.05 * L(F / "input_team_post.csv")
          + 0.05 * L(F / "input_team_mlp.csv") + 0.05 * L(F / "input_team_lstm.csv"))
    D = 0.85 * v9 + 0.15 * L(F / "input_team_v5data.csv")
    tmp = Path("outputs/_tmp_D_v10.csv")
    pd.DataFrame({"example_id": TEST, "boardings": np.expm1(D).clip(0)}).to_csv(tmp, index=False)
    subprocess.run([sys.executable, "src/stack_anchor.py", str(tmp), out, "0.7"], check=True)
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
