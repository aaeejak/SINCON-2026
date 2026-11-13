"""최종 제출 파일(cand_K, Public 0.151321)을 입력 파일들로부터 만든다.

cand_K = 0.45 * B + 0.35 * F + 0.20 * blend5   (모두 log1p 공간 가중 평균 후 expm1)
  blend5 : 팀원 블렌드 (team_inputs/input_team_blend5.csv)
  B      : blend5 + 0.22 * (lgbm_v9 - v3)   → blend5의 v3 성분(비중 0.22)을 우리 강화 LightGBM으로 교체
  F      : D에 앵커 스태커 보정(shrink 0.7)을 더한 것 (src/stack_anchor.py)
  D      : 0.85 * v9파이널 + 0.15 * v5data
  v9파이널 : 0.30*ours_v9 + 0.25*tb + 0.15*b4 + 0.10*v2 + 0.05*(5fold, post, mlp, lstm)
  ours_v9 : 0.72*lgbm_v9 + 0.18*catboost_v9 + 0.10*mlp_v9

입력: outputs/models/{lgbm_v3,lgbm_v9,catboost_v9,mlp_v9}.csv, team_inputs/input_team_*.csv, data/raw/test.csv
사용법: python src/make_final.py [출력 csv]   (기본 outputs/submission_final_K.csv, stack_anchor 때문에 약 3분)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST = pd.read_csv("data/raw/test.csv", usecols=["example_id"]).example_id
M, F = Path("outputs/models"), Path("team_inputs")


def L(p):
    return np.log1p(pd.read_csv(p).set_index("example_id").boardings.reindex(TEST).to_numpy())


def save(x, path):
    out = pd.DataFrame({"example_id": TEST, "boardings": np.expm1(x).clip(0)})
    assert len(out) == 157_194 and np.isfinite(out.boardings).all() and (out.boardings >= 0).all()
    out.to_csv(path, index=False)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/submission_final_K.csv"
    ours = 0.72 * L(M / "lgbm_v9.csv") + 0.18 * L(M / "catboost_v9.csv") + 0.10 * L(M / "mlp_v9.csv")
    v9 = (0.30 * ours + 0.25 * L(F / "input_team_blend.csv") + 0.15 * L(F / "input_team_blend4.csv")
          + 0.10 * L(F / "input_team_v2.csv") + 0.05 * L(F / "input_team_5fold.csv") + 0.05 * L(F / "input_team_post.csv")
          + 0.05 * L(F / "input_team_mlp.csv") + 0.05 * L(F / "input_team_lstm.csv"))
    D = 0.85 * v9 + 0.15 * L(F / "input_team_v5data.csv")
    tmp = Path("outputs/_tmp_D.csv"), Path("outputs/_tmp_F.csv")
    save(D, tmp[0])
    subprocess.run([sys.executable, "src/stack_anchor.py", str(tmp[0]), str(tmp[1]), "0.7"], check=True)
    f = L(tmp[1])
    b5 = L(F / "input_team_blend5.csv")
    B = b5 + 0.22 * (L(M / "lgbm_v9.csv") - L(M / "lgbm_v3.csv"))
    save(0.45 * B + 0.35 * f + 0.20 * b5, out_path)
    for t in tmp:
        t.unlink(missing_ok=True)
    print("저장:", out_path)


if __name__ == "__main__":
    main()
