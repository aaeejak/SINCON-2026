"""최종 제출 2개를 한 번에 다시 만들고 FINAL/의 파일과 같은지 확인한다 (약 1분).

  최종 1 = FINAL/final_submission.csv   (Public 0.14399)
  최종 2 = FINAL/final_submission_2.csv (Public 0.14403)

순서: 팀원 gb_v6_orig → 부천FC 보정(fc_adjust) → F 교체·v3 교체(misc_adjust) = 최종 2 → 7월 시험일 10~14시 보정(exam_all_adjust) = 최종 1
사용법 (프로젝트 루트에서): python src/run_final.py [출력 폴더=outputs/repro]
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/repro")
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, PYTHONPATH="src", PYTHONIOENCODING="utf-8")


def run(*args):
    print(">", " ".join(args))
    subprocess.run([sys.executable, *args], check=True, env=ENV)


run("src/fc_adjust.py", "team_inputs/input_team_gb_v6_orig.csv", str(OUT / "step1_fc.csv"), "0.7")
run("src/misc_adjust.py", str(OUT / "step1_fc.csv"), str(OUT / "final_submission_2.csv"), "ftuned", "v3swap")
run("src/exam_all_adjust.py", str(OUT / "final_submission_2.csv"), str(OUT / "final_submission.csv"), "0.466", "0", "7")

te = pd.read_csv("data/raw/test.csv", usecols=["example_id"])
ok = True
for name in ("final_submission.csv", "final_submission_2.csv"):
    a, b = pd.read_csv(OUT / name), pd.read_csv(Path("FINAL") / name)
    fmt = (list(a.columns) == ["example_id", "boardings"] and len(a) == 157_194 and a.example_id.is_unique
           and (a.example_id.to_numpy() == te.example_id.to_numpy()).all()
           and np.isfinite(a.boardings).all() and (a.boardings >= 0).all())
    diff = float(np.abs(a.boardings - b.boardings).max())
    print(f"{name}: 형식 {'OK' if fmt else '오류'}, FINAL과 최대 차이 {diff}")
    ok &= fmt and diff < 1e-6
print("재현 성공" if ok else "재현 실패")
sys.exit(0 if ok else 1)
