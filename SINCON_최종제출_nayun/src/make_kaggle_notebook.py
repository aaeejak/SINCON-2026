"""src/ 코드로 Kaggle 제출용 노트북(kaggle/sincon_baseline.ipynb)을 만든다.

노트북은 src/calendar_data.py + src/baseline.py를 셀로 담는다 (코드 원본은 src/ 하나로 유지).
Kaggle에서는 대회 데이터를 /kaggle/input 아래에서 자동으로 찾고
/kaggle/working/submission.csv를 저장한다. 인터넷 연결 불필요.

사용법 (프로젝트 루트에서):
    python src/make_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "kaggle" / "sincon_baseline.ipynb"

MAIN_GUARD = 'if __name__ == "__main__":\n    main()\n'


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def main():
    cal = (SRC / "calendar_data.py").read_text(encoding="utf-8")
    base = (SRC / "baseline.py").read_text(encoding="utf-8")
    assert MAIN_GUARD in base, "baseline.py의 main 실행부 형식이 바뀌었습니다"
    base = base.replace(MAIN_GUARD, "")

    cells = [
        md("# SINCON 2026 — 인천 지하철 승차 인원 예측 (베이스라인)\n\n"
           "- 오른쪽 **Add Input**에서 대회 데이터 `sincon-2026`을 추가한 뒤 **Save Version → Save & Run All**.\n"
           "- 실행이 끝나면 `/kaggle/working/submission.csv`(157,194행)가 생성됩니다.\n"
           "- 인터넷 연결이 필요 없습니다.\n"
           "- 이 노트북은 `src/make_kaggle_notebook.py`로 자동 생성됩니다. 직접 고치지 말고 `src/`를 고친 뒤 다시 생성하세요.\n"),
        md("## 1. 공휴일 목록\n"),
        code(cal),
        md("## 2. 베이스라인 코드\n"),
        code(base),
        md("## 3. 검증 · 예측 · submission.csv 저장\n"),
        code("sub = main([])\nsub.head()\n"),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUT}")
    make_lgbm_notebook()


def make_lgbm_notebook():
    """LightGBM 노트북: src 파일들을 %%writefile로 작업 폴더에 쓴 뒤 train_lgbm.py를 실행한다."""
    files = ["calendar_data.py", "baseline.py", "preprocess.py", "train_lgbm.py"]
    cells = [
        md("# SINCON 2026 — LightGBM\n\n"
           "**입력 데이터 2개를 붙여야 합니다** (오른쪽 **Add Input**)\n"
           "1. 대회 데이터 `sincon-2026`\n"
           "2. 팀 추가 데이터 (`share/sincon_추가데이터.zip`을 비공개 Kaggle Dataset으로 올린 것)\n\n"
           "그다음 **Save Version → Save & Run All**. 끝나면 `/kaggle/working/submission.csv`(157,194행)가 생깁니다.\n"
           "인터넷 연결은 필요 없습니다. 이 노트북은 `src/make_kaggle_notebook.py`로 자동 생성되니 직접 고치지 마세요.\n"),
        code("import lightgbm, pandas, numpy\n"
             "print('lightgbm', lightgbm.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__)\n"),
    ]
    for f in files:
        cells.append(code(f"%%writefile {f}\n" + (SRC / f).read_text(encoding="utf-8")))
    cells.append(md("## 학습 · 검증 · 예측\n"))
    cells.append(code("!python train_lgbm.py\n"))
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = OUT.parent / "sincon_lgbm.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
