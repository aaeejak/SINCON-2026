"""팀원(Dohyeong) 후처리를 제출 파일에 적용한다: 축제 보정 → 명절 보정(역별 차이 0.5배).

- src/postprocess/event_adjust.py : 모델이 모르는 2026 7~10월 소규모 축제 5종의 시간대별 로그 승차 증가를 더함
- src/postprocess/holiday_shape.py: 2026 추석 9/24~27과 8/15(토)를, 대회 train.csv의 과거 명절(2025 추석·설, 2026 설)의
  '일요일 대비 역×시간 로그 차'와 예측의 같은 값의 차이로 보정 (예측 자체의 일요일을 기준으로 삼으므로 입력 파일에 맞춰짐)
팀원 코드를 경로(data/raw/)만 바꿔 그대로 가져옴. 팀원 보고: 섞기만 0.151361 → 축제 보정 약 0.1503 → 명절 보정 0.148837 → 역별 차이 0.5배 0.148399.
사용법: python src/apply_postprocess.py 입력.csv 출력.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocess"))
import event_adjust as E  # noqa: E402
import holiday_shape as H  # noqa: E402


def apply(sub, test):
    sub, n = E.adjust(sub, test)
    return H.adjust(sub, test, station_weight=0.5), n


def main():
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    out, n = apply(sub, te)
    assert out.boardings.notna().all() and len(out) == len(sub) and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"축제 보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
