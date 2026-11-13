"""검바위역(3208) 인근 학교 시험 기간 보정 (12시 승차 +450명 안팎).

학습 기간(2025-01~2026-06)에서 검바위 12시 승차가 평일 4~5일 연속으로 평소(약 130명)의 3~5배(+450명)가 되는 기간이 5번 있다:
2025-04-28~05-01, 2025-07-03~09, 2025-10-16~21, 2025-12-17~23, 2026-04-27~30. 하루 합계는 변하지 않아 학교 시험일의 일찍 하교로 보인다.
2025-04-28(월)~05-01(목)과 2026-04-27(월)~04-30(목)이 364일(같은 요일)로 정확히 일치해, 예측 구간도 364일 이동한 날짜로 추정한다:
  2026-07-02·03·06·07·08, 2026-10-15·16·19·20 (학교 일정으로 확인한 날짜가 아님).
보정: 그 날 12시 예측에 +450명의 약 0.6배를 더한다 (log1p 공간: log1p(pred + 450*shrink) - log1p(pred)... 즉 pred_new = pred + 450*shrink).
사용법: python src/exam_adjust.py 입력.csv 출력.csv [shrink=0.6]
"""
import sys

import numpy as np
import pandas as pd

DAYS = ["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08",
        "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"]
EXTRA = 450.0
STATION, HOUR = 3208, 12


def adjust(sub, test, shrink=0.6):
    k = test[["example_id", "date", "station_id", "hour"]].merge(sub, on="example_id", how="left")
    k["date"] = pd.to_datetime(k["date"])
    m = ((k.station_id == STATION) & (k.hour == HOUR) & k.date.isin(pd.to_datetime(DAYS))).to_numpy()
    b = k.boardings.to_numpy(dtype=float).copy()
    b[m] = b[m] + EXTRA * shrink
    k["boardings"] = b
    return sub[["example_id"]].merge(k[["example_id", "boardings"]], on="example_id", how="left"), int(m.sum())


def main():
    te = pd.read_csv("data/raw/test.csv")
    sub = pd.read_csv(sys.argv[1])
    assert len(sub) == len(te) and (sub.example_id.to_numpy() == te.example_id.to_numpy()).all()
    out, n = adjust(sub, te, float(sys.argv[3]) if len(sys.argv) > 3 else 0.6)
    assert out.boardings.notna().all() and (out.boardings >= 0).all()
    out.to_csv(sys.argv[2], index=False)
    print(f"보정 행 {n}, 저장 {sys.argv[2]}")


if __name__ == "__main__":
    main()
