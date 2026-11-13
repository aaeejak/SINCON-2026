"""검단 지역 행정동별 월별 주민등록 인구를 받는다 (인증키 불필요).

출처: 행정안전부 주민등록 인구통계 - 행정동별 주민등록 인구 및 세대현황
      https://jumin.mois.go.kr/statMonth.do (조회 화면의 표를 읽음, 파일 다운로드 아님)

- 2026-07-01 인천 행정체제 개편으로 서구(2826000000)에서 검단구(2829000000)가 분리됨.
  2025-01 ~ 2026-06은 서구, 2026-07 ~ 2026-10은 검단구로 조회한다.
- 2025년 중 아라동 → 아라1·2동 분동, 원당·당하동 경계 조정이 있어 동별 시계열이 끊긴다.
  모델에는 8개 동 합계(gumdan_total)를 쓰는 것을 권장한다.

결과: data/external/gumdan_population_monthly.csv
    month, dong, population, households, population_adj
    dong='검단합계'는 8개 동 합. population_adj는 경계 변경 효과를 뺀 합계(2026-10 경계 기준)로,
    모델 피처로는 이 값을 쓴다.

사용법 (프로젝트 루트에서):
    python src/fetch_population.py
"""
import io
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://jumin.mois.go.kr/statMonth.do"
INCHEON = "2800000000"
QUERIES = [  # (구 코드, 시작 연월, 끝 연월)
    ("2826000000", "2025-01", "2025-12"),
    ("2826000000", "2026-01", "2026-06"),
    ("2829000000", "2026-07", "2026-10"),
]
GUMDAN = ["검단동", "불로대곡동", "원당동", "당하동", "오류왕길동", "마전동", "아라동", "아라1동", "아라2동"]


def query(gu, start, end):
    ys, ms = start.split("-")
    ye, me = end.split("-")
    data = urllib.parse.urlencode({
        "sltOrgType": "2", "sltOrgLvl1": INCHEON, "sltOrgLvl2": gu, "sltUndefType": "",
        "nowYear": "2026", "searchYearMonth": "month",
        "searchYearStart": ys, "searchMonthStart": ms, "searchYearEnd": ye, "searchMonthEnd": me,
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=60) as r:
        html = r.read().decode("utf-8")
    time.sleep(1)
    tables = pd.read_html(io.StringIO(html), match="행정기관코드", thousands=",")
    t = tables[0]
    months = sorted({re.sub(r"\D", "", c[0])[:6] for c in t.columns if "년" in str(c[0])})
    rows = []
    for _, r in t.iterrows():
        name = str(r.iloc[0]).split()[-1].lstrip("*")  # 경계 변경 표시(*)·공백 제거
        for i, m in enumerate(months):
            rows.append({"month": f"{m[:4]}-{m[4:]}", "dong": name,
                         "population": r.iloc[1 + 2 * i], "households": r.iloc[2 + 2 * i]})
    return pd.DataFrame(rows)


def main():
    df = pd.concat([query(*q) for q in QUERIES], ignore_index=True)
    df = df[df.dong.isin(GUMDAN)]
    df[["population", "households"]] = df[["population", "households"]].apply(pd.to_numeric)
    total = df.groupby("month")[["population", "households"]].sum().reset_index().assign(dong="검단합계")

    # 경계 변경 보정: 경계가 바뀐 달은 해당 동을 빼고 증가율을 계산해 연결(chain)한다.
    # 2025-07 당하동 경계 조정(2.8만 → 4.1만 명)으로 합계가 실제 입주와 무관하게 뛴다.
    pv = df.pivot_table(index="month", columns="dong", values="population")
    months = list(pv.index)
    growth = [1.0]
    for prev, cur in zip(months, months[1:]):
        both = pv.loc[[prev, cur]].dropna(axis=1)
        jump = (both.loc[cur] / both.loc[prev] - 1).abs() > 0.15
        keep = both.columns[~jump]
        if len(both.columns) < len(pv.loc[prev].dropna()):  # 분동 등으로 동 구성이 바뀐 달 → 합계 비율
            growth.append(total.set_index("month").population[cur] / total.set_index("month").population[prev])
        else:
            growth.append(both.loc[cur, keep].sum() / both.loc[prev, keep].sum())
    idx = pd.Series(growth, index=months).cumprod()
    last = total.set_index("month").population[months[-1]]
    total["population_adj"] = (idx / idx.iloc[-1] * last).round().to_numpy()  # 현재(마지막 달) 경계 기준
    out = pd.concat([df, total], ignore_index=True).sort_values(["dong", "month"])
    path = Path("data/external/gumdan_population_monthly.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} ({len(out)}행)")
    print(total.set_index("month")[["population", "population_adj"]].to_string())


if __name__ == "__main__":
    main()
