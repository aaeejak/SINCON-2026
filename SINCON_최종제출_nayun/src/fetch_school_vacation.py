"""NEIS 학사일정으로 인천 초·중·고 방학일을 수집한다.

나이스 교육정보 개방포털 서비스 이용약관(2019-01-01 시행)에 맞춰 사용한다.
- 제5조: Open API는 인증키를 발급받은 회원만 사용 → 인증키가 있어야 실행된다.
- 제7조: 인증키는 발급받은 본인만 사용, 타인 공유·공개 금지
  → 키는 NEIS_KEY 환경변수, 홈 폴더의 .neis_key 파일(프로젝트 밖), 실행 중 입력 순으로 읽는다.
    코드·로그·출력 파일에 남기지 않는다. 팀원은 각자 발급받은 키로 실행한다.
- 제6조: 오남용·네트워크 사용 초과 금지 → 학교당 1회 호출(1,000행 단위), 호출 간 0.2초 간격.
- 제11조: 저작자·출처 표시 조건 → data/external/SOURCES.md에 출처 기록.

출처: 교육부·인천광역시교육청, 나이스 교육정보 개방포털 (https://open.neis.go.kr)

주의: 2026년 학사일정은 학년도 시작 전에 정해지는 정보지만, NEIS에는 등록 시점이 남지 않는다
(LOAD_DTM은 재적재 날짜). 대략적인 방학 기간으로만 사용한다.

결과:
    data/external/school_list.csv            수집한 학교 목록 (학교명, 학교급, 구)
    data/external/school_vacation_raw.csv    학교·날짜별 방학 행
    data/external/school_vacation_daily.csv  날짜별 방학 중인 학교 비율 (학교급별)

사용법 (프로젝트 루트에서, 본인 키로 직접 실행):
    python src/fetch_school_vacation.py
"""
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://open.neis.go.kr/hub"
OFFICE = "E10"  # 인천광역시교육청
KINDS = ["초등학교", "중학교", "고등학교"]
START, END = "20250101", "20261031"
PAGE = 1000
DELAY = 0.2
OUT = Path("data/external")


KEY_FILE = Path.home() / ".neis_key"  # 프로젝트 밖이라 제출 ZIP에 들어가지 않음


def get_key():
    """인증키 읽기 순서: NEIS_KEY 환경변수 → ~/.neis_key 파일 → 화면 입력."""
    key = os.environ.get("NEIS_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        # Windows 일부 터미널에서는 getpass가 붙여넣기를 한 글자만 받으므로 일반 입력을 쓴다
        key = input("NEIS 인증키를 붙여 넣고 Enter: ").strip()
    if not key:
        raise SystemExit("인증키가 없습니다. 약관 제5조에 따라 인증키 없이는 실행하지 않습니다.")
    if len(key) != 32 or not all(c in "0123456789abcdefABCDEF" for c in key):
        raise SystemExit(f"인증키 형식이 맞지 않습니다 ({len(key)}글자 입력됨, 32글자여야 함). "
                         "붙여넣기가 끝까지 됐는지 확인하세요.")
    return key


def call(key, service, **params):
    """한 페이지 조회. 오류 메시지에 인증키가 섞이지 않도록 URL을 출력하지 않는다."""
    q = {"KEY": key, "Type": "json", "pSize": PAGE, "ATPT_OFCDC_SC_CODE": OFFICE, **params}
    url = f"{BASE}/{service}?" + urllib.parse.urlencode(q)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            break
        except Exception as e:  # 네트워크 오류: 간격을 늘려 재시도
            last = type(e).__name__
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"{service} 요청 실패 ({last})")
    time.sleep(DELAY)
    if service not in d:
        code = d.get("RESULT", {}).get("CODE", "")
        if code == "INFO-200":  # 데이터 없음
            return 0, []
        raise RuntimeError(f"{service} 응답 오류: {d.get('RESULT')}")
    head = d[service][0]["head"]
    return head[0]["list_total_count"], d[service][1]["row"]


def fetch_all(key, service, **params):
    rows, page = [], 1
    while True:
        total, r = call(key, service, pIndex=page, **params)
        rows += r
        if len(rows) >= total or not r:
            return rows
        page += 1


def main():
    key = get_key()
    OUT.mkdir(parents=True, exist_ok=True)

    schools = pd.DataFrame(fetch_all(key, "schoolInfo"))
    schools = schools[schools.SCHUL_KND_SC_NM.isin(KINDS)]
    schools = pd.DataFrame({
        "school_code": schools.SD_SCHUL_CODE,
        "school_name": schools.SCHUL_NM,
        "kind": schools.SCHUL_KND_SC_NM,
        "address": schools.ORG_RDNMA.str.split().str[:2].str.join(" "),
    }).reset_index(drop=True)
    schools.to_csv(OUT / "school_list.csv", index=False, encoding="utf-8-sig")
    print(f"인천 초·중·고 {len(schools)}곳:", schools.kind.value_counts().to_dict(), flush=True)

    recs, has_data = [], set()
    for i, sc in schools.iterrows():
        rows = fetch_all(key, "SchoolSchedule", SD_SCHUL_CODE=sc.school_code,
                         AA_FROM_YMD=START, AA_TO_YMD=END)
        if rows:
            has_data.add(sc.school_code)
        for r in rows:
            name = r["EVENT_NM"]
            if "방학" in name and "방학식" not in name:
                recs.append({"school_code": sc.school_code, "kind": sc.kind,
                             "date": f"{r['AA_YMD'][:4]}-{r['AA_YMD'][4:6]}-{r['AA_YMD'][6:]}",
                             "event": name})
        if (i + 1) % 50 == 0 or i + 1 == len(schools):
            print(f"  {i + 1}/{len(schools)}", flush=True)

    raw = pd.DataFrame(recs).drop_duplicates(["school_code", "date"]).sort_values(["date", "school_code"])
    raw.to_csv(OUT / "school_vacation_raw.csv", index=False, encoding="utf-8-sig")

    # 날짜별 방학 비율 = 방학 행이 있는 학교 수 / 학사일정이 있는 학교 수 (학교급별)
    base = schools[schools.school_code.isin(has_data)].kind.value_counts()
    cnt = raw.groupby(["date", "kind"]).school_code.nunique().unstack(fill_value=0)
    days = pd.date_range(START, END).strftime("%Y-%m-%d")
    daily = pd.DataFrame({"date": days})
    names = {"초등학교": "elem", "중학교": "middle", "고등학교": "high"}
    for kind, short in names.items():
        c = cnt[kind] if kind in cnt else pd.Series(dtype=float)
        daily[f"vac_{short}"] = (daily.date.map(c).fillna(0) / base.get(kind, 1)).round(3)
    total = sum(base.get(k, 0) for k in names)
    daily["vac_all"] = (sum(daily[f"vac_{names[k]}"] * base.get(k, 0) for k in names) / total).round(3)
    daily.to_csv(OUT / "school_vacation_daily.csv", index=False, encoding="utf-8-sig")
    print(f"저장: school_list.csv, school_vacation_raw.csv ({len(raw)}행), "
          f"school_vacation_daily.csv ({len(daily)}행)")


if __name__ == "__main__":
    main()
