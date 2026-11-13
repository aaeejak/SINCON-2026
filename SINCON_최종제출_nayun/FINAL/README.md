# 최종 제출 (2026-11-13, 마감 21:00)

| Kaggle 최종 선택 | 파일 | Public | 내용 |
|---|---|---|---|
| **1** | `final_submission.csv` | **0.14399** | 2번 + 7월 시험일 10~14시 보정 (다른 역들의 일찍 하교, 1,720행) |
| **2** | `final_submission_2.csv` | 0.14403 | 팀원 `gb_v6_orig` + 부천FC 경기일 보정 + F 교체 + nayun_v3 자리 교체 |

- Kaggle 제출 기록에서 찾을 파일 (같은 내용): 1번 = `cand_best_examall_jul.csv`, 2번 = `cand_gb_fc_Ftuned_v3swap.csv`.
- 형식: 157,194행, `example_id, boardings`, test 순서, ID 중복·결측·음수·무한대 없음 (`src/run_final.py`가 검사).
- 이전 최종 파일(OPT5 + 검바위, 0.146293)은 `outputs/scored/0.146293_ours_OPT5+exam.csv`에 그대로 있다.

## 재현
```bash
python src/run_final.py          # 약 1분, outputs/repro/에 다시 만들고 FINAL/과 비교 → "재현 성공"
```
단계별 명령, 계수의 근거, 우리 모델 재학습 순서는 프로젝트 루트 `README.md`의 "최종 제출 재현" 절에 있다.

## 마감 전 체크리스트
1. **Kaggle**: Submissions 탭에서 위 두 제출의 "Use for Final Score"에 체크 (선택하지 않으면 Public 최고가 자동 선택되는데, 0.14399 동점 파일이 여럿이라 다른 파일이 고를 수 있다).
2. **ZIP**: `share/SINCON_최종제출_nayun.zip`을 조장에게 전달 → 조장이 I-FOSS에 게시 → I-KAGGLE에서 선택.
3. **팀원 몫 (규칙 7절)**: 두 파일 모두 팀원 `gb_v6_orig`가 기반이다. 그 파이프라인 코드·비율·시드·requirements와 팀원 외부 데이터 출처를 ZIP에 함께 넣어야 완전히 재현된다.
4. **확인**: "제출 무제한" 공지가 공식 채널에 있는지, 모든 제출이 같은 Kaggle 팀으로 나갔는지 (원 규칙은 팀당 하루 10회).
5. 규칙 점검: `docs/최종2개_규칙점검_2026-11-13.md`.
