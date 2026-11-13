# Public 점수가 확인된 파일 (낮을수록 좋음) — 파일은 `outputs/scored/`에 점수_이름으로 보관

| Public | 파일 | 만든 사람 | 내용 |
|---|---|---|---|
| **0.14399** | `ours_best+examall` | 우리 | 0.14403 + 다른 역 시험일 10~14시 보정 (9일, k=0.466) (**현재 최고**) |
| **0.14399** | `ours_best+examall_jul` | 우리 | 위와 같되 7월 시험일만 (공동 최고) |
| 0.14399 | `ours_best+examall+Bjul` | 우리 | examall + 두 번째 학교 무리 7/1 (변화 없음) |
| 0.14399 | `ours_best+examjul+Bjul` | 우리 | examall_jul + 두 번째 무리 7/1 (변화 없음) |
| 0.14405 | `ours_best+examall+B` | 우리 | examall + 두 번째 무리 5일 (가을 9/28~10/1 포함 → 악화, 추석 이동으로 날짜가 달라진 듯) |
| 0.14401 | `ours_best+examall_thr` | 우리 | 상승 뚜렷한 153행만, k=0.548 |
| 0.14403 | `ours_gb+fc+Ftuned+v3swap` | 우리 | 부천FC + F 교체 + nayun_v3 자리 교체 |
| 0.14404 | `ours_gb+Ftuned+v3swap` | 우리 | F 교체 + nayun_v3 자리 교체 |
| 0.14406 | `ours_gb+v3swap` | 우리 | nayun_v3 자리(0.085)를 0.80·LGBM v9+0.12·XGB+0.08·Cat로 교체 |
| 0.14407 | `ours_gb+fc+Ftuned` | 우리 | 부천FC + F 교체 |
| 0.14408 | `ours_gb+Ftuned` | 우리 | gb_v6_orig + F 안 우리 몫을 검증 B 최적 혼합으로 교체 |
| 0.14409 | `ours_gb+fc` | 우리 | gb_v6_orig + 부천FC 경기 종료·관중 보정 |
| 0.144110 | `gb_v6_orig` | 팀원 | 우리 검바위 보정본에 팀원 v6 계열을 섞고 문학 10/24~25 등을 조정한 것으로 보임 |
| 0.14416 | `ours_gb+fc+pungmul` | 우리 | 부천FC + 10/2 풍물 축제 |
| 0.14418 | `ours_gb+pungmul` | 우리 | 10/2 풍물 축제 (0.5배) |
| 0.14446 | `ours_gb+hangul` | 우리 | 10/9 한글날 전 역 −0.035 |
| 0.14449 | `ours_gb+fc+pungmul+Ftuned+hangul` | 우리 | 위 넷 조합 (사용자가 파일명 없이 순서로 알려 줌, 한글날 포함 조합으로 추정) |
| 0.144388 | `ours_fixcancel+geombawi` | 우리 | fix_cancel + 검바위 시험일 12·11·13시 + 15·16시 (43행) |
| 0.144518 | `OPT5pp_team50_fix_cancel` | 팀원 | OPT5에서 취소된 행사(OKF 9/26~27 등)와 문학경기장 경기일 보정을 고침 |
| 0.146180 | `ours_OPT5+exam+hours+1516` | 우리 | OPT5 + 검바위 시험일 12·11·13시 + 15·16시 반복 패턴 |
| 0.146200 | `ours_OPT5+exam+1516` | 우리 | OPT5 + 검바위 시험일 12시 + 15·16시 반복 패턴 |
| 0.146293 | `ours_OPT5+exam` | 우리 | OPT5 + 검바위 시험일 12시 보정(9행, +270명) |
| 0.146311 | `OPT5pp_team50` | 팀원 | 팀 블렌드 + 후처리 |
| 0.147532 | `ours_AF_OPT4+SSG` | 우리 | OPT4 + SSG 경기 종료 시각 보정 |
| 0.147755 | `OPT4_final_geo` | 팀원 | OPT + 검단·공연·축제 보정 |
| 0.148301 | `OPT_ev_hol_s05` | 팀원 | 블렌드 + 축제 보정 + 명절 보정(0.5배) |
| 0.151321 | `ours_K` | 우리 | 0.45·B + 0.35·F + 0.20·blend5 |
| 0.151721 | `ours_B_blend5swap` | 우리 | blend5에서 v3를 v9로 교체 |
| 0.151760 | `blend5` | 팀원 | 팀 블렌드 5개 |
| 0.152304 | `ours_F_v10_stack` | 우리 | v10 + 앵커 스태커 보정 |
| 0.152404 | `ours_D_v10` | 우리 | v9 파이널 0.85 + v5data 0.15 |
| 0.152881 | `ours_I_v9_stack` | 우리 | v9 + 앵커 스태커 보정 |
| 0.152912 | `ours_G_v9` | 우리 | 우리 v9 파이널(팀 모델과 혼합) |
| 0.152980 | `blend4` | 팀원 | 팀 블렌드 4개 |
| 0.153075 | `ours_C_v7blend` | 우리 | v7 기반 첫 최종 블렌드 |
| 0.157810 | `blend_team065_v6` | 팀원 | 우리 v3 + 팀 모델 블렌드 |
| 0.160870 | `ours_lgbm_v3` | 우리 | 처음 제출한 LightGBM |

- 팀 단독 모델: v2 0.16215, 5fold 0.16257, MLP 0.16578, LSTM 0.16485, post 0.16706.
- OPT5 위에 우리 SSG 보정을 얹은 후보(`cand_AK`, `cand_AL`)는 원본보다 나쁘다고 보고받아 `outputs/archive/candidates_unscored/`로 옮겼다 (점수 미기록).
- 그 외 점수를 확인하지 않은 시험 후보(앵커, 버스, 축제만 등)도 `archive/candidates_unscored/`에 있다.
