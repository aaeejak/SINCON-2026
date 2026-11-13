"""섞은 원본 → 전체 후처리 (Public 검증 순서): 축제 → 명절(0.5) → 공연장·인접역(+OKF) → 검단(0.5) → SSG 종료 시간대.
사용: python scripts/full_pp.py 입력.csv 출력.csv"""
import sys, pandas as pd
sys.path.insert(0, "scripts")
import event_adjust as E, holiday_shape as H, venue_adjust as V, geomdan_adjust as G, ssg_end_adjust as S
def run(sub, te, okf=True):
    sub, _ = E.adjust(sub, te)
    sub = H.adjust(sub, te, station_weight=0.5)
    sub = V.adjust(sub, te, okf=okf)
    g = G.adjust(sub, te); sub = g[0] if isinstance(g, tuple) else g
    sub, _ = S.adjust(sub, te)
    assert len(sub) == 157194 and sub.boardings.notna().all() and (sub.boardings >= 0).all()
    return sub
if __name__ == "__main__":
    te = pd.read_csv("data/test.csv"); out = run(pd.read_csv(sys.argv[1]), te, okf=not (len(sys.argv) > 3 and sys.argv[3] == "no_okf"))
    out.to_csv(sys.argv[2], index=False); print("저장", sys.argv[2])
