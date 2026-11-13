"""Public 점수와 예측 차이만으로 여러 제출의 최적 섞기 비율 계산 (log 공간 가중 평균).
섞은 오차 제곱평균 = w' C w, C_ij = (M_i + M_j - D_ij)/2, M = Public^2, D_ij = log 예측 차이 제곱평균."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
U = "/mnt/user-data/uploads/"
F = {  # 파일, Public
    "nayun_lgbm": (U + "306d5f54-submission_lgbm.csv", 0.16087),
    "dh_v2": (U + "1e66c41e-submission_v2.csv", 0.16215),
    "dh_5fold": (U + "06560d94-submission_5fold_grand_ensemble.csv", 0.16257),
    "dh_lstm": (U + "b7db1fa2-submission_lstm_ensemble.csv", 0.16485),
    "dh_mlp": (U + "465c769e-submission_mlp_ensemble.csv", 0.16578),
    "dh_post": (U + "0e00fd6a-submission_grand_post_opening.csv", 0.16706),
    "v6": ("submission_v6_univ.csv", 0.16791),
}
te = pd.read_csv("data/test.csv")[["example_id"]]
L = {}
for k, (f, _) in F.items():
    s = pd.read_csv(f)
    assert len(s) == 157194 and s.example_id.is_unique and set(s.example_id) == set(te.example_id), k
    assert s.boardings.notna().all() and (s.boardings >= 0).all() and np.isfinite(s.boardings).all(), k
    L[k] = np.log1p(te.merge(s, on="example_id").boardings.values)
L = pd.DataFrame(L); names = list(F); M = np.array([F[k][1] ** 2 for k in names])
D = np.array([[((L[a] - L[b]) ** 2).mean() for b in names] for a in names])
C = (M[:, None] + M[None, :] - D) / 2
print("log 차이 RMS\n", pd.DataFrame(np.sqrt(D), names, names).round(3).to_string())
def score(w): return np.sqrt(max(w @ C @ w, 0))
# 검산: 이미 낸 섞기 (nayun 0.65 + v6 0.35) → 실제 0.15781
w0 = np.zeros(len(names)); w0[0], w0[-1] = 0.65, 0.35; print("검산 nayun0.65+v6 예상", round(score(w0), 5), "(실제 0.15781)")
def opt(idx):
    n = len(idx); Cs = C[np.ix_(idx, idx)]
    r = minimize(lambda w: w @ Cs @ w, np.ones(n) / n, bounds=[(0, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    w = np.zeros(len(names)); w[idx] = r.x; return w
w = opt(list(range(len(names))))
print("\n전체 최적:", {k: round(v, 3) for k, v in zip(names, w) if v > 1e-3}, "예상", round(score(w), 5))
for drop in names:
    idx = [i for i, k in enumerate(names) if k != drop]; wd = opt(idx)
    print(f"  {drop} 빼면 예상 {score(wd):.5f}")
np.save("/tmp/blend_C.npy", C); L.to_csv("/tmp/blend_L.csv", index=False)
