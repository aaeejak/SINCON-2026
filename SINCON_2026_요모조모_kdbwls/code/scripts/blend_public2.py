"""팀원 후보 F~K 점수 포함, Public 점수로 최적 섞기 (log 공간)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
U = "/mnt/user-data/uploads/"
F = {
 "K": (U+"8c6ac3b4-cand_K_B45_F35_blend5_20.csv", .151321), "J": (U+"a4158900-cand_J_blend_of_8.csv", .151856),
 "F_v10_stack": (U+"bcd65ad3-cand_F_v10_stack.csv", .152304), "I_v9_stack": (U+"f762b028-cand_I_v9_stack.csv", .152881),
 "G_v9": (U+"0325da7c-cand_G_v9.csv", .152912), "H_v9_anchor": (U+"3a4e3435-cand_H_v9_anchor.csv", .154823),
 "E_v10_anchor": (None, .154445),
 "blend5": ("submission_blend5.csv", .15176), "sub_final": (U+"37297348-submission_final.csv", .15307),
 "new_v5data": (U+"e71c39ad-submission_v5data_kfold5x3_lgbmB0.1_cat0.1_growth.csv", .16107), "nayun_v3": (U+"306d5f54-submission_lgbm.csv", .16087),
 "grand_tabm": (U+"cb44a93f-submission_grand_tabm_blend.csv", .161296), "dh_v2": (U+"1e66c41e-submission_v2.csv", .16215),
 "dh_5fold": (U+"06560d94-submission_5fold_grand_ensemble.csv", .16257), "dh_lstm": (U+"b7db1fa2-submission_lstm_ensemble.csv", .16485),
 "dh_mlp": (U+"465c769e-submission_mlp_ensemble.csv", .16578), "dh_post": (U+"0e00fd6a-submission_grand_post_opening.csv", .16706),
 "v6": ("submission_v6_univ.csv", .16791), "v7": ("submission_v7_monthly.csv", .16885), "v8": ("submission_v8_tweedie.csv", .172116),
}
F = {k: v for k, v in F.items() if v[0]}
te = pd.read_csv("data/test.csv")[["example_id"]]
L = pd.DataFrame({k: np.log1p(te.merge(pd.read_csv(f), on="example_id").boardings.values) for k, (f, _) in F.items()})
names = list(F); M = np.array([F[k][1] ** 2 for k in names])
D = np.array([[((L[a] - L[b]) ** 2).mean() for b in names] for a in names]); C = (M[:, None] + M[None, :] - D) / 2
sc = lambda w: float(np.sqrt(max(w @ C @ w, 0)))
def opt(idx):
    Cs = C[np.ix_(idx, idx)]; n = len(idx)
    r = minimize(lambda w: w @ Cs @ w, np.ones(n) / n, bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    w = np.zeros(len(names)); w[idx] = r.x; return w
print("K와 다른 파일 차이 rms:", {k: round(float(np.sqrt(D[0, i])), 3) for i, k in enumerate(names)})
w = opt(list(range(len(names)))); print("\n전체 최적", {k: round(v, 3) for k, v in zip(names, w) if v > 1e-3}, "예상", round(sc(w), 5))
np.save("/tmp/C2.npy", C); L.to_parquet("/tmp/L2.parquet"); pd.Series(names).to_csv("/tmp/names2.csv", index=False)
