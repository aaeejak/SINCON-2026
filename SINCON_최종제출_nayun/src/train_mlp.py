"""임베딩 MLP (SINCON 2026) — LightGBM과 오차 방향이 다른 보조 모델.

범주(역, 시간, 요일 구분, 월, 노선)는 임베딩, 나머지는 표준화한 숫자로 넣고
y = te_st_dt_h(역·요일·시간 평균) + MLP 출력 형태로 잔차를 학습한다.

사용법 (프로젝트 루트에서):
    python src/train_mlp.py valid   # 두 검증 구간, 에폭별 점수와 LightGBM과의 혼합 이득
    python src/train_mlp.py final [에폭]   # 최종 학습 → outputs/submission_mlp.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as P  # noqa: E402
import train_lgbm as T  # noqa: E402

torch.set_num_threads(8)
CATS = {"station_id": 16, "hour": 4, "daytype": 4, "month": 4, "line": 2}
EPOCHS = 14
DROP = 0.1
WD = 1e-5
BATCH = 4096


class Net(nn.Module):
    def __init__(self, sizes, n_num):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(sizes[c], d) for c, d in CATS.items()])
        n_in = sum(CATS.values()) + n_num
        self.mlp = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xc, xn):
        z = torch.cat([e(xc[:, i]) for i, e in enumerate(self.embs)] + [xn], 1)
        return self.mlp(z).squeeze(1)


def base_of(df):
    """기준값: 역·요일·시간 평균 (비어 있으면 역·시간 → 역 평균 → 전체 평균 순)."""
    b = df.te_st_dt_h.fillna(df.te_st_h).fillna(df.te_st)
    return torch.tensor(b.fillna(b.mean()).to_numpy(), dtype=torch.float32)


def encode(df, feats, fit=None):
    cat_cols = list(CATS)
    num_cols = [c for c in feats if c not in cat_cols]
    if fit is None:
        maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in cat_cols}
        X = df[num_cols].astype(float).fillna(0.0)
        fit = dict(maps=maps, num_cols=num_cols, mu=X.mean(), sd=X.std().replace(0, 1.0))
    xc = np.stack([df[c].map(fit["maps"][c]).fillna(0).astype(int).to_numpy() for c in cat_cols], 1)
    xn = ((df[fit["num_cols"]].astype(float).fillna(0.0) - fit["mu"]) / fit["sd"]).clip(-6, 6).to_numpy()
    return torch.tensor(xc, dtype=torch.long), torch.tensor(xn, dtype=torch.float32), fit


def train_mlp(tr, feats, va=None, epochs=EPOCHS, seed=42, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xc, xn, fit = encode(tr, feats)
    sizes = {c: len(fit["maps"][c]) + 1 for c in CATS}
    net = Net(sizes, xn.shape[1])
    base = base_of(tr)
    y = torch.tensor(tr.y.to_numpy(), dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=WD)
    steps = epochs * ((len(y) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    if va is not None:
        vc, vn, _ = encode(va, feats, fit)
        vbase = base_of(va)
    hist = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), BATCH):
            idx = perm[i:i + BATCH]
            pred = base[idx] + net(xc[idx], xn[idx])
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if va is not None:
            p = predict(net, vc, vn, vbase)
            hist.append(T.rmsle_log(va.y.to_numpy(), p))
            if log:
                log(ep, hist[-1])
    return net, fit, hist


@torch.no_grad()
def predict(net, xc, xn, base):
    net.eval()
    out = []
    for i in range(0, len(xc), 65536):
        out.append((base[i:i + 65536] + net(xc[i:i + 65536], xn[i:i + 65536])).numpy())
    return np.concatenate(out).clip(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "valid"
    data = P.load_raw()
    if mode == "valid":
        import lightgbm as lgb
        params = {**T.BASE, **T.GRID[0]}
        for fi, (cut, a, b) in enumerate(T.FOLDS):
            t0 = time.time()
            tr, va, feats = T.fold_data(data, cut, a, b)
            net, fit, hist = train_mlp(tr, feats, va, log=lambda e, s: print(f"  fold{fi} 에폭 {e + 1} {s:.4f}", flush=True))
            vc, vn, _ = encode(va, feats, fit)
            p = predict(net, vc, vn, base_of(va))
            Xtr = T.to_lgb(tr, feats)
            m = lgb.train(params, lgb.Dataset(Xtr, tr.y.to_numpy()), [961, 1049][fi])
            pb = m.predict(T.to_lgb(va, feats, Xtr)).clip(0)
            y = va.y.to_numpy()
            print(f"fold{fi} MLP {T.rmsle_log(y, p):.4f} LightGBM {T.rmsle_log(y, pb):.4f} 거리 {np.sqrt(np.mean((p - pb) ** 2)):.4f} ({time.time() - t0:.0f}초)")
            for w in (0.1, 0.2, 0.3, 0.4, 0.5):
                print(f"   MLP 비중 {w}: {T.rmsle_log(y, (1 - w) * pb + w * p):.4f}", flush=True)
    else:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
        tr, te, feats = P.make_dataset(data=data)
        preds = []
        for sd in (42, 7, 2026):
            net, fit, _ = train_mlp(tr, feats, epochs=epochs, seed=sd)
            xc, xn, _ = encode(te, feats, fit)
            preds.append(predict(net, xc, xn, base_of(te)))
            print(f"시드 {sd} 완료", flush=True)
        p = np.mean(preds, axis=0).clip(0)
        sub = pd.DataFrame({"example_id": te.example_id, "boardings": np.expm1(p).clip(0)})
        order = pd.read_csv(P.RAW / "test.csv", usecols=["example_id"]).example_id
        assert len(sub) == 157_194 and (sub.example_id.to_numpy() == order.to_numpy()).all()
        assert np.isfinite(sub.boardings).all()
        sub.to_csv("outputs/submission_mlp.csv", index=False)
        print("저장: outputs/submission_mlp.csv")


if __name__ == "__main__":
    main()
