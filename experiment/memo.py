"""Feasibility experiment: when does each noise level of a diffusion model start to memorize?

Trains a small EDM-style U-Net on a subset of MNIST, tracks the train/held-out denoising gap per
noise band, and compares three ways of choosing weights at sampling time: the final checkpoint, the
single best checkpoint by held-out loss, and a per-band "stitched" model.

    python memo.py --data DIR --n 300 --seed 0 --steps 30000 --out runs/n300_s0
"""
import argparse
import copy
import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
SIGMA_MIN, SIGMA_MAX, SIGMA_DATA = 0.01, 40.0, 0.5
N_BANDS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_mnist(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(MIRROR + name, path)
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if key.endswith("x"):
            arr = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 1, 28, 28)
            out[key] = torch.from_numpy(arr.copy()).float() / 127.5 - 1.0
        else:
            out[key] = torch.from_numpy(np.frombuffer(raw, np.uint8, offset=8).copy()).long()
    return out


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, c_emb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c_in)
        self.c1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(c_emb, c_out)
        self.n2 = nn.GroupNorm(8, c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=32, c_emb=128):
        super().__init__()
        self.c_emb = c_emb
        self.mlp = nn.Sequential(nn.Linear(c_emb, c_emb), nn.SiLU(), nn.Linear(c_emb, c_emb))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.d1 = ResBlock(ch, ch, c_emb)
        self.d2 = ResBlock(ch, ch * 2, c_emb)
        self.d3 = ResBlock(ch * 2, ch * 4, c_emb)
        self.m1 = ResBlock(ch * 4, ch * 4, c_emb)
        self.m2 = ResBlock(ch * 4, ch * 4, c_emb)
        self.u3 = ResBlock(ch * 8, ch * 2, c_emb)
        self.u2 = ResBlock(ch * 4, ch, c_emb)
        self.u1 = ResBlock(ch * 2, ch, c_emb)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def embed(self, c_noise):
        half = self.c_emb // 2
        freqs = torch.exp(-np.log(1e4) * torch.arange(half, device=c_noise.device) / half)
        a = c_noise[:, None] * 100 * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))

    def forward(self, x, c_noise):
        e = self.embed(c_noise)
        h1 = self.d1(self.inp(x), e)
        h2 = self.d2(F.avg_pool2d(h1, 2), e)
        h3 = self.d3(F.avg_pool2d(h2, 2), e)
        h = self.m2(self.m1(h3, e), e)
        h = self.u3(torch.cat([h, h3], 1), e)
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2), h2], 1), e)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2), h1], 1), e)
        return self.out(h)


def denoise(net, x, sigma):
    """EDM preconditioning (Karras et al. 2022)."""
    s = sigma.view(-1, 1, 1, 1)
    d = s**2 + SIGMA_DATA**2
    return SIGMA_DATA**2 / d * x + s * SIGMA_DATA / d.sqrt() * net(x / d.sqrt(), sigma.log() / 4)


def per_sample_loss(net, x, sigma, noise):
    w = (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA) ** 2
    pred = denoise(net, x + sigma.view(-1, 1, 1, 1) * noise, sigma)
    return w * ((pred - x) ** 2).mean((1, 2, 3))


@torch.no_grad()
def band_losses(net, X, centers, repeats, batch=500):
    """Mean weighted loss in each noise band, with noise fixed across evaluations."""
    out = []
    for b, s in enumerate(centers):
        g = torch.Generator().manual_seed(1000 + b)
        vals = []
        for _ in range(repeats):
            noise = torch.randn(X.shape, generator=g)
            for i in range(0, len(X), batch):
                xb, nb = X[i:i + batch].to(DEVICE), noise[i:i + batch].to(DEVICE)
                vals.append(per_sample_loss(net, xb, torch.full((len(xb),), float(s), device=DEVICE), nb).cpu())
        out.append(torch.cat(vals).mean().item())
    return np.array(out)


def karras_sigmas(steps=18, rho=7):
    r = np.linspace(0, 1, steps)
    return (SIGMA_MAX ** (1 / rho) + r * (SIGMA_MIN ** (1 / rho) - SIGMA_MAX ** (1 / rho))) ** rho


@torch.no_grad()
def sample(nets, band_of, count, seed, batch=500):
    """Deterministic Heun sampler. `nets[band_of(sigma)]` denoises at noise level sigma."""
    sig = karras_sigmas()
    g = torch.Generator().manual_seed(seed)
    out = []
    for start in range(0, count, batch):
        k = min(batch, count - start)
        x = (torch.randn(k, 1, 28, 28, generator=g) * sig[0]).to(DEVICE)
        for i in range(len(sig) - 1):
            s0, s1 = float(sig[i]), float(sig[i + 1])
            d0 = (x - denoise(nets[band_of(s0)], x, torch.full((k,), s0, device=DEVICE))) / s0
            x1 = x + (s1 - s0) * d0
            d1 = (x1 - denoise(nets[band_of(s1)], x1, torch.full((k,), s1, device=DEVICE))) / s1
            x = x + (s1 - s0) * (d0 + d1) / 2
        out.append(x.cpu())
    return torch.cat(out).clamp(-1, 1)


def memorized(samples, train):
    """Yoon et al. (2023): copy if distance to the nearest neighbour < 1/3 of distance to the second."""
    d = torch.cdist(samples.flatten(1), train.flatten(1))
    top = d.topk(2, largest=False)
    return top.values[:, 0] < top.values[:, 1] / 3, top.indices[:, 0]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.conv(x))


def get_classifier(data, cache):
    clf = Classifier()
    cache = Path(cache)
    if cache.exists():
        clf.load_state_dict(torch.load(cache, map_location="cpu"))
        return clf.to(DEVICE).eval()
    clf.to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(3):
        perm = torch.randperm(len(data["train_x"]))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(clf(data["train_x"][idx].to(DEVICE)), data["train_y"][idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(data["test_x"].to(DEVICE)).argmax(1).cpu() == data["test_y"]).float().mean().item()
    print(f"classifier test accuracy {acc:.4f}", flush=True)
    torch.save({k: v.cpu() for k, v in clf.state_dict().items()}, cache)
    return clf


@torch.no_grad()
def features(clf, X, batch=1000):
    return np.concatenate([clf.conv(X[i:i + batch].to(DEVICE)).cpu().numpy() for i in range(0, len(X), batch)])


def feature_fid(f1, f2):
    m1, m2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1.T), np.cov(f2.T)
    w, v = np.linalg.eigh(c1)
    root = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    tr = np.sqrt(np.clip(np.linalg.eigvalsh(root @ c2 @ root), 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * tr)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--checkpoints", type=int, default=20)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--track-samples", type=int, default=500)
    ap.add_argument("--final-samples", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_mnist(args.data)
    clf = get_classifier(data, Path(args.data) / "classifier.pt")
    perm = rng.permutation(len(data["train_x"]))
    train = data["train_x"][perm[: args.n]]
    held = data["train_x"][perm[-1000:]]  # selects checkpoints; never overlaps the training subset
    test = data["test_x"]                 # only used for the final quality score
    test_feats = features(clf, test)

    edges = np.geomspace(SIGMA_MIN, SIGMA_MAX, N_BANDS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    band_of = lambda s: int(np.clip(np.searchsorted(edges, s) - 1, 0, N_BANDS - 1))
    repeats = max(1, int(np.ceil(1000 / args.n)))

    net = UNet(args.ch).to(DEVICE)
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    print(f"params {sum(p.numel() for p in net.parameters()):,} device {DEVICE}", flush=True)

    marks = np.unique(np.geomspace(min(250, args.steps), args.steps, args.checkpoints).astype(int))
    ckpts, L_train, L_held, mem, fid = [], [], [], [], []
    t0 = time.time()
    train_dev = train.to(DEVICE)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, args.n, (args.batch,), device=DEVICE)
        x = train_dev[idx]
        sigma = torch.exp(
            torch.rand(args.batch, device=DEVICE) * (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) + np.log(SIGMA_MIN)
        )
        loss = per_sample_loss(net, x, sigma, torch.randn_like(x)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - 0.999)
        if step in marks:
            ckpts.append({k: v.cpu().clone() for k, v in ema.state_dict().items()})
            L_train.append(band_losses(ema, train, centers, repeats))
            L_held.append(band_losses(ema, held, centers, 1))
            s = sample([ema] * N_BANDS, band_of, args.track_samples, seed=7)
            mem.append(memorized(s, train)[0].float().mean().item())
            fid.append(feature_fid(features(clf, s), test_feats))
            print(
                f"step {step} t={time.time() - t0:.0f}s mem={mem[-1]:.3f} fid={fid[-1]:.1f} "
                f"gap={np.round(L_held[-1] - L_train[-1], 3).tolist()}",
                flush=True,
            )

    L_train, L_held = np.array(L_train), np.array(L_held)

    def build(state):
        m = UNet(args.ch).to(DEVICE).eval()
        m.load_state_dict(state)
        return m

    best_global = int(L_held.mean(1).argmin())
    best_band = L_held.argmin(0)
    cache = {i: build(ckpts[i]) for i in {len(ckpts) - 1, best_global, *best_band.tolist()}}
    variants = {
        "final": [cache[len(ckpts) - 1]] * N_BANDS,
        "global": [cache[best_global]] * N_BANDS,
        "stitched": [cache[int(i)] for i in best_band],
    }
    results = {}
    samples = {}
    for name, nets in variants.items():
        s = sample(nets, band_of, args.final_samples, seed=11)
        flag, nn_idx = memorized(s, train)
        results[name] = {"mem": flag.float().mean().item(), "fid": feature_fid(features(clf, s), test_feats)}
        samples[name] = s[:64].numpy()
        if name == "final":
            counts = np.bincount(nn_idx[flag].numpy(), minlength=args.n)
            d = torch.cdist(train.flatten(1), train.flatten(1))
            d.fill_diagonal_(float("inf"))
            outlier = d.topk(min(5, args.n - 1), largest=False).values.mean(1).numpy()
            results["outlier_spearman"] = spearman(counts, outlier)
            results["train_points_copied"] = float((counts > 0).mean())

    # Where is copying decided? Use the final weights only above, or only below, a noise threshold.
    late, early = cache[len(ckpts) - 1], cache[best_global]
    results["split"] = []
    for cut in range(1, N_BANDS):
        row = {"sigma": float(edges[cut])}
        for name, nets in {
            "late_high": [late if b >= cut else early for b in range(N_BANDS)],
            "late_low": [early if b >= cut else late for b in range(N_BANDS)],
        }.items():
            s = sample(nets, band_of, args.track_samples, seed=11)
            row[name] = memorized(s, train)[0].float().mean().item()
            row[name + "_fid"] = feature_fid(features(clf, s), test_feats)
        results["split"].append(row)

    results["baseline_mem_test_images"] = memorized(test[:2000], train)[0].float().mean().item()
    results["train_vs_test_fid"] = feature_fid(features(clf, train), test_feats) if args.n > 64 else None
    results["best_global_step"] = int(marks[best_global])
    results["best_band_step"] = marks[best_band].tolist()
    results["args"] = vars(args)
    results["seconds"] = time.time() - t0
    print(json.dumps(results, indent=1), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(results, indent=1))
    np.savez(
        out.with_suffix(".npz"), marks=marks, centers=centers, L_train=L_train, L_held=L_held,
        mem=np.array(mem), fid=np.array(fid), train=train[:64].numpy(), **{f"samples_{k}": v for k, v in samples.items()},
    )


if __name__ == "__main__":
    main()
