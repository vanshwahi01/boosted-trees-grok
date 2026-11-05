"""Does gradient boosting learn a parity hidden among random bits, and when?

    python boost.py timing --out runs/boost
    python boost.py grid --out runs/boost
    python boost.py scale --out runs/boost
    python boost.py knobs --out runs/boost
    python boost.py others --out runs/boost
    python boost.py limit --out runs/boost
    python boost.py mlp --out runs/boost
    python boost.py mlp6 --out runs/boost

Each run appends one JSON line. Decision rules fixed before the first full run:
  jump        first round with test accuracy above 0.9
  grokking    training accuracy reaches 0.99 at least 50 rounds before the jump
  luck        jump rounds at k=4, d=31, n=3000 have a coefficient of variation of at least 0.8
  build-up    that coefficient is at most 0.5; values in between are called inconclusive
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
N_TEST, EVERY = 4000, 5


def parity_data(d, k, n, seed, flip=0.0, shuffle=False):
    rng = np.random.default_rng([d, k, n, seed])
    rel = rng.choice(d, k, replace=False)
    x = rng.integers(0, 2, (n + N_TEST, d)).astype(np.float32)
    y = (x[:, rel].sum(1) % 2).astype(int)
    xa, ya, xb, yb = x[:n], y[:n].copy(), x[n:], y[n:]
    if flip:
        ya = np.where(rng.random(n) < flip, 1 - ya, ya)
    if shuffle:
        ya = rng.permutation(ya)
    return xa, ya, xb, yb, rel


def tree_stats(nodes, rel):
    """Splits, splits on relevant bits, and leaves whose path uses every relevant bit."""
    relevant, splits, hits, complete = set(rel.tolist()), 0, 0, 0
    stack = [(0, frozenset())]
    while stack:
        i, seen = stack.pop()
        if nodes["is_leaf"][i]:
            complete += len(seen) == len(relevant)
            continue
        f = int(nodes["feature_idx"][i])
        splits += 1
        if f in relevant:
            hits += 1
            seen = seen | {f}
        stack += [(int(nodes["left"][i]), seen), (int(nodes["right"][i]), seen)]
    return splits, hits, complete


def first_above(curve, level):
    return next((i + 1 for i, a in enumerate(curve) if a > level), None)


def run_boost(cell):
    xa, ya, xb, yb, rel = parity_data(cell["d"], cell["k"], cell["n"], cell["seed"], cell.get("flip", 0.0), cell.get("shuffle", False))
    model = HistGradientBoostingClassifier(
        max_iter=cell["rounds"], early_stopping=False, random_state=cell["seed"], **cell.get("params", {})
    ).fit(xa, ya)
    test = [float((p == yb).mean()) for p in model.staged_predict(xb)]
    train = [float((p == ya).mean()) for p in model.staged_predict(xa)]
    stats = np.array([tree_stats(t[0].nodes, rel) for t in model._predictors], dtype=float)
    bins = range(0, len(stats), EVERY)
    return dict(
        final=test[-1], final_train=train[-1], jump=first_above(test, 0.9), fit_round=first_above(train, 0.989),
        first_hit=first_above(stats[:, 2], 0.5),
        test=[round(test[min(i + EVERY, len(test)) - 1], 3) for i in bins],
        train=[round(train[min(i + EVERY, len(train)) - 1], 3) for i in bins],
        share=[round(stats[i:i + EVERY, 1].sum() / max(stats[i:i + EVERY, 0].sum(), 1), 3) for i in bins],
        complete=[int(stats[i:i + EVERY, 2].sum()) for i in bins],
    )


def run_other(cell):
    xa, ya, xb, yb, _ = parity_data(cell["d"], cell["k"], cell["n"], cell["seed"])
    seed = cell["seed"]
    if cell["model"] == "tree":
        model = DecisionTreeClassifier(random_state=seed)
    elif cell["model"] == "forest":
        model = RandomForestClassifier(500, n_jobs=1, random_state=seed)
    elif cell["model"] == "adaboost":
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, random_state=seed), n_estimators=500, random_state=seed)
    elif cell["model"] == "lightgbm":
        import lightgbm
        model = lightgbm.LGBMClassifier(n_estimators=1000, num_leaves=31, learning_rate=0.1, verbose=-1, n_jobs=1, random_state=seed)
    else:
        import xgboost
        model = xgboost.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=1, random_state=seed)
    model.fit(xa, ya)
    return dict(final=float((model.predict(xb) == yb).mean()), final_train=float((model.predict(xa) == ya).mean()))


def run_mlp(cell):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(cell["seed"])
    xa, ya, xb, yb, _ = parity_data(cell["d"], cell["k"], cell["n"], cell["seed"])
    xa, xb = torch.tensor(2 * xa - 1), torch.tensor(2 * xb - 1)
    ya, yb = torch.tensor(2.0 * ya - 1), torch.tensor(2.0 * yb - 1)
    net = nn.Sequential(nn.Linear(cell["d"], 512), nn.ReLU(), nn.Linear(512, 1))
    if cell["optim"] == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.01)
        loss_fn = lambda out, y: (1 - out * y).clamp(min=0).mean()
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = lambda out, y: nn.functional.soft_margin_loss(out, y)
    test, train = [], []
    for step in range(1, cell["steps"] + 1):
        idx = torch.randint(0, len(xa), (32,))
        loss = loss_fn(net(xa[idx]).squeeze(1), ya[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                test.append(round(float((net(xb).squeeze(1).sign() == yb).float().mean()), 3))
                train.append(round(float((net(xa).squeeze(1).sign() == ya).float().mean()), 3))
    jump = first_above(test, 0.9)
    return dict(final=test[-1], final_train=train[-1], jump=jump and jump * 200, test=test, train=train)


def cells(exp):
    out = []
    if exp == "timing":
        out = [dict(arm="timing", d=31, k=4, n=3000, seed=s, rounds=2000) for s in range(60)]
    elif exp == "grid":
        out = [dict(arm="grid", d=31, k=k, n=n, seed=s, rounds=1000) for k in (2, 3, 4, 5, 6) for n in (1000, 3000, 10000) for s in range(10)]
    elif exp == "scale":
        out = [dict(arm="scale", d=d, k=k, n=10000, seed=s, rounds=1000) for d in (15, 63, 127) for k in (2, 3, 4) for s in range(5)]
    elif exp == "knobs":
        base = dict(d=31, k=4, n=3000, rounds=1000)
        sweeps = dict(
            max_leaf_nodes=(4, 8, 16, 63), learning_rate=(0.03, 0.3), min_samples_leaf=(5, 100, 400), max_depth=(2, 3, 4, 6),
        )
        out = [dict(base, arm=f"{key}={value}", seed=s, params={key: value}) for key, values in sweeps.items() for value in values for s in range(10)]
        out += [dict(base, arm=f"flip={f}", seed=s, flip=f) for f in (0.02, 0.05, 0.1) for s in range(10)]
        out += [dict(base, arm="shuffled", seed=s, shuffle=True) for s in range(10)]
    elif exp == "others":
        out = [dict(arm=m, d=31, k=4, n=3000, seed=s, model=m) for m in ("tree", "forest", "adaboost", "lightgbm", "xgboost") for s in range(10)]
    elif exp == "limit":
        out = [dict(arm=m, d=d, k=3, n=10000, seed=s, model=m) for d in (63, 127) for m in ("tree", "forest", "adaboost") for s in range(3)]
    elif exp == "mlp6":
        out = [dict(arm="mlp-sgd", d=31, k=6, n=n, seed=s, optim="sgd", steps=40000) for n in (3000, 10000) for s in range(3)]
    elif exp == "mlp":
        out = [dict(arm=f"mlp-{o}", d=31, k=k, n=n, seed=s, optim=o, steps=20000) for o in ("sgd", "adamw") for k in (3, 4, 5) for n in (3000, 10000) for s in range(3)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["timing", "grid", "scale", "knobs", "others", "limit", "mlp", "mlp6"])
    ap.add_argument("--out", default="runs/boost")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    runner = dict(others=run_other, limit=run_other, mlp=run_mlp, mlp6=run_mlp).get(args.exp, run_boost)
    t0 = time.time()
    with open(f"{args.out}/{args.exp}.jsonl", "a") as f:
        for i, cell in enumerate(cells(args.exp)):
            start = time.time()
            res = runner(cell)
            f.write(json.dumps(dict(cell, **res, seconds=round(time.time() - start, 1))) + "\n")
            f.flush()
            print(f"{args.exp} {i + 1} {cell['arm']} k={cell['k']} d={cell['d']} n={cell['n']} s={cell['seed']} final={res['final']:.3f} jump={res.get('jump')} {time.time() - t0:.0f}s", flush=True)
    print(f"{args.exp} DONE", flush=True)
