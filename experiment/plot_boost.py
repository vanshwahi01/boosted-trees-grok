"""Summarize the boosting runs and draw the proposal figure.

    python plot_boost.py --runs runs/boost --out ../project/feasibility.pdf
"""
import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SOLVED, EVERY = 0.9, 5


def load(root, name):
    path = Path(root) / f"{name}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def solved(group):
    return np.mean([r["jump"] is not None for r in group])


def median_jump(group):
    """Median over all runs, counting unsolved runs as later than any solved run; None if it falls among them."""
    mid = np.median([r["jump"] if r["jump"] else np.inf for r in group])
    return None if np.isinf(mid) else float(mid)


def aligned_share(runs, before=40, after=8):
    rows = []
    for r in runs:
        if r["jump"] is None:
            continue
        at = (r["jump"] - 1) // EVERY
        if at < before:
            continue
        rows.append(r["share"][at - before:at + after])
    return np.arange(-before, after) * EVERY, np.array(rows)


def summarize(root):
    timing = load(root, "timing")
    if timing:
        jumps = np.array([r["jump"] for r in timing if r["jump"]], dtype=float)
        rng = np.random.default_rng(0)
        boot = [np.std(s, ddof=1) / np.mean(s) for s in (rng.choice(jumps, len(jumps)) for _ in range(5000))]
        grok = [r for r in timing if r["jump"] and r["fit_round"] and r["jump"] - r["fit_round"] >= 50]
        print(f"timing runs {len(timing)} solved {len(jumps)} final>.9 {np.mean([r['final'] > SOLVED for r in timing]):.2f}")
        print(f"jump median {np.median(jumps):.0f} iqr {np.percentile(jumps, 25):.0f}-{np.percentile(jumps, 75):.0f} min {jumps.min():.0f} max {jumps.max():.0f}")
        print(f"cv {np.std(jumps, ddof=1) / jumps.mean():.2f} ci {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f}")
        print(f"fit round median {np.median([r['fit_round'] for r in timing if r['fit_round']]):.0f}; grokking runs {len(grok)}/{len(timing)}")
        print(f"first hit median {np.median([r['first_hit'] for r in timing if r['first_hit']]):.0f}")
        print("lag jump-first hit median", np.median([r["jump"] - r["first_hit"] for r in timing if r["jump"] and r["first_hit"]]))
        x, rows = aligned_share(timing)
        if len(rows):
            mean = rows.mean(0)
            print("aligned share runs", len(rows), "at -200,-100,-50,-25,-10,0:", [round(float(mean[list(x).index(t)]), 3) for t in (-200, -100, -50, -25, -10, 0)])
    for name in ("grid", "scale"):
        cells = defaultdict(list)
        for r in load(root, name):
            cells[(r["k"], r["d"], r["n"])].append(r)
        print(name, "k d n | solved | median jump | median fit | final")
        for key, g in sorted(cells.items()):
            print(key, f"{solved(g):.2f}", median_jump(g), np.median([r["fit_round"] or np.nan for r in g]), f"{np.mean([r['final'] for r in g]):.3f}")
    for name in ("knobs", "others", "libs", "limit", "mlp", "mlp6", "mlp_flip"):
        cells = defaultdict(list)
        for r in load(root, name):
            cells[(r["arm"], r["k"], r["n"])].append(r)
        print(name)
        for key, g in cells.items():
            line = f"{key} final {np.mean([r['final'] for r in g]):.3f} +- {np.std([r['final'] for r in g]):.3f} train {np.mean([r['final_train'] for r in g]):.3f} n={len(g)}"
            if "jump" in g[0]:
                line += f" solved {solved(g):.2f} median {median_jump(g)}"
            print(line)


def score_reweight(root):
    """Apply the rules fixed in boost.py to the reweighted forest."""
    core, hard = load(root, "reweight_core"), load(root, "reweight_hard")
    if not core:
        return
    rows = core + hard
    pick = lambda arm, d, k, n: [r for r in rows if r["arm"] == arm and (r["d"], r["k"], r["n"]) == (d, k, n)]
    base = pick("reweight", 31, 4, 3000)
    shuffled = pick("reweight-shuffled", 31, 4, 3000)
    n_base = sum(r["jump"] is not None for r in base)
    top_shuffled = max(r["best"] for r in shuffled) if shuffled else None
    passed = n_base >= 8 and top_shuffled is not None and top_shuffled <= 0.55
    print(f"baseline: solved {n_base}/{len(base)}, shuffled best {top_shuffled}, pass {passed}")
    print("iteration of solve, 4 inputs:", [r["jump"] for r in base], "share after first forest", round(np.mean([r["share"][0] for r in base]), 3), "chance", round(4 / 31, 3))
    long = load(root, "boost_long")
    flips = load(root, "boost_flip")
    wins = []
    for d, k, n in ((31, 5, 10000), (63, 4, 10000)):
        a, b = pick("reweight", d, k, n), [r for r in long if (r["d"], r["k"], r["n"]) == (d, k, n)]
        sa, sb = sum(r["jump"] is not None for r in a), sum(r["jump"] is not None for r in b)
        wins.append(sa - sb >= 3 and len(a) == len(b) and len(a) > 0)
        print(f"d={d} k={k}: reweight {sa}/{len(a)}, boost-4000 {sb}/{len(b)}, seconds {np.mean([r['seconds'] for r in a]) if a else 0:.0f} vs {np.mean([r['seconds'] for r in b]) if b else 0:.0f}")
    for f in (0.02, 0.1):
        a, b = pick(f"reweight-flip={f}", 31, 4, 3000), [r for r in flips if r["arm"] == f"boost-4000-flip={f}"]
        ma, mb = np.mean([r["final"] for r in a]) if a else 0, np.mean([r["final"] for r in b]) if b else 0
        wins.append(bool(a) and bool(b) and ma - mb >= 0.10)
        print(f"flip {f}: reweight {100 * ma:.1f}+-{100 * np.std([r['final'] for r in a]) if a else 0:.1f}, boost-4000 {100 * mb:.1f}+-{100 * np.std([r['final'] for r in b]) if b else 0:.1f}")
    print(f"flip 0.05: reweight {100 * np.mean([r['final'] for r in pick('reweight-flip=0.05', 31, 4, 3000)] or [0]):.1f}")
    print(f"main comparison: {passed and any(wins)}  (wins {wins})")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["d"], r["k"], r["n"])].append(r)
    for key, g in cells.items():
        print(key, f"solved {sum(r['jump'] is not None for r in g)}/{len(g)} final {100 * np.mean([r['final'] for r in g]):.1f} top {np.mean([r['top'][-1] for r in g]):.2f} sec {np.mean([r['seconds'] for r in g]):.0f}")
    groups = defaultdict(list)
    for r in long + flips:
        groups[(r["arm"], r["d"], r["k"], r["n"])].append(r)
    for key, g in groups.items():
        print(key, f"solved {sum(r['jump'] is not None for r in g)}/{len(g)} final {100 * np.mean([r['final'] for r in g]):.1f} sec {np.mean([r['seconds'] for r in g]):.0f} median jump {median_jump(g)}")


def figure(root, out):
    timing = load(root, "timing")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.4), constrained_layout=True)
    a, b, c, d = axes

    picks = sorted((r for r in timing if r["jump"]), key=lambda r: r["jump"])
    picks = [picks[int(q * (len(picks) - 1))] for q in (0.15, 0.5, 0.85)]
    for i, r in enumerate(picks):
        rounds = EVERY * np.arange(1, len(r["test"]) + 1)
        a.plot(rounds, 100 * np.array(r["train"]), color=ORANGE, lw=0.9, ls="--", label="train" if i == 0 else None)
        a.plot(rounds, 100 * np.array(r["test"]), color=BLUE, lw=1.2, label="test" if i == 0 else None)
    a.set(xscale="log", xlim=(5, 2000), ylim=(8, 106), xlabel="round", ylabel="accuracy (%)")
    a.set_xticks([10, 100, 1000], ["10", "100", "1k"])
    a.minorticks_off()
    a.set_title("(a) Three runs", loc="left", fontsize=7, color=INK)
    a.legend(frameon=False, fontsize=6, loc="lower left", handlelength=1.4, labelspacing=0.1, borderaxespad=0.1)

    late = [r for r in timing if r["jump"] and r["jump"] >= 450]
    x = EVERY * np.arange(-80, 9)
    rows = np.array([[r["complete"][(r["jump"] - 1) // EVERY + i] / EVERY for i in range(-80, 9)] for r in late])
    smooth = np.convolve(rows.mean(0), np.ones(5) / 5, mode="same")
    shuffled = [r for r in load(root, "knobs") if r["arm"] == "shuffled"]
    level = np.mean([np.mean(r["complete"]) / EVERY for r in shuffled])
    b.plot(x[2:-2], smooth[2:-2], color=BLUE, lw=1.3)
    b.axhline(level, color=MUTED, lw=0.7, ls=":")
    b.text(x[0], level * 1.25, "shuffled labels", fontsize=6, color=MUTED)
    b.set(yscale="log", xlabel="rounds from transition", ylabel="leaves per tree", xticks=[-400, -200, 0])
    b.set_title("(b) Complete leaves", loc="left", fontsize=7, color=INK)

    cells = defaultdict(list)
    for r in load(root, "grid") + load(root, "scale"):
        if r["n"] == 10000:
            cells[(r["k"], r["d"])].append(r)
    longer = defaultdict(list)
    for r in load(root, "boost_long"):
        longer[(r["k"], r["d"])].append(r)
    cells.update(longer)
    for k, color, marker, style in ((2, MUTED, "s", ":"), (3, BLUE, "o", "-"), (4, ORANGE, "^", "--")):
        pts = sorted((comb(d, k), median_jump(g)) for (kk, d), g in cells.items() if kk == k and median_jump(g))
        c.plot(*zip(*pts), color=color, ls=style, marker=marker, ms=2.5, lw=1, label=f"k={k}")
    c.set(xscale="log", yscale="log", xlabel="$k$-subsets of $d$ bits", ylabel="median round")
    c.set_xticks([1e2, 1e4, 1e6])
    c.minorticks_off()
    c.set_title("(c) Scaling", loc="left", fontsize=7, color=INK)
    c.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.8, labelspacing=0.1, borderaxespad=0.0)

    forests = sorted((r for r in load(root, "reweight_core") if r["arm"] == "reweight" and (r["k"], r["n"]) == (4, 3000)), key=lambda r: r["jump"])
    for r in (forests[int(q * (len(forests) - 1))] for q in (0.15, 0.5, 0.85)):
        iters = np.arange(1, len(r["test"]) + 1)
        d.plot(iters, 100 * np.array(r["share"]), color=AQUA, lw=0.9, ls="--")
        d.plot(iters, 100 * np.array(r["test"]), color=BLUE, lw=1.2)
    d.set(xlabel="forest", ylabel="percent", ylim=(0, 106), xticks=[1, 5, 10])
    d.set_title("(d) Reweighted", loc="left", fontsize=7, color=INK)
    d.text(1, 66, "test", fontsize=6, color=MUTED)
    d.text(10, 3, "hidden-bit share", fontsize=6, color=MUTED, ha="right")
    for axis in axes:
        axis.grid(axis="y", color=GRID, lw=0.5)
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/boost")
    ap.add_argument("--out", default="../project/feasibility.pdf")
    args = ap.parse_args()
    summarize(args.runs)
    score_reweight(args.runs)
    if load(args.runs, "timing"):
        figure(args.runs, args.out)
