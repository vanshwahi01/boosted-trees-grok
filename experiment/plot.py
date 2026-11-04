"""Summarize finished runs and draw the proposal figure.

    python plot.py --runs runs --out ../project/feasibility.pdf
"""
import argparse
import glob
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DELTA, COPY = 0.02, 0.10  # gap threshold for a warning, copy rate for onset
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def first(mask, marks):
    return int(marks[np.argmax(mask)]) if mask.any() else None


def load(root):
    runs = {}
    for path in sorted(glob.glob(f"{root}/n*_s*.json")):
        res = json.load(open(path))
        z = np.load(path[:-4] + "npz")
        runs.setdefault(res["args"]["n"], []).append((res, {k: z[k] for k in z.files}))
    return runs


def summarize(runs):
    for n, group in sorted(runs.items()):
        rows = []
        for res, z in group:
            m, gap = z["marks"], z["L_held"] - z["L_train"]
            onset = [first(gap[:, b] > DELTA, m) for b in range(gap.shape[1])]
            warn = min(s for s in onset if s is not None)
            band = onset.index(warn)
            copy = first(z["mem"] > COPY, m)
            low = onset[0]
            fid_warn = float(z["fid"][np.searchsorted(m, warn)])
            fid_copy = float(z["fid"][np.searchsorted(m, copy)]) if copy else None
            rows.append(dict(
                warn=warn, band_sigma=float(z["centers"][band]), low=low, copy=copy,
                lead=copy - warn if copy else None, fid_warn=fid_warn, fid_copy=fid_copy,
                final_mem=res["final"]["mem"], final_fid=res["final"]["fid"],
                global_mem=res["global"]["mem"], global_fid=res["global"]["fid"],
                stitched_mem=res["stitched"]["mem"], stitched_fid=res["stitched"]["fid"],
                global_step=res["best_global_step"], spearman=res["outlier_spearman"],
                floor=res["train_vs_test_fid"], onset=onset,
            ))
        print(f"\nn={n} ({len(rows)} seeds)")
        for key in rows[0]:
            if key == "onset":
                print("  onset by band", [r["onset"] for r in rows])
                continue
            vals = [r[key] for r in rows if r[key] is not None]
            if vals:
                print(f"  {key:13s} {np.mean(vals):10.3f} +- {np.std(vals):8.3f}  {np.round(vals, 3).tolist()}")


def figure(runs, out, n=300):
    group = runs[n]
    marks = group[0][1]["marks"]
    centers = group[0][1]["centers"]
    gap = np.mean([z["L_held"] - z["L_train"] for _, z in group], 0)
    mem = np.array([z["mem"] for _, z in group])
    split = group[0][0]["split"]
    sig = np.array([r["sigma"] for r in split])
    high = np.array([[r["late_high"] for r in res["split"]] for res, _ in group])
    low = np.array([[r["late_low"] for r in res["split"]] for res, _ in group])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(5.5, 2.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[2.1, 1])
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[1, 0], sharex=a)
    c = fig.add_subplot(grid[:, 1])

    edges_x = np.sqrt(marks[:-1] * marks[1:])
    edges_x = np.r_[marks[0] ** 2 / edges_x[0], edges_x, marks[-1] ** 2 / edges_x[-1]]
    ratio = centers[1] / centers[0]
    edges_y = np.r_[centers / np.sqrt(ratio), centers[-1] * np.sqrt(ratio)]
    im = a.pcolormesh(
        edges_x, edges_y, np.clip(gap.T, DELTA / 2, None), cmap="Blues",
        norm=LogNorm(vmin=DELTA / 2, vmax=gap.max()), rasterized=True,
    )
    a.contour(marks, centers, gap.T, levels=[DELTA], colors=INK, linewidths=0.9)
    a.set(xscale="log", yscale="log", ylabel=r"noise level $\sigma$")
    a.tick_params(labelbottom=False)
    bar = fig.colorbar(im, ax=a, pad=0.01, fraction=0.05)
    bar.set_label("held-out − train loss", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    a.set_title("(a) Where the gap opens", loc="left", fontsize=8, color=INK)

    b.plot(marks, 100 * mem.mean(0), color=INK, lw=1.4)
    b.fill_between(marks, 100 * mem.min(0), 100 * mem.max(0), color=INK, alpha=0.15, lw=0)
    b.set(xscale="log", xlabel="training step", ylabel="copies (%)", ylim=(-3, 103))
    b.grid(axis="y", color=GRID, lw=0.5)
    b.set_yticks([0, 50, 100])
    for axis in (a, b):
        axis.set_xlim(marks[0], marks[-1])

    for ys, color, label, marker in ((high, BLUE, "late above $\\sigma$", "o"), (low, ORANGE, "late below $\\sigma$", "s")):
        c.plot(sig, 100 * ys.mean(0), color=color, lw=1.4, marker=marker, ms=3.5, label=label)
        c.fill_between(sig, 100 * ys.min(0), 100 * ys.max(0), color=color, alpha=0.18, lw=0)
    c.set(xscale="log", xlabel=r"noise cutoff $\sigma$", ylabel="copies (%)", ylim=(-3, 103))
    c.grid(axis="y", color=GRID, lw=0.5)
    c.set_title("(b) Where copying is decided", loc="left", fontsize=8, color=INK)
    c.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.02, 0.42))

    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="../project/feasibility.pdf")
    args = ap.parse_args()
    runs = load(args.runs)
    summarize(runs)
    figure(runs, args.out)
