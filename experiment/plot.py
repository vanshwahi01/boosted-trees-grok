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

DELTA, COPY = 0.02, 0.10  # gap threshold for a warning, copy rate for onset
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


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


def figure(root, out):
    arms = [
        ("no rule", "base", BLUE, "-"),
        ("live rule", "rule_online", ORANGE, "--"),
        ("fixed rule", "rule_fixed", AQUA, "-."),
    ]
    data = {key: np.load(f"{root}/{key}_n300_s0.npz") for _, key, _, _ in arms}
    floor = json.load(open(f"{root}/base_n300_s0.json"))["train_vs_test_fid"]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.5, 1.55), constrained_layout=True)

    for label, key, color, style in arms:
        z = data[key]
        a.plot(z["marks"], 100 * z["mem"], color=color, ls=style, lw=1.3, label=label)
        b.plot(z["marks"], z["fid"], color=color, ls=style, lw=1.3)
        if key != "rule_fixed":
            c.plot(z["marks"], z["L_train"][:, 0], color=color, ls=style, lw=1.3)
        if key != "rule_online":
            stop = np.argmin(z["L_held"].mean(1))
            b.plot(z["marks"][stop], z["fid"][stop], "o", ms=4, mfc="white", mec=color, mew=1.1, zorder=3)

    a.set(ylabel="copies (%)", ylim=(-3, 103))
    b.set(ylabel="FID", yscale="log", ylim=(7, 400))
    b.axhline(floor, color=MUTED, lw=0.7, ls=":")
    c.set(ylabel="train loss", yscale="log", ylim=(0.05, 5))
    c.axvline(13250, color=MUTED, lw=0.7, ls=":")
    c.text(12300, 3.6, "band\nfrozen", fontsize=6.5, color=MUTED, ha="right", va="top")
    c.text(1100, 0.06, r"$\sigma\approx 0.02$", fontsize=6.5, color=MUTED)

    ticks = [1000, 3000, 10000, 30000]
    for axis, title in ((a, "(a) Copying"), (b, "(b) Image quality"), (c, "(c) Forgetting")):
        axis.set(xscale="log", xlim=(1000, 30000), xlabel="training step")
        axis.set_xticks(ticks, ["1k", "3k", "10k", "30k"])
        axis.minorticks_off()
        axis.grid(axis="y", color=GRID, lw=0.5)
        axis.set_title(title, loc="left", fontsize=8, color=INK)
    a.legend(loc="upper left", frameon=False, fontsize=6.5, labelspacing=0.2, borderaxespad=0.1, handlelength=2.2)

    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="../project/feasibility.pdf")
    args = ap.parse_args()
    summarize(load(args.runs))
    figure(args.runs, args.out)
