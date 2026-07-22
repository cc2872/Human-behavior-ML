"""
plot_gates.py -- one plotting script for both gates.

Reads gate_a.csv / gate_c.csv (whichever exist), plots the poison-eat rate
(eat0) vs training iteration, averaged over seeds with a +/-1 SEM band, one
line per cell. Also prints the last-quarter mean per cell as the raw numbers.

    python plot_gates.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def curve(rows, key, val):
    """(iters, mean_eat0, sem_eat0) over seeds for rows where row[key]==val."""
    sub = [r for r in rows if r[key] == val]
    seeds = sorted({r["seed"] for r in sub})
    iters = sorted({int(r["iter"]) for r in sub})
    M = np.full((len(seeds), len(iters)), np.nan)
    for si, s in enumerate(seeds):
        for it_i, it in enumerate(iters):
            m = [r for r in sub if r["seed"] == s and int(r["iter"]) == it]
            if m:
                M[si, it_i] = float(m[0]["eat0"])
    mean = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0) / np.sqrt(max(1, M.shape[0]))
    return np.array(iters), mean, sem, len(seeds)


def smooth(y, w=15):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def plot_gate(rows, key, values, labels, title, out_png):
    plt.figure(figsize=(7, 4.5))
    print(f"\n{title}  (last-quarter mean eat0)")
    for v, lab in zip(values, labels):
        it, mean, sem, ns = curve(rows, key, v)
        if not len(it):
            continue
        ms = smooth(mean)
        plt.plot(it, ms, label=lab)
        plt.fill_between(it, ms - sem, ms + sem, alpha=0.2)
        lastq = mean[int(0.75 * len(mean)):]
        print(f"  {lab:16s} n={ns}  eat0 = {np.nanmean(lastq):6.1f}  "
              f"(start {np.nanmean(mean[:max(1,len(mean)//10)]):.1f})")
    plt.xlabel("PPO iteration (episode)")
    plt.ylabel("poison berries eaten / episode (eat0)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"  -> {out_png}")


if __name__ == "__main__":
    if os.path.exists("gate_a.csv"):
        rows = load("gate_a.csv")
        plot_gate(rows, "poison_delay", ["5", "25", "50"],
                  ["delay 5", "delay 25", "delay 50"],
                  "Gate A: single agent, no marks -- avoidance vs delay",
                  "gate_a.png")
    if os.path.exists("gate_c.csv"):
        rows = load("gate_c.csv")
        plot_gate(rows, "condition", ["none", "0", "01"],
                  ["() no rule", "(0,) important", "(0,1) silly"],
                  "Gate C: 6 agents, D=25 -- avoidance speed by condition",
                  "gate_c.png")
