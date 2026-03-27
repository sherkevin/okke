#!/usr/bin/env python3
"""
A-OSP Appendix C — Layer Sensitivity Plot
===========================================
Plots the "inverted-U" curve showing intervention efficiency
(ΔPPL / |ΔF1|) peaking at Layer 24.

Higher efficiency = intervention disrupts language-inertia fluency (↑PPL)
with minimal accuracy cost (↓|ΔF1|).

Input: layer_sensitivity.csv from layer-scan experiments
Output: layer_sensitivity_appxC.{pdf,png}
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_PATH = Path("/root/autodl-tmp/A-OSP_Project/logs/features/layer_sensitivity.csv")
OUT_DIR = Path("/root/autodl-tmp/A-OSP_Project/logs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def render_layer_sensitivity():
    rows = load_data()

    base = [r for r in rows if r["config"] == "base"][0]
    base_f1 = float(base["f1"])
    base_ppl = float(base["ppl"])

    layers = []
    delta_f1 = []
    delta_ppl = []
    f1_vals = []
    ppl_vals = []
    efficiency = []
    intv_rate = []

    for r in rows:
        if r["config"] == "base":
            continue
        idx = int(r["layer_idx"])
        f1 = float(r["f1"])
        ppl = float(r["ppl"])
        df1 = abs(f1 - base_f1)
        dppl = ppl - base_ppl

        layers.append(idx)
        delta_f1.append(df1)
        delta_ppl.append(dppl)
        f1_vals.append(f1)
        ppl_vals.append(ppl)
        intv_rate.append(float(r["intv_per_sample"]))

        eff = dppl / max(df1, 1e-6)
        efficiency.append(eff)

    layers = np.array(layers)
    efficiency = np.array(efficiency)
    f1_vals = np.array(f1_vals)
    ppl_vals = np.array(ppl_vals)
    delta_f1 = np.array(delta_f1)
    intv_rate = np.array(intv_rate)

    # ══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # ── Top panel: Efficiency (inverted-U) ──
    ax1 = axes[0]
    colors = ["#E85D4A" if l == 24 else "#4A90D9" for l in layers]
    bars = ax1.bar(layers, efficiency, width=2.8, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=5)

    peak_idx = np.argmax(efficiency)
    ax1.annotate(
        f"Layer {layers[peak_idx]}\nEfficiency = {efficiency[peak_idx]:.1f}",
        xy=(layers[peak_idx], efficiency[peak_idx]),
        xytext=(layers[peak_idx] + 2, efficiency[peak_idx] + 1),
        fontsize=10, fontweight="bold", color="#E85D4A",
        arrowprops=dict(arrowstyle="->", color="#E85D4A", lw=1.5),
    )

    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax1.set_ylabel("Efficiency (ΔPPL / |ΔF1|)", fontsize=11, fontweight="bold")
    ax1.set_title("Appendix C: Layer Sensitivity — Intervention Efficiency",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Bottom panel: F1 and PPL dual axis ──
    ax2 = axes[1]

    ln1 = ax2.plot(layers, f1_vals, "o-", color="#4A90D9", linewidth=2,
                   markersize=7, label="F1", zorder=5)
    ax2.axhline(y=base_f1, color="#4A90D9", linewidth=1, linestyle="--",
                alpha=0.5, label=f"Base F1={base_f1:.3f}")
    ax2.set_ylabel("POPE F1", fontsize=11, color="#4A90D9")
    ax2.tick_params(axis="y", labelcolor="#4A90D9")
    ax2.set_ylim(0.5, 0.90)

    ax2r = ax2.twinx()
    ln2 = ax2r.plot(layers, ppl_vals, "s--", color="#E85D4A", linewidth=2,
                    markersize=7, label="PPL", zorder=5)
    ax2r.axhline(y=base_ppl, color="#E85D4A", linewidth=1, linestyle="--",
                 alpha=0.5, label=f"Base PPL={base_ppl:.2f}")
    ax2r.set_ylabel("Perplexity", fontsize=11, color="#E85D4A")
    ax2r.tick_params(axis="y", labelcolor="#E85D4A")

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="upper left", fontsize=9, framealpha=0.85)

    ax2.set_xlabel("Intervention Layer Index", fontsize=11, fontweight="bold")
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.2, axis="x")

    # Highlight layer 24
    for ax in [ax1, ax2]:
        ax.axvline(x=24, color="#E85D4A", linewidth=0.8, linestyle=":",
                   alpha=0.4, zorder=1)

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"layer_sensitivity_appxC.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[AppxC] Saved → {OUT_DIR}/layer_sensitivity_appxC.{{pdf,png}}")

    # Print data summary for verification
    print(f"\nLayer Sensitivity Data:")
    print(f"  {'Layer':>5s} {'F1':>6s} {'PPL':>6s} {'|ΔF1|':>6s} {'ΔPPL':>6s} {'Eff':>8s} {'Intv/S':>6s}")
    for i, l in enumerate(layers):
        print(f"  {l:5d} {f1_vals[i]:6.3f} {ppl_vals[i]:6.2f} "
              f"{delta_f1[i]:6.4f} {delta_ppl[i]:6.2f} {efficiency[i]:8.2f} "
              f"{intv_rate[i]:6.2f}")

    plt.close(fig)


if __name__ == "__main__":
    render_layer_sensitivity()
