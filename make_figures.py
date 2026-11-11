"""论文图 2–7。输出 ../04_figures/*.pdf 与 300 dpi *.png。"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from h1_mixed_models import load, PRIMARY

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})
BLUE, ORANGE, GREEN, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#333333", "#d9d9d9"
HC = {"L": BLUE, "R": ORANGE}
HN = {"L": "Left handle", "R": "Right handle"}
LAB = {"w_mean": "Mean angular speed", "w_p95": "95th-pct. angular speed", "qlow_5": "Low-speed fraction (<5 °/s)",
       "pause_frac": "Pause-time fraction", "sparc": "SPARC smoothness", "peaks_per_s": "Speed-peak rate"}
OUT = "../04_figures/"


def save(fig, name):
    fig.savefig(OUT + name + ".pdf")
    fig.savefig(OUT + name + ".png")
    plt.close(fig)


df = load()
pids = sorted(df.participant.unique())
idx = pd.read_csv("../02_数据/trial_index.csv")
idx = idx[idx.valid_trial.astype(bool)]


def grid(ycol, ylabel, name, logy=False, hline=None):
    fig, axes = plt.subplots(4, 5, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for ax, pid in zip(axes.ravel(), pids):
        if ycol == "T_log":
            g = idx[idx.participant == pid]
            ax.plot(g.order, g.T_log, ".", ms=2.2, color=INK, alpha=0.6)
            x = np.log(g.order)
            b = np.polyfit(x, np.log(g.T_log), 1)
            ax.plot(g.order, np.exp(np.polyval(b, x)), color=GREEN, lw=1.1)
        else:
            for hand in "LR":
                g = df[(df.participant == pid) & (df.hand == hand)].sort_values("order")
                ax.plot(g.order, g[ycol], ".", ms=2.0, color=HC[hand], alpha=0.45)
                if len(g) > 5:
                    b = np.polyfit(np.log(g.order), g[ycol], 1)
                    ax.plot(g.order, np.polyval(b, np.log(g.order)), color=HC[hand], lw=1.1)
        if hline:
            ax.axhline(hline, color=INK, lw=0.6, ls="--")
        if logy:
            ax.set_yscale("log")
        ax.set_title(pid, pad=2)
        ax.grid(True, color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    ax = axes.ravel()[-1]
    ax.axis("off")
    if ycol == "T_log":
        ax.plot([], [], ".", color=INK, label="Trial")
        ax.plot([], [], color=GREEN, label="Power-law fit")
        ax.plot([], [], color=INK, ls="--", lw=0.6, label="48 s criterion")
    else:
        for hand in "LR":
            ax.plot([], [], color=HC[hand], marker=".", label=HN[hand])
    ax.legend(loc="center", frameon=False)
    fig.supxlabel("Valid-trial order", fontsize=8, y=0.03)
    fig.supylabel(ylabel, fontsize=8, x=0.03)
    fig.tight_layout(rect=(0.03, 0.03, 1, 1), h_pad=0.6, w_pad=0.5)
    save(fig, name)


grid("T_log", "Completion time (s, log scale)", "fig2_learning_curves_time", logy=True, hline=48)
grid("w_mean", "Mean angular speed (°/s)", "fig3_learning_curves_speed")

# Fig 4: forest of standardized fixed effects
res = pd.read_csv("../07_results/h1_primary.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, mod, ttl in zip(axes, ("M1", "M2"), ("(a) M1: practice only", "(b) M2: adjusted for completion time and drops")):
    for i, feat in enumerate(PRIMARY):
        for hand, off in (("L", 0.16), ("R", -0.16)):
            r = res[(res.model == mod) & (res.feature == feat) & (res.hand == hand)].iloc[0]
            y = len(PRIMARY) - 1 - i + off
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=HC[hand], lw=1.2)
            ax.plot(r.beta_std, y, "o" if r.p_fdr < 0.05 else "o", ms=4.5, color=HC[hand],
                    mfc=HC[hand] if r.p_fdr < 0.05 else "white", mew=1.0, label=HN[hand] if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(PRIMARY)))
    ax.set_yticklabels([LAB[f] for f in PRIMARY][::-1])
    ax.set_title(ttl, loc="left")
    ax.set_xlabel("Standardized effect of log(trial order) (SD per log-unit)")
    ax.grid(True, axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.7, 0.7)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig4_forest_fixed_effects")

# Fig 5: per-participant OLS slopes (adjusted)
s = pd.read_csv("../07_results/h1_ols_slopes.csv")
fig, axes = plt.subplots(1, 6, figsize=(7.2, 2.5), sharey=True)
rng = np.random.default_rng(1)
for ax, feat in zip(axes, PRIMARY):
    for k, hand in enumerate("LR"):
        v = s[(s.feature == feat) & (s.hand == hand)].slope_M2.values
        vin = v[v >= -1.9]
        ax.plot(k + rng.uniform(-0.13, 0.13, vin.size), vin, "o", ms=3.2, color=HC[hand], alpha=0.75, mec="white", mew=0.4)
        for vo in v[v < -1.9]:          # off-scale value: drawn at the axis edge and labelled
            ax.plot(k, -1.9, "v", ms=4, color=HC[hand])
            ax.text(k + 0.12, -1.9, f"{vo:.1f}", fontsize=6, va="center")
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color=INK, lw=1.3)
        npos = int((v > 0).sum())
        ax.text(k, 1.02, f"{max(npos, v.size - npos)}/{v.size}", ha="center", va="bottom", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["L", "R"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_title(LAB[feat].replace(" (<5 °/s)", "").replace("angular ", "ang. ").replace("95th-pct.", "P95"), fontsize=7, pad=11)
    ax.grid(True, axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Per-participant adjusted slope\n(SD per log-unit of trial order)")
axes[0].set_ylim(-2.05, 1.25)
fig.tight_layout(w_pad=0.4)
save(fig, "fig5_participant_slopes")

# Fig 6: robustness
r = pd.read_csv("../07_results/h1_robustness.csv")
order = ["main", "without_D1_exclusion", "spline_time", "within_between_time", "both_hands_only", "no_duplicate_ids", "no_drops", "noise_0.165", "noise_0.5",
         "leave_one_out_min", "leave_one_out_max"]
names = ["Main model", "Without sensor-fault rule", "Spline for time", "Within/between time", "Both handles valid", "No repeated trial IDs", "No dropped objects",
         "+0.165 °/s noise", "+0.5 °/s noise", "Leave-one-out (min)", "Leave-one-out (max)"]
fig, axes = plt.subplots(1, 6, figsize=(7.2, 3.1), sharey=True)
for ax, feat in zip(axes, PRIMARY):
    for hand, off in (("L", 0.15), ("R", -0.15)):
        for i, v in enumerate(order):
            q = r[(r.variant == v) & (r.feature == feat) & (r.hand == hand)]
            if len(q):
                q = q.iloc[0]
                y = len(order) - 1 - i + off
                if np.isfinite(q.get("ci_lo", np.nan)):
                    ax.plot([q.ci_lo, q.ci_hi], [y, y], color=HC[hand], lw=0.9)
                ax.plot(q.beta_std, y, "o", ms=3.2, color=HC[hand], label=HN[hand] if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_title(LAB[feat].replace(" (<5 °/s)", "").replace("angular ", "ang. ").replace("95th-pct.", "P95"), fontsize=7)
    ax.grid(True, axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([-0.4, 0, 0.4])
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(names[::-1])
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.03))
fig.supxlabel("Adjusted standardized effect of log(trial order)", fontsize=8, y=0.04)
fig.tight_layout(w_pad=0.3, rect=(0, 0.03, 1, 0.95))
save(fig, "fig6_robustness")

# Fig 7: prognostic comparison
h2 = json.load(open("../07_results/h2_prognostic.json"))
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
mods = [("time", "Time only"), ("imu", "IMU only"), ("time_imu", "Time + IMU")]
for ax, key, yl in zip(axes, ("auc", "brier"), ("LOSO AUC", "LOSO Brier score")):
    for j, (res_, mk, nm) in enumerate(zip(h2, ("o", "s"), ("Outcome A", "Outcome B"))):
        ax.plot(np.arange(3) + (j - 0.5) * 0.16, [res_[f"{key}_{m}"] for m, _ in mods], marker=mk, ms=4.5, color=(INK if j == 0 else GREEN), ls="none", label=nm)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in mods], rotation=25, ha="right")
    ax.set_ylabel(yl)
    ax.set_xlim(-0.4, 2.4)
    ax.grid(True, axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
axes[0].set_ylim(0.5, 1.0)
axes[1].set_ylim(0, 0.1)
axes[0].legend(frameon=False, loc="lower left", title="Outcome", labels=["A: next trial proficient", "B: first proficient trial"], fontsize=6, title_fontsize=6)
fig.tight_layout(w_pad=1.0)
save(fig, "fig7_prognostic")
print("ok")
