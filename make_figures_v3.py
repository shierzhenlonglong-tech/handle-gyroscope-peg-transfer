"""主文图 7（腕部传感器：耦合检查、与手柄的并列效应、双手同时运动）。输出 figures/fig7_wrist.pdf/.png。
需先运行 h1_mixed_models.py（results/h1_primary.csv）与 a2_wrist_vs_handle.py（results/a2_wrist_models.csv）。"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h1_mixed_models as h1
from h1_mixed_models import PRIMARY
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
                     "mathtext.fontset": "stix"})
BLUE, ORANGE, GREEN, INK, GRID, MUTED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#333333", "#d9d9d9", "#8c8c8c", "#7a4fc4"
LAB = {"w_mean": "Mean speed", "w_p95": "P95 speed", "qlow_5": "Low-speed", "pause_frac": "Pause time", "sparc": "SPARC", "peaks_per_s": "Peak rate"}


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4); ax.set_axisbelow(True)


fig = plt.figure(figsize=(7.2, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.55)

# (a) handle vs wrist trial-mean speed, with the coupling rule
h1.MIN_W_MEAN = 0.0
hd = h1.load()
w = pd.read_csv("wrist_features.csv", dtype={"session": str})
j = hd.merge(w[["participant", "trial", "hand", "w_mean"]], on=["participant", "trial", "hand"], suffixes=("", "_wr"))
j["rho"] = j.w_mean / j.w_mean_wr
ax = fig.add_subplot(gs[0])
f_h = j.rho < 0.25
f_w = j.rho > 5
p10 = (j.participant == "P10") & (j.hand == "L") & (j.order >= 57)
rest = ~(f_h | f_w | p10)
ax.scatter(j.w_mean_wr[rest], j.w_mean[rest], s=2, color=MUTED, alpha=0.35, lw=0, rasterized=True)
ax.scatter(j.w_mean_wr[f_h], j.w_mean[f_h], s=9, color=ORANGE, lw=0, label=f"Handle sensor not moving (n = {int(f_h.sum())})")
ax.scatter(j.w_mean_wr[f_w], j.w_mean[f_w], s=9, color=BLUE, lw=0, label=f"Wrist sensor not moving (n = {int(f_w.sum())})")
ax.scatter(j.w_mean_wr[p10], j.w_mean[p10], s=9, color=PURPLE, lw=0, marker="^", label=f"Partial fault, not flagged (n = {int(p10.sum())})")
xx = np.array([0.8, 60])
for r, t in ((0.25, "0.25"), (5, "5")):
    ax.plot(xx, r * xx, color=INK, lw=0.6, ls="--")
ax.plot(xx, 1.27 * xx, color=INK, lw=0.6)
ax.text(46, 4.2, "ratio 0.25", fontsize=6.5, ha="right", va="center")
ax.text(1.0, 3.2, "ratio 5", fontsize=6.5, ha="left", va="center")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.9, 50); ax.set_ylim(0.9, 50)
ax.set_xticks([1, 3, 10, 30]); ax.set_xticklabels(["1", "3", "10", "30"]); ax.set_yticks([1, 3, 10, 30]); ax.set_yticklabels(["1", "3", "10", "30"])
ax.set_xlabel("Wrist mean angular speed (°/s)"); ax.set_ylabel("Handle mean angular speed (°/s)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.1), handletextpad=0.2, borderaxespad=0, labelspacing=0.25)
grid(ax); panel(ax, "a")

# (b) adjusted practice effect: handle vs wrist
res = pd.read_csv("results/h1_primary.csv")
wr = pd.read_csv("results/a2_wrist_models.csv")
ax = fig.add_subplot(gs[1])
yt, yl = [], []
for i, f in enumerate(PRIMARY):
    for k, hand in enumerate("LR"):
        y = -(i * 2.6 + k)
        m = res[(res.model == "M2") & (res.feature == f) & (res.hand == hand)].iloc[0]
        q_ = wr[(wr.feature == f) & (wr.hand == hand)].iloc[0]
        ax.plot([m.ci_lo, m.ci_hi], [y + 0.18, y + 0.18], color=INK, lw=0.8)
        ax.plot(m.beta_std, y + 0.18, "o", ms=3.8, color=INK)
        ax.plot([q_.ci_lo, q_.ci_hi], [y - 0.18, y - 0.18], color=GREEN, lw=0.8)
        ax.plot(q_.b1_M2, y - 0.18, "s", ms=3.6, mfc="white", mec=GREEN, mew=1.0)
        yt.append(y); yl.append(f"{LAB[f]}  {hand}" if k == 0 else hand)
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.5)
ax.set_xlabel("Adjusted practice effect (SD per log trial)"); ax.set_xlim(-0.5, 0.5)
ax.legend(handles=[Line2D([], [], marker="o", color=INK, lw=0.8, ms=3.8, label="Handle"), Line2D([], [], marker="s", color=GREEN, mfc="white", lw=0.8, ms=3.6, label="Wrist")],
          frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=1, borderaxespad=0, labelspacing=0.25)
grid(ax, "x"); panel(ax, "b")

# (c) bimanual shares over practice
b = pd.read_csv("wrist_bimanual.csv")
bad = w[w.w_mean < 5.0][["participant", "trial"]].drop_duplicates().assign(_x=1)
b = b.merge(bad, on=["participant", "trial"], how="left"); b = b[b._x.isna()]
ax = fig.add_subplot(gs[2])
for col, c, lab in (("bi_both", GREEN, "Both hands moving"), ("bi_single", PURPLE, "One hand moving")):
    for pid, g in b.groupby("participant"):
        g = g.sort_values("order")
        ax.plot(g.order, 100 * g[col].rolling(5, min_periods=1, center=True).median(), color=c, lw=0.4, alpha=0.25)
    grp = b.assign(bb=np.round(np.log2(b.order) * 2) / 2)
    grp = grp.groupby("bb").filter(lambda x: x.participant.nunique() >= 10).groupby("bb")[col].median()
    ax.plot(2 ** grp.index, 100 * grp.values, color=c, lw=1.8, label=lab)
ax.set_xscale("log"); ax.set_xlim(1, 160); ax.set_ylim(0, 100)
ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["1", "3", "10", "30", "100"])
ax.set_xlabel("Valid-trial order (log scale)"); ax.set_ylabel("Share of trial time (%)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=1, borderaxespad=0, labelspacing=0.25)
grid(ax); panel(ax, "c")
fig.savefig("figures/fig7_wrist.pdf"); fig.savefig("figures/fig7_wrist.png")
print("ok", int(f_h.sum()), int(f_w.sum()), int(p10.sum()))
