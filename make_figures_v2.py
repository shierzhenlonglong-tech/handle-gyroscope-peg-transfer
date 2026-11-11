"""主文图 2–6（v2）与补充图。输出 ../04_figures/v2/*.pdf 与 300 dpi *.png。"""
import os, sys, math, zipfile, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import oh2026_qc_sensitivity as q
from h1_mixed_models import load, PRIMARY

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
                     "mathtext.fontset": "stix"})
BLUE, ORANGE, GREEN, INK, GRID, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#333333", "#d9d9d9", "#8c8c8c"
HC = {"L": BLUE, "R": ORANGE}
HN = {"L": "Left handle", "R": "Right handle"}
LAB = {"logT": "Completion time (log)", "w_mean": "Mean angular speed", "w_p95": "P95 angular speed", "qlow_5": "Low-speed fraction",
       "pause_frac": "Pause-time fraction", "sparc": "SPARC", "peaks_per_s": "Speed-peak rate"}
OUT = "../04_figures/v2/"
os.makedirs(OUT, exist_ok=True)
FS = 1 / 0.032


def save(fig, name):
    fig.savefig(OUT + name + ".pdf")
    fig.savefig(OUT + name + ".png")
    plt.close(fig)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


df = load()
idx = pd.read_csv("../02_数据/trial_index.csv")
idx = idx[idx.valid_trial.astype(bool)]
pids = sorted(df.participant.unique())

# ---------------------------------------------------------------- Fig 2: signals + group curves
def raw_trace(src, pid, trial_key, hand="LH"):
    for z in glob.glob(os.path.join(src, "*.zip")):
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            if n.endswith(trial_key + "_IMU.txt") and f"/{pid}_" in "/" + n:
                d, _ = q.parse_imu(zf.read(n).decode("utf-8", "replace"), 180 / math.pi)
                a = d[hand]
                a = a[np.argsort(a[:, 0])]
                return a[:, 0] - a[0, 0], np.linalg.norm(a[:, 1:4], axis=1)
    raise FileNotFoundError(trial_key)


SRC = sys.argv[1] if len(sys.argv) > 1 else None
EX = "P03"
g = df[(df.participant == EX) & (df.hand == "L")].sort_values("order")
early, late = g.iloc[2], g.iloc[-3]
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55, wspace=0.28)
gsa = gs[0, :].subgridspec(1, 2, width_ratios=[early.T_log, late.T_log + 0.25 * early.T_log], wspace=0.08)
axs = [fig.add_subplot(gsa[0, 0])]
axs.append(fig.add_subplot(gsa[0, 1], sharey=axs[0]))
for ax, r, ttl in zip(axs, (early, late), ("early", "late")):
    if SRC:
        t, s = raw_trace(SRC, EX, r.trial)
        ax.fill_between(t, 0, 5, color=GRID, lw=0, zorder=0)
        ax.plot(t, s, color=BLUE, lw=0.5)
    ax.set_xlim(0, r.T_log)
    ax.set_title(f"{ttl.capitalize()} (trial {int(r.order)}): {r.T_log:.0f} s, mean {r.w_mean:.1f} °/s, {100 * r.qlow_5:.0f}% < 5 °/s",
                 fontsize=7, loc="left", pad=3)
    ax.set_xlabel("Time in trial (s)")
axs[0].set_ylabel("Handle angular\nspeed (°/s)")
axs[0].set_ylim(0, 150)
plt.setp(axs[1].get_yticklabels(), visible=False)
panel(axs[0], "a")

axb, axc = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
med = {}
for pid in pids:
    gg = idx[idx.participant == pid].sort_values("order")
    axb.plot(gg.order, gg.T_log.rolling(5, min_periods=1, center=True).median(), color=MUTED, lw=0.5, alpha=0.7)
grp = idx.assign(b=np.round(np.log2(idx.order) * 2) / 2)
grp = grp.groupby("b").filter(lambda x: x.participant.nunique() >= 10).groupby("b").T_log.median()
axb.plot(2 ** grp.index, grp.values, color=INK, lw=1.8, label="Median across trials")
axb.axhline(48, color=INK, lw=0.6, ls="--")
axb.text(1.05, 44, "48 s proficiency time", fontsize=6.5, va="top")
axb.set_xscale("log"); axb.set_yscale("log")
axb.set_yticks([30, 50, 100, 200, 400]); axb.set_yticklabels(["30", "50", "100", "200", "400"])
axb.set_xlabel("Valid-trial order (log scale)"); axb.set_ylabel("Completion time (s)")
axb.set_xlim(1, 160); grid(axb); panel(axb, "b")
for hand in "LR":
    h = df[df.hand == hand]
    for pid in pids:
        gg = h[h.participant == pid].sort_values("order")
        axc.plot(gg.order, gg.w_mean.rolling(5, min_periods=1, center=True).median(), color=HC[hand], lw=0.4, alpha=0.3)
    grp = h.assign(b=np.round(np.log2(h.order) * 2) / 2)
    grp = grp.groupby("b").filter(lambda x: x.participant.nunique() >= 10).groupby("b").w_mean.median()
    axc.plot(2 ** grp.index, grp.values, color=HC[hand], lw=1.8, label=HN[hand])
axc.set_xscale("log"); axc.set_xlim(1, 160)
axc.set_xlabel("Valid-trial order (log scale)"); axc.set_ylabel("Mean angular speed (°/s)")
axc.legend(frameon=False, loc="upper left"); grid(axc); panel(axc, "c")
for a in (axb, axc):
    a.set_xticks([1, 3, 10, 30, 100]); a.set_xticklabels(["1", "3", "10", "30", "100"])
save(fig, "fig2_signals_and_curves")

# ---------------------------------------------------------------- Fig 3: dumbbell M1 -> M2 with SRM
res = pd.read_csv("../07_results/h1_primary.csv")
srm = pd.read_csv("../07_results/h1_srm.csv")
ols = pd.read_csv("../07_results/h1_ols_slopes.csv")
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ys, labels = [], []
for i, feat in enumerate(PRIMARY):
    for j, hand in enumerate("LR"):
        y = -(i * 2.6 + j)
        m1 = res[(res.model == "M1") & (res.feature == feat) & (res.hand == hand)].iloc[0]
        m2 = res[(res.model == "M2") & (res.feature == feat) & (res.hand == hand)].iloc[0]
        ax.plot([m1.beta_std, m2.beta_std], [y, y], color=HC[hand], lw=0.8, alpha=0.5, zorder=1)
        ax.plot([m2.ci_lo, m2.ci_hi], [y, y], color=HC[hand], lw=2.2, alpha=0.35, solid_capstyle="butt", zorder=2)
        ax.plot(m1.beta_std, y, "o", ms=5, mfc="white", mec=HC[hand], mew=1.1, zorder=3)
        ax.plot(m2.beta_std, y, "o", ms=5, color=HC[hand], zorder=4)
        v = ols[(ols.feature == feat) & (ols.hand == hand)].slope_M2.values
        n_same = int(np.sum(np.sign(v) == np.sign(m2.beta_std)))
        s = srm[(srm.feature == feat) & (srm.hand == hand)].SRM.iloc[0]
        ax.text(0.80, y, f"{s:+.2f}", ha="right", va="center", fontsize=7)
        ax.text(0.95, y, f"{n_same}/19", ha="right", va="center", fontsize=7, fontweight="bold" if n_same >= 15 else "normal")
        ax.text(-0.72, y, hand, ha="left", va="center", fontsize=7, color=HC[hand])
    ys.append(-(i * 2.6 + 0.5)); labels.append(LAB[feat])
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(ys); ax.set_yticklabels(labels)
ax.set_xlim(-0.75, 0.97); ax.set_xticks(np.arange(-0.6, 0.61, 0.2))
ax.set_xlabel("Standardized effect of practice (SD per log-unit of trial order)", x=0.42)
ax.text(0.80, 1.5, "SRM", ha="right", fontsize=7); ax.text(0.95, 1.5, "Sign", ha="right", fontsize=7)
ax.set_ylim(-(5 * 2.6 + 1.8), 2.4)
grid(ax, "x")
ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, label="M1: practice only"),
                   Line2D([], [], marker="o", ls="none", color=INK, label="M2: adjusted for completion time"),
                   Line2D([], [], color=MUTED, lw=2.2, alpha=0.6, label="95% CI of M2")],
          frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=3, columnspacing=1.2, handletextpad=0.4)
save(fig, "fig3_dumbbell_effects")

# ---------------------------------------------------------------- Fig 4: participant slopes
fig, axes = plt.subplots(1, 6, figsize=(7.2, 2.4), sharey=True)
rng = np.random.default_rng(1)
for ax, feat in zip(axes, PRIMARY):
    for k, hand in enumerate("LR"):
        v = ols[(ols.feature == feat) & (ols.hand == hand)].slope_M2.values
        vin = v[v >= -1.9]
        ax.plot(k + rng.uniform(-0.14, 0.14, vin.size), vin, "o", ms=3.2, color=HC[hand], alpha=0.75, mec="white", mew=0.4)
        for vo in v[v < -1.9]:
            ax.plot(k, -1.9, "v", ms=4, color=HC[hand]); ax.text(k + 0.12, -1.9, f"{vo:.1f}", fontsize=6, va="center")
        ax.plot([k - 0.27, k + 0.27], [np.median(v)] * 2, color=INK, lw=1.3)
        npos = int((v > 0).sum()); nmaj = max(npos, v.size - npos)
        ax.text(k, 1.02, f"{nmaj}/{v.size}", ha="center", va="bottom", fontsize=6.5, transform=ax.get_xaxis_transform(),
                fontweight="bold" if nmaj >= 15 else "normal")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["L", "R"]); ax.set_xlim(-0.6, 1.6)
    ax.set_title(LAB[feat], fontsize=7, pad=11); grid(ax, "y")
axes[0].set_ylabel("Per-participant adjusted slope\n(SD per log-unit of trial order)"); axes[0].set_ylim(-2.05, 1.25)
fig.tight_layout(w_pad=0.4)
save(fig, "fig4_participant_slopes")

# ---------------------------------------------------------------- Fig 5: measurement properties
rel = pd.read_csv("../07_results/x1_reliability.csv")
order = ["logT"] + PRIMARY
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, col, xl, ref, reft in ((axes[0], "ICC1", "Single-trial reliability, ICC(1)", 0.8, "0.8"),
                               (axes[1], "gain_over_MDC", "Practice gain / single-trial MDC$_{95}$", 1.0, "gain = MDC")):
    for i, f in enumerate(order):
        y = len(order) - 1 - i
        if f == "logT":
            r = rel[(rel.feature == f) & (rel.hand == "L")].iloc[0]
            ax.plot(r[col], y, "s", ms=5.5, color=INK, label="Completion time" if col == "ICC1" else None)
        else:
            for hand, off in (("L", 0.15), ("R", -0.15)):
                r = rel[(rel.feature == f) & (rel.hand == hand)].iloc[0]
                ax.plot(r[col], y + off, "o", ms=5, color=HC[hand], label=HN[hand] if (i == 1 and col == "ICC1") else None)
    ax.axvline(ref, color=INK, lw=0.6, ls="--")
    ax.text(ref - 0.02 * (ax.get_xlim()[1] if col != "ICC1" else 1), len(order) - 0.55, reft, fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel(xl); grid(ax, "x"); ax.set_ylim(-0.6, len(order) - 0.2)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([LAB[f] for f in order][::-1])
axes[0].set_xlim(0, 1); axes[1].set_xlim(0, 2.4)
axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.2)
panel(axes[0], "a"); panel(axes[1], "b")
fig.tight_layout(w_pad=1.5)
save(fig, "fig5_measurement_properties")

# ---------------------------------------------------------------- Fig 6: LASK external cohort
lk = pd.read_csv("../07_results/h3_lask_trials.csv")
ls = pd.read_csv("../07_results/h3_lask_summary.csv").set_index("metric")
CO = {"DatasetA_7DOF_train": (BLUE, "o", "Cohort A (n = 19)"), "DatasetB_BAPES_val": (ORANGE, "s", "Cohort B (n = 10)"),
      "DatasetC_6DOF_test": (GREEN, "^", "Cohort C (n = 8)")}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for ax, col, yl in ((axes[0], "duration_s", "Trial duration (s)"), (axes[1], "w_p95", "P95 angular speed (°/s)")):
    for dsn, (c, mk, nm) in CO.items():
        g = lk[lk.dataset == dsn]
        ax.plot(1 + g.procedures_lifetime, g[col], mk, ms=4.5, color=c, mec="white", mew=0.5, label=nm)
    ax.set_xscale("log"); ax.set_xticks([1, 11, 101, 1001]); ax.set_xticklabels(["0", "10", "100", "1000"])
    ax.set_xlabel("Lifetime procedures (self-reported)"); ax.set_ylabel(yl); grid(ax)
    r = ls.loc[col]
    ax.set_title(f"$\\rho$ = {r.rho:+.2f} (95% CI {r.ci_lo:+.2f} to {r.ci_hi:+.2f})", fontsize=7, loc="left")
hc_, lc_ = axes[0].get_legend_handles_labels()
ax = axes[2]
mets = ["duration_s", "w_mean", "w_p95", "peaks_per_s", "qlow_5", "pause_frac", "sparc"]
names = ["Trial duration", "Mean angular speed", "P95 angular speed", "Speed-peak rate", "Low-speed fraction", "Pause-time fraction", "SPARC"]
for i, m in enumerate(mets):
    y = len(mets) - 1 - i
    r = ls.loc[m]
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=INK, lw=1.0)
    ax.plot(r.rho, y, "o", ms=5, color=INK, mfc=INK if r.p < 0.05 else "white", mew=1.0)
    ax.plot(r.rho_within_cohort, y - 0.25, "D", ms=3.5, color=MUTED)
ax.axvline(0, color=INK, lw=0.6); ax.set_yticks(range(len(mets))); ax.set_yticklabels(names[::-1])
ax.set_xlim(-0.75, 0.75); ax.set_xlabel("Spearman $\\rho$ with experience"); grid(ax, "x")
h2_ = [Line2D([], [], marker="o", color=INK, lw=1, label="Pooled $\\rho$ (95% CI; filled: p < 0.05)"),
       Line2D([], [], marker="D", ls="none", color=MUTED, ms=3.5, label="Within-cohort ranks")]
for a, s_ in zip(axes, "abc"):
    panel(a, s_)
axes[2].set_title("All metrics", fontsize=7, loc="left")
fig.tight_layout(w_pad=1.0)
fig.legend(hc_ + h2_, lc_ + [h.get_label() for h in h2_], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=5,
           columnspacing=1.2, handletextpad=0.3)
save(fig, "fig6_lask_external")
print("ok", EX, early.trial, late.trial)
