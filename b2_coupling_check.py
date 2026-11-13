"""补充方案 B2：手柄/腕部平均角速度比 rho 作为传感器耦合检查。输出 results/b2_rule_summary.csv、b2_flagged.csv、b2_rho_by_participant.csv。
读取仓库根目录的 trial_features.csv、wrist_features.csv、wrist_bimanual.csv。
另含一项未预先指定的探索性检查（B1 的延伸）：双手同时运动占比超出独立假设预期的部分。"""
import warnings
import numpy as np, pandas as pd
import h1_mixed_models as h1
from h1_mixed_models import fit, PRIMARY
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
LO, HI = 0.25, 5.0
h1.MIN_W_MEAN = 0.0
hd = h1.load()                                  # handle records passing conventional QC, before D1
w = pd.read_csv("wrist_features.csv", dtype={"session": str})
j = hd.merge(w[["participant", "trial", "hand", "w_mean", "acc_mag_sd"]], on=["participant", "trial", "hand"], suffixes=("", "_wr"))
j["rho"] = j.w_mean / j.w_mean_wr
j["d1"] = j.w_mean < 5.0
print("handle records before D1:", hd.groupby("hand").size().to_dict(), "| with wrist pair:", j.groupby("hand").size().to_dict())
print("D1 records (all):", int((hd.w_mean < 5).sum()), "| D1 records with a wrist pair:", int(j.d1.sum()))
qs = j.rho.quantile([.001, .01, .05, .5, .95, .99, .999])
print("rho quantiles:\n", qs.round(3).to_string())
rows = []
for lo in (0.15, 0.25, 0.35):
    fl = j.rho < lo
    rows.append({"rule": f"rho<{lo}", "flagged": int(fl.sum()), "of_which_D1": int((fl & j.d1).sum()), "D1_missed": int((~fl & j.d1).sum()),
                 "extra_not_D1": int((fl & ~j.d1).sum()), "participants": ",".join(sorted(j[fl].participant.unique()))})
for hi in (4.0, 5.0, 6.0):
    fl = j.rho > hi
    rows.append({"rule": f"rho>{hi}", "flagged": int(fl.sum()), "of_which_D1": 0, "D1_missed": np.nan, "extra_not_D1": int(fl.sum()), "participants": ",".join(sorted(j[fl].participant.unique()))})
s = pd.DataFrame(rows)
s.to_csv("results/b2_rule_summary.csv", index=False)
print(s.to_string())
fl = j[(j.rho < LO) | (j.rho > HI) | j.d1].sort_values(["participant", "order"])
fl[["participant", "session", "trial", "hand", "order", "session_idx", "qc", "n_rows", "T_valid", "T_log", "w_mean", "w_mean_wr", "rho", "acc_mag_sd", "d1"]].to_csv("results/b2_flagged.csv", index=False)
print(fl.groupby(["participant", "hand", "session_idx"]).agg(n=("rho", "size"), rho_med=("rho", "median"), rho_max=("rho", "max"), h=("w_mean", "median"), wr=("w_mean_wr", "median"), d1=("d1", "sum")).round(3).to_string())
# gap between flagged and unflagged
ok = j[~j.d1]
print("lowest rho among non-D1 records:", ok.rho.nsmallest(8).round(3).tolist(), "| highest:", ok.rho.nlargest(8).round(2).tolist())
print("highest rho among D1 records:", j[j.d1].rho.max().round(3) if j.d1.any() else None)
# the participant-session of the fault: all records incl. left hand, and neighbours
p = fl[fl.d1].participant.unique()
for pid in p:
    g = j[j.participant == pid]
    print(pid, g.groupby(["hand", "session_idx"]).agg(n=("rho", "size"), rho_med=("rho", "median"), rho_min=("rho", "min"), rho_max=("rho", "max")).round(3).to_string())
# P10 left hand description
g = j[(j.participant == "P10")]
g = g.assign(third=pd.qcut(g.order, 3, labels=["early", "mid", "late"]))
print("P10:\n", g.groupby(["hand", "third"]).agg(n=("rho", "size"), rho_med=("rho", "median"), rho_min=("rho", "min"), acc_sd_handle=("acc_mag_sd", "median"), acc_sd_wrist=("acc_mag_sd_wr", "median"), h=("w_mean", "median"), wr=("w_mean_wr", "median")).round(3).to_string())
# per-participant median rho (between-person spread of the ratio)
pp = ok.groupby(["participant", "hand"]).rho.median().unstack()
print("per-participant median rho: min", pp.min().round(2).to_dict(), "max", pp.max().round(2).to_dict())
pp.to_csv("results/b2_rho_by_participant.csv")
# effect on main results if extra flagged records (beyond D1) are removed
extra = j[((j.rho < LO) | (j.rho > HI)) & ~j.d1][["participant", "trial", "hand"]]
print("extra flagged beyond D1:", len(extra))
if len(extra):
    base = hd[hd.w_mean >= 5.0]
    k = base.merge(extra.assign(x=1), on=["participant", "trial", "hand"], how="left")
    k = k[k.x.isna()]
    for hand in "LR":
        for f in PRIMARY:
            a, b_ = fit(base[base.hand == hand], f, True)[0], fit(k[k.hand == hand], f, True)[0]
            print(hand, f, round(a.params["lo"], 3), "->", round(b_.params["lo"], 3), "p", round(b_.pvalues["lo"], 4))

# ---- exploratory (not pre-specified): simultaneous movement beyond what independent hands would give ----
b = pd.read_csv("wrist_bimanual.csv")
pl, pr = b.bi_both + b.bi_left_only, b.bi_both + b.bi_right_only
b["bi_excess"] = b.bi_both - pl * pr
b["bi_any"] = 1 - b.bi_idle
for f in ("bi_excess",):
    m1, m2 = fit(b, f, False)[0], fit(b, f, True)[0]
    print("exploratory", f, "mean", round(b[f].mean(), 3), "M1", round(m1.params["lo"], 3), round(m1.pvalues["lo"], 4), "M2", round(m2.params["lo"], 3), round(m2.pvalues["lo"], 4), "logT", round(m2.params["lT"], 3))
print("within-person r of bi_both with left/right moving share:", round(np.corrcoef(b.bi_both - b.groupby("participant").bi_both.transform("mean"), pl - pl.groupby(b.participant).transform("mean"))[0, 1], 3),
      round(np.corrcoef(b.bi_both - b.groupby("participant").bi_both.transform("mean"), pr - pr.groupby(b.participant).transform("mean"))[0, 1], 3))
