"""B2 的后续检查（均为看到 rho 分布之后才做，探索性）：
(a) 腕部传感器自身平均角速度 < 5 deg/s 的记录；剔除后 A2 腕部模型与 B1 双手指标是否变化；
(b) rho 相对本人中位数偏离较大的记录（部分失耦），剔除后主结果是否变化。
输出 results/b2_wrist_models_excl_lowwrist.csv、b2_bimanual_excl_lowwrist.csv。"""
import warnings
import numpy as np, pandas as pd
import h1_mixed_models as h1
from h1_mixed_models import fit, PRIMARY
import os as _os
_os.makedirs("results", exist_ok=True)
_os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
w = pd.read_csv("wrist_features.csv", dtype={"session": str})
low = w[w.w_mean < 5]
print("wrist records with mean speed < 5:", len(low), "\n", low.groupby(["participant", "hand", "session_idx"]).agg(n=("w_mean", "size"), med=("w_mean", "median"), mx=("w_mean", "max")).round(2).to_string())
print("wrist w_mean quantiles", w.w_mean.quantile([.001, .01, .05, .5]).round(2).to_dict(), "| lowest above 5:", w[w.w_mean >= 5].w_mean.nsmallest(5).round(2).tolist())
wk = w[w.w_mean >= 5]
rows = []
for hand in "LR":
    for f in PRIMARY:
        a, b = fit(w[w.hand == hand], f, True)[0], fit(wk[wk.hand == hand], f, True)[0]
        rows.append({"hand": hand, "feature": f, "b1_all": a.params["lo"], "p_all": a.pvalues["lo"], "b1_excl": b.params["lo"], "p_excl": b.pvalues["lo"], "n_excl": int((wk.hand == hand).sum())})
r = pd.DataFrame(rows); r.to_csv("results/b2_wrist_models_excl_lowwrist.csv", index=False); print(r.round(4).to_string())
bi = pd.read_csv("wrist_bimanual.csv")
bad = low[["participant", "trial"]].drop_duplicates().assign(x=1)
bk = bi.merge(bad, on=["participant", "trial"], how="left"); bk = bk[bk.x.isna()]
print("bimanual trials", len(bi), "->", len(bk))
rows = []
for f in ("bi_both", "bi_single", "bi_corr", "bi_idle"):
    a, b = fit(bi, f, True)[0], fit(bk, f, True)[0]
    a1, b1 = fit(bi, f, False)[0], fit(bk, f, False)[0]
    rows.append({"feature": f, "M1_all": a1.params["lo"], "M1_excl": b1.params["lo"], "M2_all": a.params["lo"], "p_all": a.pvalues["lo"], "M2_excl": b.params["lo"], "p_excl": b.pvalues["lo"], "ci_lo": b.conf_int().loc["lo"][0], "ci_hi": b.conf_int().loc["lo"][1]})
r = pd.DataFrame(rows); r.to_csv("results/b2_bimanual_excl_lowwrist.csv", index=False); print(r.round(4).to_string())

# (b) relative rule
h1.MIN_W_MEAN = 0.0
hd = h1.load()
j = hd.merge(wk[["participant", "trial", "hand", "w_mean"]], on=["participant", "trial", "hand"], suffixes=("", "_wr"))
j["rho"] = j.w_mean / j.w_mean_wr
j["rel"] = j.rho / j.groupby(["participant", "hand"]).rho.transform("median")
print("rel quantiles", j.rel.quantile([.001, .01, .05, .5, .95, .99, .999]).round(3).to_dict())
for th in (0.6, 0.7, 0.8):
    f_ = j[j.rel < th]
    print("rel <", th, len(f_), f_.groupby(["participant", "hand"]).size().to_dict())
print("rel > 1.4:", j[j.rel > 1.4].groupby(["participant", "hand"]).size().to_dict())
p10 = j[(j.participant == "P10") & (j.hand == "L")].sort_values("order")
print("P10 L rho by order:\n", p10[["order", "session_idx", "rho", "acc_mag_sd", "w_mean", "w_mean_wr"]].round(2).to_string(index=False))
ex = j[(j.rel < 0.7) & (j.w_mean >= 5)][["participant", "trial", "hand"]].assign(x=1)
base = hd[hd.w_mean >= 5]
k = base.merge(ex, on=["participant", "trial", "hand"], how="left"); k = k[k.x.isna()]
print("removed", len(base) - len(k))
for f in PRIMARY:
    a, b = fit(base[base.hand == "L"], f, True)[0], fit(k[k.hand == "L"], f, True)[0]
    print("L", f, round(a.params["lo"], 3), "->", round(b.params["lo"], 3), "p", round(b.pvalues["lo"], 4))
