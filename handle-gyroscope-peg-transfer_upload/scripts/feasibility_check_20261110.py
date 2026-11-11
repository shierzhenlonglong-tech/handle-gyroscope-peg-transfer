"""可行性检查（2026-11-10）：手柄 IMU 指标是否随练习变化，且在控制完成时间后仍有趋势。
输入：论文2/oh2026_out/trial_metrics_baseline.csv（oh2026_qc_sensitivity.py 的输出）。
用法：python feasibility_check_20261110.py <trial_metrics_baseline.csv>
已知缺陷：试次序号按时间戳排序，未剔除每人第一会话的 30 s 校准记录；未处理掉落试次。正式分析需修正。
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1], dtype={"session": str})
d["ts"] = d.trial.str.split("_").str[0]
key = d[["participant", "ts"]].drop_duplicates().sort_values(["participant", "ts"])
key["order"] = key.groupby("participant").cumcount() + 1
d = d.merge(key, on=["participant", "ts"]).sort_values(["participant", "order"])
M = ["M04", "Qlow_5", "cth_5", "speed_p95"]
d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=M + ["T_log"])
d = d[d.T_log > 0]
for hand in "LR":
    rows = []
    for p, g in d[d.hand == hand].groupby("participant"):
        if len(g) < 20:
            continue
        x = np.log(g.T_log.values)
        ro = g.order.values - np.polyval(np.polyfit(x, g.order.values, 1), x)
        r = {"participant": p, "n": len(g), "rho_T": stats.spearmanr(g.order, g.T_log)[0]}
        for m in M:
            y = g[m].values
            r["rho_" + m] = stats.spearmanr(g.order, y)[0]
            r["partial_" + m] = stats.spearmanr(ro, y - np.polyval(np.polyfit(x, y, 1), x))[0]
        rows.append(r)
    r = pd.DataFrame(rows).set_index("participant")
    print("HAND", hand)
    print(r.round(2).to_string())
    print("median:", r.median().round(2).to_dict())
    for c in [c for c in r.columns if c.startswith("partial_")]:
        print(c, "n>0 =", int((r[c] > 0).sum()), "/", len(r), " Wilcoxon p =", round(stats.wilcoxon(r[c])[1], 4))
