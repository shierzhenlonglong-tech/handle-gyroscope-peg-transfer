"""H2 预后分析（冻结方案第 6 节，探索性）。留一受试者交叉验证；受试者聚类 bootstrap。"""
import warnings, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from h1_mixed_models import PRIMARY
warnings.filterwarnings("ignore")
K = 5
rng = np.random.default_rng(20261110)

f = pd.read_csv("../02_数据/trial_features.csv", dtype={"session": str})
f = f[f.valid_trial.astype(bool)]
idx = f.drop_duplicates(["participant", "trial"])[["participant", "trial", "order", "T_log", "dropped", "proficient"]]
feat = f[(f.qc == "pass") & (f.w_mean >= 5.0)].groupby(["participant", "trial"])[PRIMARY].mean().reset_index()   # mean over available hands
t = idx.merge(feat, on=["participant", "trial"], how="left").sort_values(["participant", "order"])
t["lT"] = np.log(t.T_log)
t["y"] = t.proficient.astype(int)
g = t.groupby("participant")
t["lo"] = np.log(t.order)
t["lT_roll"] = g.lT.transform(lambda s: s.shift(1).rolling(K, min_periods=K).mean())
t["lT_min"] = g.lT.transform(lambda s: s.shift(1).rolling(K, min_periods=K).min())
for c in PRIMARY:
    t[c + "_roll"] = g[c].transform(lambda s: s.shift(1).rolling(K, min_periods=3).mean())
BASE = ["lo", "lT_roll", "lT_min"]
IMU = [c + "_roll" for c in PRIMARY]
d = t.dropna(subset=BASE + IMU).copy()
d["prior_events"] = d.groupby("participant").y.cumsum() - d.y
print("person-trials:", len(d), "events:", int(d.y.sum()), "participants:", d.participant.nunique())


def loso(data, cols):
    pred = pd.Series(index=data.index, dtype=float)
    for pid in data.participant.unique():
        tr, te = data[data.participant != pid], data[data.participant == pid]
        if tr.y.nunique() < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(tr[cols], tr.y)
        pred[te.index] = m.predict_proba(te[cols])[:, 1]
    return pred


def evaluate(data, name):
    out = {"analysis": name, "n": len(data), "events": int(data.y.sum())}
    P = {"time": loso(data, BASE), "imu": loso(data, ["lo"] + IMU), "time_imu": loso(data, BASE + IMU)}
    for k, p in P.items():
        out[f"auc_{k}"] = roc_auc_score(data.y, p)
        out[f"brier_{k}"] = brier_score_loss(data.y, p)
    pids = data.participant.unique()
    groups = {pid: np.where(data.participant.values == pid)[0] for pid in pids}
    y = data.y.values
    dA, dB = [], []
    for _ in range(2000):
        ii = np.concatenate([groups[p] for p in rng.choice(pids, len(pids))])
        if y[ii].min() == y[ii].max():
            continue
        dA.append(roc_auc_score(y[ii], P["time_imu"].values[ii]) - roc_auc_score(y[ii], P["time"].values[ii]))
        dB.append(brier_score_loss(y[ii], P["time_imu"].values[ii]) - brier_score_loss(y[ii], P["time"].values[ii]))
    out["d_auc"] = out["auc_time_imu"] - out["auc_time"]
    out["d_auc_ci"] = [float(np.quantile(dA, 0.025)), float(np.quantile(dA, 0.975))]
    out["d_brier"] = out["brier_time_imu"] - out["brier_time"]
    out["d_brier_ci"] = [float(np.quantile(dB, 0.025)), float(np.quantile(dB, 0.975))]
    out["base_rate_brier"] = float(np.mean((y - y.mean()) ** 2))
    return out, P


resA, PA = evaluate(d, "A_proficient_next_trial")
atrisk = d[d.prior_events == 0]          # discrete-time hazard of the first proficient trial
resB, PB = evaluate(atrisk, "B_first_proficient_hazard")
# subject-level concordance (descriptive, n = 19): mean predicted hazard over the first 10 at-risk person-trials vs time to first event
rows = []
for pid, gg in atrisk.groupby("participant"):
    ev = gg[gg.y == 1]
    rows.append({"participant": pid, "time": ev.order.min() if len(ev) else gg.order.max(), "event": int(len(ev) > 0),
                 "risk_time": PB["time"][gg.index[:10]].mean(), "risk_time_imu": PB["time_imu"][gg.index[:10]].mean()})
s = pd.DataFrame(rows)


def cindex(time, event, risk):
    num = den = 0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if time[i] < time[j] and event[i]:
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den


resB["c_index_time"] = cindex(s.time.values, s.event.values, s.risk_time.values)
resB["c_index_time_imu"] = cindex(s.time.values, s.event.values, s.risk_time_imu.values)
json.dump([resA, resB], open("../07_results/h2_prognostic.json", "w"), indent=1)
d.assign(p_time=PA["time"], p_imu=PA["imu"], p_time_imu=PA["time_imu"])[["participant", "trial", "order", "y", "p_time", "p_imu", "p_time_imu"]].to_csv("../07_results/h2_predictions_A.csv", index=False)
s.to_csv("../07_results/h2_subject_level.csv", index=False)
for r in (resA, resB):
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
