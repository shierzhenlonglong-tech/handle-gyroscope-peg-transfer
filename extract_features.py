"""论文a 特征提取：SPD-FLS1 v2 手柄 IMU -> 每个手-试次一行。
用法：python extract_features.py <含 zip 或解压目录的根目录> <输出 csv> [--only part1]
质控沿用 oh2026_qc_sensitivity.py（论文2 脚本的副本，未改动）；陀螺按 rad/s 解释并换算为 deg/s。
"""
import sys, re, zipfile, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal
import oh2026_qc_sensitivity as q

FS = 1.0 / 0.032
CFG = q.QCConfig(gyro_scale=180.0 / math.pi)
TH = (2.0, 5.0, 10.0)


def parse_acc(text):
    out = {"LH": [], "RH": []}
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip().replace("−", "-").split(",")]
        if not parts or not parts[0]:
            continue
        if parts[0] in out:
            lab, kv = parts[0], parts[1:]
        elif len(parts) > 1 and parts[1] in out:
            lab, kv = parts[1], parts[2:]
        else:
            continue
        d = dict(i.split(":", 1) for i in kv if ":" in i)
        try:
            out[lab].append((float(d["Time"]) / 1000.0, float(d["Acc_X"]), float(d["Acc_Y"]), float(d["Acc_Z"])))
        except (KeyError, ValueError):
            pass
    return {h: np.array(v, float).reshape(-1, 4) for h, v in out.items()}


def sparc(sp, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """Balasubramanian et al. 2015 spectral arc length; more negative = less smooth."""
    if sp.size < 16 or sp.max() <= 0:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(sp.size)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(sp, nfft))
    Mf = Mf / Mf.max()
    sel = f <= fc
    f, Mf = f[sel], Mf[sel]
    idx = np.nonzero(Mf >= amp_th)[0]
    if idx.size < 2:
        return np.nan
    f, Mf = f[idx[0]:idx[-1] + 1], Mf[idx[0]:idx[-1] + 1]
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(Mf) ** 2)))


def ldlj(sp, fs):
    """Log dimensionless jerk of the angular-speed profile (sensitive to quantisation; reported with caution)."""
    if sp.size < 16 or sp.max() <= 0:
        return np.nan
    acc2 = np.gradient(np.gradient(sp, 1 / fs), 1 / fs)
    T = sp.size / fs
    return float(-np.log(T ** 3 / sp.max() ** 2 * np.sum(acc2 ** 2) / fs))


def pauses(sp, fs, th=5.0, min_s=0.5):
    low = np.concatenate([[0], (sp < th).astype(int), [0]])
    d = np.diff(low)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    dur = (e - s) / fs
    dur = dur[dur >= min_s]
    return int(dur.size), float(dur.sum())


_BA = signal.butter(4, 5.0 / (FS / 2))


def hand_features(segs):
    sps = [np.linalg.norm(y, axis=1) for y in segs]
    sp = np.concatenate(sps)
    T = sp.size / FS
    o = {"T_valid": T, "n_segments": len(segs), "w_mean": float(sp.mean()), "w_p50": float(np.median(sp)),
         "w_p95": float(np.quantile(sp, 0.95)), "ang_disp": float(sp.sum() / FS), "w_cv": float(sp.std() / sp.mean())}
    for th in TH:
        o[f"qlow_{th:g}"] = float(np.mean(sp < th))
    n_p = t_p = n_pk = 0
    sw = lw = s5 = 0.0
    for s in sps:
        a, b = pauses(s, FS)
        n_p += a
        t_p += b
        if s.size > 30:
            sf = signal.filtfilt(*_BA, s)
            n_pk += signal.find_peaks(sf, prominence=5.0)[0].size
        v = sparc(s, FS)
        l = ldlj(s, FS)
        v5 = sparc(s, FS, fc=5.0)
        if not np.isnan(v):
            sw += v * s.size
            lw += (l if not np.isnan(l) else 0) * s.size
            s5 += (v5 if not np.isnan(v5) else 0) * s.size
    o.update({"pause_n": n_p, "pause_rate": n_p / T, "pause_frac": t_p / T, "peaks_per_s": n_pk / T,
              "sparc": sw / sp.size if sw else np.nan, "ldlj": lw / sp.size if lw else np.nan,
              "sparc_fc5": s5 / sp.size if s5 else np.nan})
    return o, sps


def bimanual(clean):
    """Coarse bimanual features. The two handle IMUs have separate device clocks; each is referenced to its own
    first sample, so alignment error of a few tens of ms is possible. Only 1 s-smoothed features are computed."""
    (tl, gl), (tr, gr) = clean["L"], clean["R"]
    tl, tr = tl - tl[0], tr - tr[0]
    end = min(tl[-1], tr[-1])
    if end < 5:
        return {}
    tg = np.arange(0, end, 1 / FS)
    sl = np.interp(tg, tl, np.linalg.norm(gl, axis=1))
    sr = np.interp(tg, tr, np.linalg.norm(gr, axis=1))
    k = np.ones(int(FS)) / int(FS)
    sl_s, sr_s = np.convolve(sl, k, "same"), np.convolve(sr, k, "same")
    return {"bi_corr": float(np.corrcoef(sl_s, sr_s)[0, 1]), "bi_both_move": float(np.mean((sl_s > 5) & (sr_s > 5))),
            "bi_both_idle": float(np.mean((sl_s <= 5) & (sr_s <= 5))), "bi_left_share": float(sl.sum() / (sl.sum() + sr.sum()))}


def main(root, out, only=None, noise=0.0):
    trials, _ = q.discover(Path(root))
    rows = []
    for tr in trials:
        if only and only not in str(tr.members.get("log", tr.members.get("imu", ("", "", 0)))[0]):
            continue
        mem = str(tr.members.get("log", tr.members.get("imu", ("", "", 0)))[1]).replace("\\", "/")
        sm = re.search(r"(P\d+)_FLS Task A_(\d+)", mem)   # subset zips nest the session folder inside the archive
        pid, ses = (sm.group(1).upper(), f"{int(sm.group(2)):02d}") if sm else (tr.participant, tr.session)
        ident = {"participant": pid, "session": ses, "trial": tr.key, "ts": tr.key.split("_")[0],
                 "block": tr.block, "trial_no": tr.trial}
        T_log = drops = None
        comments = ""
        if "log" in tr.members and tr.members["log"][2] > 0:
            txt = q._read(tr.members["log"]).decode("utf-8", errors="replace")
            lg = q.parse_log(txt)
            T_log, drops = lg["T_log"], lg["dropped"]
            m = re.search(r"Comments:(.*)", txt, re.S)
            comments = " ".join(m.group(1).split()) if m else ""
        base = {**ident, "T_log": T_log, "dropped": drops, "comments": comments}
        imu = tr.members.get("imu")
        if imu is None or imu[2] == 0:
            rows.append({**base, "hand": "none", "qc": "no IMU"})
            continue
        text = q._read(imu).decode("utf-8", errors="replace")
        data, _ = q.parse_imu(text, CFG.gyro_scale)
        acc = parse_acc(text)
        clean = {}
        hrows = []
        for h, hh in (("LH", "L"), ("RH", "R")):
            stage, reason, info, cl = q.qc_hand(data[h], T_log, CFG)
            r = {**base, "hand": hh, "qc": reason, "n_rows": info.get("n_rows")}
            if stage == "S6":
                segs = q.segments_uniform(cl[0], cl[1], FS, CFG.gap_factor, CFG.min_segment_s)
                if segs and noise > 0:   # robustness: additive white noise per axis (deg/s), deterministic per hand-trial
                    rng = np.random.default_rng(q._seed(pid, ses, tr.key, hh, noise))
                    segs = [y + rng.normal(0.0, noise, y.shape) for y in segs]
                if segs:
                    f, _ = hand_features(segs)
                    r.update(f)
                    a = acc[h]
                    if a.shape[0] > 10:
                        am = np.linalg.norm(a[:, 1:4], axis=1)
                        r["acc_mag_sd"] = float(am.std())
                        r["acc_mag_mean"] = float(am.mean())
                    clean[hh] = cl
                else:
                    r["qc"] = "no continuous segment"
            hrows.append(r)
        if len(clean) == 2:
            b = bimanual(clean)
            for r in hrows:
                r.update(b)
        rows.extend(hrows)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows))


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    noise = float(sys.argv[sys.argv.index("--noise") + 1]) if "--noise" in sys.argv else 0.0
    if "--unit-deg" in sys.argv:      # alternative reading: logged gyro values already in deg/s (no conversion)
        import dataclasses
        CFG = dataclasses.replace(CFG, gyro_scale=1.0)
    main(sys.argv[1], sys.argv[2], only, noise)
