"""补充方案 B1：同步腕部传感器的双手协调指标。每试次一行。
用法：python extract_wrist_bimanual.py <shimmer_raw 目录> <输出 csv> [k n]   （分块输出命名为 wristbi_part{k}.csv，k = 0..11）
需要 SPD-FLS1 腕部 Shimmer 原始文件（不随仓库发布）。"""
import sys, re, io, glob, os
import numpy as np, pandas as pd
import oh2026_qc_sensitivity as q
from extract_features import FS
from extract_wrist import CFG, STRIP, NAMES

LEFT, RIGHT = "Shimmer_DA3F", "Shimmer_DB28"     # mapping established in a2_wrist_vs_handle.py (all 19 participants)
K = 16                                           # 0.5 s moving average at 31.25 Hz
TH = 5.0


def main(root, out, k=0, n=1):
    files = sorted(glob.glob(os.path.join(root, "**", "*_Shimmer.txt"), recursive=True))
    rows = []
    for i, f in enumerate(files):
        if i % n != k or os.path.getsize(f) < 2000:
            continue
        m = re.search(r"(P\d+)_FLS Task A_(\d+)", f.replace("\\", "/"))
        key = re.search(r"(\d{14})[_ ]Trial(\d+)(?:-(\d+))?_Shimmer", os.path.basename(f))
        if not m or not key:
            continue
        trial = f"{key.group(1)}_Trial{key.group(2)}-{int(key.group(3)) if key.group(3) else 0}"
        try:
            txt = STRIP.sub("", open(f, encoding="utf8", errors="replace").read())
            d = pd.read_csv(io.StringIO(txt), header=None, on_bad_lines="skip", usecols=range(13), names=NAMES)
            for c in NAMES[2:9]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=NAMES[2:9])
        except Exception:
            continue
        cl = {}
        for dev, g in d.groupby("dev"):
            arr = np.column_stack([g.t.values / 1000.0, g.gx.values, g.gy.values, g.gz.values])
            stage, _, _, c = q.qc_hand(arr, None, CFG)
            if stage == "S6":
                cl[dev] = c
        if LEFT not in cl or RIGHT not in cl:
            continue
        (tl, gl), (tr, gr) = cl[LEFT], cl[RIGHT]
        t0, t1 = max(tl[0], tr[0]), min(tl[-1], tr[-1])
        if t1 - t0 < 10:
            continue
        tg = np.arange(t0, t1, 1 / FS)
        ker = np.ones(K) / K
        sl = np.convolve(np.interp(tg, tl, np.linalg.norm(gl, axis=1)), ker, "valid")
        sr = np.convolve(np.interp(tg, tr, np.linalg.norm(gr, axis=1)), ker, "valid")
        ml, mr = sl >= TH, sr >= TH
        a, b = sl - sl.mean(), sr - sr.mean()
        L = int(2 * FS)
        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        xc = np.array([np.sum(a[max(0, -j):len(a) - max(0, j)] * b[max(0, j):len(b) - max(0, -j)]) for j in range(-L, L + 1)]) / den
        jm = int(np.argmax(np.abs(xc)))
        rows.append({"participant": m.group(1), "trial": trial, "overlap_s": float(t1 - t0), "start_offset_s": float(tl[0] - tr[0]),
                     "bi_both": float(np.mean(ml & mr)), "bi_single": float(np.mean(ml ^ mr)), "bi_idle": float(np.mean(~ml & ~mr)),
                     "bi_left_only": float(np.mean(ml & ~mr)), "bi_right_only": float(np.mean(~ml & mr)),
                     "bi_corr": float(xc[L]), "bi_lag": float((jm - L) / FS), "bi_xcmax": float(xc[jm])})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 1)
