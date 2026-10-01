"""H1: robot-to-robot spread of shoulder_pan hysteresis in SO-100 (STS3215) community datasets.

Method fixed in partgap/studies/business_validation/PLAN.md before the data was downloaded.
"""
import glob
import json

import numpy as np
from scipy.stats import spearmanr

TOL, MOVE = 0.3, 2.0     # deg: rest = command range <= 0.3 deg; approach = >= 2 deg net command move in the prior 1 s


def rests(a, s, fps):
    """Yield (direction, error) for each rest window of one episode."""
    L = int(round(0.5 * fps))
    i, n = fps, len(a)
    while i < n - L:
        j = i + 1
        lo = hi = a[i]
        while j < n:
            lo, hi = min(lo, a[j]), max(hi, a[j])
            if hi - lo > TOL:
                break
            j += 1
        if j - i >= L:
            move = a[i] - a[i - fps]
            if abs(move) >= MOVE:
                k = i + (j - i) // 2
                yield (1 if move > 0 else -1), float(np.mean(s[k:j] - a[k:j]))
            i = j
        else:
            i += 1


def hyst(items):
    pos = [e for d, e in items if d > 0]
    neg = [e for d, e in items if d < 0]
    if len(pos) < 5 or len(neg) < 5:
        return None
    return float(np.mean(neg) - np.mean(pos)), len(pos), len(neg)


def main():
    rows = []
    for f in sorted(glob.glob("data/*.npz")):
        d = np.load(f)
        names = [str(x) for x in d["names"]]
        snames = [str(x) for x in d["snames"]]
        ja = next((i for i, x in enumerate(names) if x.endswith("shoulder_pan")), None)
        js = next((i for i, x in enumerate(snames) if x.endswith("shoulder_pan")), None)
        if ja is None or js is None:
            continue
        fps = int(d["fps"])
        a_all, s_all, ep = d["action"][:, ja], d["state"][:, js], d["episode"]
        if np.nanmax(np.abs(a_all)) > 400:            # not in degrees
            continue
        items = {0: [], 1: []}
        for e in np.unique(ep):
            m = ep == e
            for r in rests(a_all[m], s_all[m], fps):
                items[int(e) % 2].append(r)
        h_all = hyst(items[0] + items[1])
        h0, h1 = hyst(items[0]), hyst(items[1])
        if h_all is None or h0 is None or h1 is None:
            continue
        rows.append({"dataset": f[5:-4], "H": h_all[0], "H_even": h0[0], "H_odd": h1[0],
                     "n_pos": h_all[1], "n_neg": h_all[2]})
    H = np.array([r["H"] for r in rows])
    rho, p = spearmanr([r["H_even"] for r in rows], [r["H_odd"] for r in rows])
    p10, p50, p90 = np.percentile(H, [10, 50, 90])
    summ = {"n_robots": len(rows), "H_p10": p10, "H_median": p50, "H_p90": p90,
            "p90_over_p10": p90 / p10 if p10 > 0 else None, "split_half_spearman": rho, "split_half_p": p,
            "share_negative_H": float((H <= 0).mean())}
    json.dump({"summary": summ, "robots": rows}, open("h1_results.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
