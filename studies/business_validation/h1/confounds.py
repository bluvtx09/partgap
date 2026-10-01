"""Exploratory (after H1 result): is the robot-to-robot spread explained by recording date (LeRobot default changes),
approach speed, or how much the arm moves? Also a robust spread measure."""
import json, glob
import numpy as np
from scipy.stats import spearmanr
from analyze_h1 import rests, TOL

res = json.load(open("h1_results.json"))
meta = {d["id"].replace("/", "__"): d for d in json.load(open("picked.json"))}
rows = res["robots"]
H = np.array([r["H"] for r in rows])
dates = np.array([np.datetime64(meta[r["dataset"]]["created"][:10]).astype("datetime64[D]").astype(float) for r in rows])
speed, err_move = [], []
for r in rows:
    d = np.load(f"data/{r['dataset']}.npz")
    names = [str(x) for x in d["names"]]
    j = next(i for i, x in enumerate(names) if x.endswith("shoulder_pan"))
    a, s = d["action"][:, j], d["state"][:, j]
    v = np.abs(np.diff(a)) * int(d["fps"])
    speed.append(float(np.percentile(v[v > 1], 75)) if (v > 1).any() else 0.0)
    moving = v > 20
    err_move.append(float(np.median(np.abs((s - a)[1:][moving]))) if moving.any() else np.nan)
speed, err_move = np.array(speed), np.array(err_move)
out = {
    "iqr_ratio_p75_p25": float(np.percentile(H, 75) / np.percentile(H, 25)),
    "p90_over_median": float(np.percentile(H, 90) / np.median(H)),
    "spearman_H_vs_date": [float(x) for x in spearmanr(H, dates)],
    "spearman_H_vs_speed": [float(x) for x in spearmanr(H, speed)],
    "spearman_H_vs_tracking_error_while_moving": [float(x) for x in spearmanr(H[~np.isnan(err_move)], err_move[~np.isnan(err_move)])],
}
# spread within a narrow date band and speed band (controls)
early = dates < np.median(dates)
out["p90_p10_older_half"] = float(np.percentile(H[early], 90) / np.percentile(H[early], 10))
out["p90_p10_newer_half"] = float(np.percentile(H[~early], 90) / np.percentile(H[~early], 10))
q1, q3 = np.percentile(speed, [33, 67])
mid = (speed >= q1) & (speed <= q3)
out["n_mid_speed"] = int(mid.sum())
out["p90_p10_mid_speed_third"] = float(np.percentile(H[mid], 90) / np.percentile(H[mid], 10))
out["p75_p25_mid_speed_third"] = float(np.percentile(H[mid], 75) / np.percentile(H[mid], 25))
# same uploader-independent check: how many robots have H within 1 encoder tick (0.088 deg)
out["share_H_below_1tick"] = float((H < 0.088).mean())
json.dump(out, open("h1_confounds.json", "w"), indent=1)
print(json.dumps(out, indent=1))
