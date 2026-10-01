"""figures/transfer.png: what carries over and what does not."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = json.load(open(os.path.join(ROOT, "results", "analysis.json")))
rob = json.load(open(os.path.join(ROOT, "results", "q2_robustness.json")))
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.1, 1]}, facecolor=SURF)
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)

splits = [("heavy", "heaviest load\nheld out"), ("kp", "highest P gain\nheld out"), ("traj", "drop test\n(lift_and_drop) held out")]
rng = np.random.default_rng(0)
for i, (s, _) in enumerate(splits):
    r = np.array([v["ratio"] for k, v in a["q1"].items() if k.endswith("/" + s)])
    x = i + rng.uniform(-0.12, 0.12, len(r))
    ax1.scatter(x, r, s=36, color=S1, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax1.text(i, r.max() * 1.25, f"{int((r <= 1.5).sum())}/{len(r)} pass", ha="center", color=INK, fontsize=9)
ax1.axhline(1.5, color=INK2, linestyle="--", linewidth=1)
ax1.text(-0.45, 1.5 * 1.05, "pass if ≤ x1.5", ha="left", color=INK2, fontsize=8)
ax1.set_yscale("log")
ax1.set_yticks([1, 1.5, 2, 5, 10, 20]); ax1.set_yticklabels(["x1", "x1.5", "x2", "x5", "x10", "x20"])
ax1.set_ylim(0.9, 40)
ax1.set_xticks(range(3)); ax1.set_xticklabels([l for _, l in splits])
ax1.set_xlim(-0.5, 2.5)
ax1.grid(axis="y", color=GRID, linewidth=0.8); ax1.set_axisbelow(True)
ax1.set_ylabel("error growth when the condition was never seen")
ax1.set_title("Within one unit: 6 servos x 2 friction models", loc="left", color=INK, fontsize=11)

m = rob["m1"]
cases = [("own_insample", "12 V unit,\nits own fit (in-sample)"), ("friction_only", "7.4 V friction,\nmotor refit on 12 V"),
         ("direct", "7.4 V entry\nused as is")]
x = np.arange(len(cases)); w = 0.36
for j, (ph, lab, col) in enumerate([("torque_on_deg", "powered", S1), ("torque_off_deg", "back-driven (torque off)", S2)]):
    vals = [m[c][ph] for c, _ in cases]
    bars = ax2.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=col, label=lab, zorder=3)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}°", ha="center", color=INK, fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels([l for _, l in cases])
ax2.set_ylabel("open-loop position error (deg)")
ax2.grid(axis="y", color=GRID, linewidth=0.8); ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)
ax2.set_title("Across units: Feetech STS3215 7.4 V → 12 V (M1, 97 logs)", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "transfer.png"), dpi=150, facecolor=SURF)
