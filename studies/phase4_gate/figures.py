import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = json.load(open("results/sweep.json"))
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
x = np.array([r["ratio"] * r["friction_scale"] for r in rows])
y = np.array([r["backdriven_gap_deg"] for r in rows])
src = np.array([r["ratio"] > 1 for r in rows])
fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURF)
ax.set_facecolor(SURF)
o = np.argsort(x)
ax.plot(x[o], y[o], color=INK2, linewidth=1.2, zorder=2)
ax.scatter(x[~src], y[~src], s=48, color=S1, edgecolor=SURF, linewidth=2, zorder=3, label="same servo, friction scaled down")
ax.scatter(x[src], y[src], s=48, color=S2, edgecolor=SURF, linewidth=2, zorder=3, label="linkage ratio 2:1 ... 64:1")
for xi, yi, r in zip(x, y, rows):
    lab = f"{r['ratio']}:1" if r["ratio"] > 1 else (f"x{r['friction_scale']}" if r["friction_scale"] != 1 else "today's servo")
    ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 64]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "64"])
ax.set_ylim(0, 29)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.set_xlabel("friction seen at the output, relative to today's servo (ratio x friction scale)")
ax.set_ylabel("unit-to-unit gap while torque is off (deg)")
ax.text(0.1, 27.5, "transparent:\narm swings freely", fontsize=8.5, color=INK2, va="top")
ax.text(64, 27.5, "near self-locking:\narm barely falls", fontsize=8.5, color=INK2, va="top", ha="right")
ax.legend(frameon=False, loc="center right", fontsize=9, labelcolor=INK2)
ax.set_title("Back-driven gap peaks when friction is comparable to the load (STS3215 units A vs B)", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig("figures/backdrive_gap.png", dpi=150, facecolor=SURF)
