"""Field reference for U3 (PLAN.md): spread of shoulder_pan hysteresis across n random SO-100 arms.

Draws n arms (without replacement) from the 193 arms of the H1 analysis, 20,000 times, and
reports percentiles of the range (max - min) in degrees.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H1 = os.path.join(HERE, "..", "..", "business_validation", "h1", "h1_results.json")


def field_range(n, draws=20000, seed=0):
    H = np.array([r["H"] for r in json.load(open(H1))["robots"]])
    rng = np.random.default_rng(seed)
    r = np.array([np.ptp(rng.choice(H, n, replace=False)) for _ in range(draws)])
    return {"n_arms_pool": len(H), "n": n, **{f"p{p}": float(np.percentile(r, p)) for p in (10, 25, 50, 75, 90)}}


if __name__ == "__main__":
    out = {str(n): field_range(n) for n in (3, 5)}
    json.dump(out, open(os.path.join(HERE, "field_reference.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
