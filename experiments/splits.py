"""Train/test splits fixed in PLAN.md."""
import numpy as np

FRICTION_KEYS = ("friction_base", "friction_stribeck", "friction_viscous", "dtheta_stribeck", "alpha",
                 "load_friction_base", "load_friction_stribeck", "load_friction_motor", "load_friction_external",
                 "load_friction_motor_stribeck", "load_friction_external_stribeck",
                 "load_friction_motor_quad", "load_friction_external_quad")


def split(logs, kind):
    """Return (train_idx, test_idx)."""
    n = len(logs)
    idx = np.arange(n)
    if kind == "full":
        return idx, idx
    if kind == "random":
        perm = np.random.default_rng(0).permutation(n)
        k = int(round(0.8 * n))
        return np.sort(perm[:k]), np.sort(perm[k:])
    if kind == "heavy":
        g = np.array([round(l["mass"], 1) for l in logs])
        test = g == g.max()
    elif kind == "kp":
        kp = np.array([l["kp"] for l in logs])
        test = kp == kp.max()
    elif kind == "traj":
        test = np.array([l["trajectory"] == "lift_and_drop" for l in logs])
    else:
        raise ValueError(kind)
    return idx[~test], idx[test]
