"""Roll out a BAM actuator model against logs and score it (per-log position MAE).

Uses BAM's own simulator (external/bam, commit e9a619d) so numbers are comparable
with the BAM paper. Rollouts are open loop over the whole 6 s log: the model gets
the recorded goal positions and computes its own control, as in bam.fit.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external", "bam"))

from bam.actuators import actuators  # noqa: E402
from bam.model import models  # noqa: E402
from bam import simulate  # noqa: E402

from .ingest import SOURCES  # noqa: E402

PROCESSED = os.path.join(ROOT, "data", "processed")


def load_logs(part):
    logs = []
    for f in sorted(glob.glob(os.path.join(PROCESSED, part, "*.json"))):
        d = json.load(open(f))
        d["filename"] = os.path.basename(f)
        logs.append(d)
    return logs


def make_batch(logs):
    n = min(len(l["entries"]) for l in logs)
    batch = {k: np.array([l[k] for l in logs]) for k in logs[0]
             if k not in ("entries", "filename") and not isinstance(logs[0][k], (list, dict))}
    batch["dt"] = logs[0]["dt"]
    keys = logs[0]["entries"][0].keys()
    batch["entries"] = [{k: np.array([l["entries"][i][k] for l in logs]) for k in keys} for i in range(n)]
    return batch


def make_model(part, model_name, params=None):
    model = models[model_name]()
    model.set_actuator(actuators[SOURCES[part][1]]())
    if params:
        p = model.get_parameters()
        for k, v in params.items():
            if k in p:
                p[k].value = v
    return model


def per_log_mae(model, batch):
    """Mean |q_sim - q_log| per log [rad]."""
    sim = simulate.Simulator(model)
    pos = np.array(sim.rollout_log(batch, simulate_control=True)[0])          # (T, N)
    ref = np.array([e["position"] for e in batch["entries"]])                 # (T, N)
    err = np.abs(pos - ref)
    err = np.where(np.isfinite(err), err, np.pi)
    return err.mean(axis=0)


def score(model, batch):
    return float(per_log_mae(model, batch).mean())
