"""Fit BAM friction-model parameters with CMA-ES (cmaes library, bounded, normalized space).

Same objective as bam.fit: mean over logs of the open-loop position MAE.
"""
import numpy as np
from cmaes import CMA

from .evaluate import make_model, score


def fit(part, model_name, batch, budget=3000, seed=0, fixed=None, start=None, restarts=1):
    """Return (best_params, best_score). `fixed` = {name: value} kept constant."""
    fixed = fixed or {}
    proto = make_model(part, model_name, start)
    params = proto.get_parameters()
    names = [n for n, p in params.items() if p.optimize and n not in fixed]
    lo = np.array([params[n].min for n in names])
    hi = np.array([params[n].max for n in names])
    x0 = np.clip((np.array([params[n].value for n in names]) - lo) / (hi - lo), 0, 1)

    def evaluate(x):
        vals = {n: float(lo[i] + x[i] * (hi[i] - lo[i])) for i, n in enumerate(names)}
        vals.update(fixed)
        return score(make_model(part, model_name, {**(start or {}), **vals}), batch), vals

    best = evaluate(x0)
    used = 1
    rng = np.random.default_rng(seed)
    per_run = budget // restarts
    for r in range(restarts):
        mean = x0 if r == 0 else rng.uniform(0.1, 0.9, len(names))
        opt = CMA(mean=mean, sigma=0.25, bounds=np.array([[0.0, 1.0]] * len(names)), seed=seed + r)
        run_used = 0
        while run_used < per_run and not opt.should_stop():
            sols = []
            for _ in range(opt.population_size):
                x = opt.ask()
                s, vals = evaluate(x)
                sols.append((x, s))
                if s < best[0]:
                    best = (s, vals)
            opt.tell(sols)
            run_used += opt.population_size
        used += run_used
    params_out = {**(start or {}), **best[1]}
    params_out["model"] = model_name
    return params_out, best[0], used
