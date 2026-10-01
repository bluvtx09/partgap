"""P0: does my CMA-ES fitter reach the MAE of BAM's published parameters (fit on all logs)?"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from partgap.evaluate import load_logs, make_batch, make_model, score
from partgap.fit import fit

part, act, m, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
b = make_batch(load_logs(part))
pub = json.load(open(f"external/bam/bam/params/{act}/{m}.json"))
s_pub = score(make_model(part, m, pub), b)
t = time.time()
p, s, used = fit(part, m, b, budget=budget)
print(f"{part} {m} budget {budget} used {used}: mine {np.degrees(s):.3f} deg, published {np.degrees(s_pub):.3f} deg, "
      f"ratio {s / s_pub:.3f}, {time.time() - t:.0f}s")
