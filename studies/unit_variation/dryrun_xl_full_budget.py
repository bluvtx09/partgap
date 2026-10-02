"""Dry-run check of U2 at the full fitting budget: XL330 x1-x3 (friction x0.95/1.0/1.05), M6, session 1 fits."""
import sys, json, numpy as np
from multiprocessing import Pool
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, 'analysis')
import run
data = run.load('dryrun/raw')
def job(u):
    p, s = run.fit_v2('xl330', 'm6', data[('xl330', u, 1, 'B1')], 1.0)
    print('fit', u, np.degrees(s), flush=True)
    return u, p
if __name__ == '__main__':
    with Pool(2) as pool:
        fits = dict(pool.map(job, ['x1', 'x2', 'x3']))
    T = {}
    for j in fits:
        test = data[('xl330', j, 2, 'B2')]
        own = run.phase_mae('dynamixel_xl330', 'm6', fits[j], test)['all']
        for i in fits:
            if i != j:
                other = run.phase_mae('dynamixel_xl330', 'm6', fits[i], test)['all']
                T[f'{i}->{j}'] = {'own_deg': own, 'other_deg': other, 'T': other / own, 'dE_deg': other - own}
    med = {k: float(np.median([v[k] for v in T.values()])) for k in ('T', 'dE_deg', 'own_deg')}
    print(json.dumps(T, indent=1), med)
    json.dump({'pairs': T, 'median': med, 'fits': fits}, open('dryrun/xl330_full_budget.json', 'w'), indent=1)
