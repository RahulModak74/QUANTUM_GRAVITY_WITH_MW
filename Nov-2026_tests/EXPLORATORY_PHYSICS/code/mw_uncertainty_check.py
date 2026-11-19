#!/usr/bin/env python3
"""
Position-time bound on the manifold:  dx*dt >= l_P^2 (1 + beta R l_P^2)   (Appendix A, Jan 2026)

On the same 240 test states as mw_exact_geometry.py (l_P = 1):
  * fraction of decoded states with dx*dt >= l_P^2
  * beta_eff = (dx*dt / l_P^2 - 1) / R with the EXACT Ricci scalar, and the correlation of the
    excess with R.
Note: the v5 generator builds dx*dt >= l_P^2 into the data (dt = l_P^2/dx * (1 + Exp)), with no R
dependence, and prior C3 enforces the bound. The manifold can therefore check consistency of the
bound but cannot measure beta; beta remains a prediction for experiment.
Writes ../results/uncertainty_check.json
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mw_exact_geometry as m                                                    # noqa: E402
from qg_toy_data_generator_v5 import QGJointGenerator, QGParams                  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    d, lab, _ = QGJointGenerator(QGParams(seed=123, n_samples=6000)).generate()
    out = {'data_min_dxdt': float(d[:, m.CI['dxdt']].min())}
    for name in ['A_full_prior', 'B_full_noprior']:
        vae = m.load(name)
        with torch.no_grad():
            mu, _ = vae.encode((torch.tensor(d, dtype=torch.float64) - vae.mean) / vae.std)
        act = [i for i in range(20) if i not in (3, 8)]
        rng = np.random.default_rng(7)                       # same 240 states as mw_exact_geometry.py
        idx = np.concatenate([rng.choice(np.where(lab == k)[0], 40, replace=False) for k in range(6)])
        with torch.no_grad():
            b = m.geom(vae, mu[idx], act)
            R = m.einstein(b['II'], b['g'], b['gi'])[2].numpy()
            x = vae.denormalize(vae.decode(mu[idx])).numpy()
        dxdt = x[:, m.CI['dxdt']]; ex = dxdt - 1.0
        beta = ex / R
        rk = lambda v: np.argsort(np.argsort(v))
        out[name] = {'states': int(len(idx)), 'frac_bound_holds': float((dxdt >= 1.0).mean()),
                     'min_dxdt': float(dxdt.min()), 'beta_eff_median': float(np.median(beta)),
                     'beta_eff_iqr': [float(v) for v in np.percentile(beta, [25, 75])],
                     'pearson_R_excess': float(np.corrcoef(R, ex)[0, 1]),
                     'spearman_R_excess': float(np.corrcoef(rk(R), rk(ex))[0, 1])}
        print(name, out[name])
    json.dump(out, open(os.path.join(HERE, '..', 'results', 'uncertainty_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
