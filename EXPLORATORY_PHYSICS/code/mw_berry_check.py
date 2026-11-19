#!/usr/bin/env python3
"""
Quantum phase connection built from the decoded wavefunction psi = psi_re + i psi_im:

    A_a = Im(psi* d_a psi) / |psi|^2 = (psi_re d_a psi_im - psi_im d_a psi_re) / |psi|^2   (= d_a theta)
    F_ab = d_a A_b - d_b A_a

For a single complex amplitude A is the gradient of the phase, so F = 0 wherever psi != 0:
the connection term hbar A_a dz^a in the v1 action is exact and only adds an endpoint phase.
This script evaluates F with exact derivatives (mw_taylor jets) on test states and reports
|F| relative to |dA|, plus the phase accumulated along the black-hole geodesic.
Writes ../results/berry_check.json
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mw_exact_geometry as m                                   # noqa: E402
from mw_taylor import decoder_jet                                # noqa: E402
from qg_toy_data_generator_v5 import QGJointGenerator, QGParams, SCENARIOS   # noqa: E402

torch.set_default_dtype(torch.float64)
iR, iI = m.CI['psi_re'], m.CI['psi_im']


def main():
    d, lab, _ = QGJointGenerator(QGParams(seed=123, n_samples=6000)).generate()
    out = {}
    for name in ['A_full_prior', 'B_full_noprior']:
        vae = m.load(name)
        with torch.no_grad():
            mu, lv = vae.encode((torch.tensor(d, dtype=torch.float64) - vae.mean) / vae.std)
        act = [i for i in range(20) if i not in (3, 8)]
        idx = np.random.default_rng(7).choice(len(d), 600, replace=False)
        with torch.no_grad():
            f, J, H = decoder_jet(vae, mu[idx], act)
        s = vae.std
        re, im = f[:, iR] * s[iR] + vae.mean[iR], f[:, iI] * s[iI] + vae.mean[iI]
        dre, dim_ = J[:, iR] * s[iR], J[:, iI] * s[iI]
        Hre, Him = H[:, iR] * s[iR], H[:, iI] * s[iI]
        n2 = re ** 2 + im ** 2
        num = re[:, None] * dim_ - im[:, None] * dre                       # [B, n]
        # d_b A_a = d_b(num_a)/n2 - num_a d_b(n2)/n2^2
        dnum = (dre[:, None, :] * dim_[:, :, None] + re[:, None, None] * Him
                - dim_[:, None, :] * dre[:, :, None] - im[:, None, None] * Hre)   # [B, a, b]
        dn2 = 2 * (re[:, None] * dre + im[:, None] * dim_)
        dA = dnum / n2[:, None, None] - num[:, :, None] * dn2[:, None, :] / n2[:, None, None] ** 2
        F = dA - dA.transpose(1, 2)
        rel = (torch.linalg.norm(F, dim=(1, 2)) / torch.linalg.norm(dA, dim=(1, 2))).numpy()
        out[name] = {'points': int(len(idx)), 'F_over_dA_median': float(np.median(rel)),
                     'F_over_dA_max': float(np.max(rel)),
                     'psi_abs_min': float(n2.sqrt().min())}
        print(name, out[name])
    json.dump(out, open(os.path.join(HERE, '..', 'results', 'berry_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
