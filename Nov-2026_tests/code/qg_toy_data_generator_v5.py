#!/usr/bin/env python3
"""
Quantum Gravity toy data - V5 (single joint distribution per row)
=================================================================

Fixes over v4:
  * Each row is ONE draw from ONE joint distribution: the physical parameters
    (scenario, coordinates, mass M, density, ...) are sampled once, and every
    column of that row is derived from those same draws. In v4 the black-hole
    mass was drawn twice (once for T_mu_nu, once for the M_BH / r_s / T_H columns).
  * Hawking scenario now carries the mass of the evaporating hole, so its
    radiation density (T_H^4) and its M_BH column describe the same object.
  * Spacetime-uncertainty column stores the product dx*dt (>= l_p^2) instead of
    |dx*dt_min - l_p^2|, which was identically ~0.
  * Bekenstein-Hawking entropy S = A/4G = 4*pi*G*M^2 for horizons; 0.1|R| elsewhere.
  * Columns are addressed by NAME (COLS) so trainer indices cannot drift again.

Options (used by the experiments):
  mass_range      - black-hole / hawking mass interval (for out-of-distribution tests)
  drop_cols       - remove columns entirely (held-out tests)
  shuffle         - permute every column independently within each scenario
                    (destroys within-row coupling, keeps each column's values)
"""

import numpy as np
from dataclasses import dataclass, field

COLS = [
    't', 'x', 'y', 'z',
    'g_00', 'g_11', 'g_22', 'g_33',
    'dg_01', 'dg_02', 'dg_03', 'dg_12', 'dg_13', 'dg_23', 'dg_11', 'dg_22',
    'psi_re', 'psi_im',
    'R',
    'T_00', 'T_11', 'T_22', 'T_33',
    'M_BH', 'r_s', 'T_H',
    'topology', 'dxdt', 'S_ent', 'disc_flag',
]
SCENARIOS = ['dust', 'radiation', 'vacuum', 'black_hole', 'wormhole', 'hawking']


@dataclass
class QGParams:
    hbar: float = 1.0
    c: float = 1.0
    G: float = 1.0
    l_planck: float = 1.0
    n_samples: int = 30000
    x_range: float = 10.0
    metric_fluctuation: float = 0.02
    mass_range: tuple = (0.5, 2.0)
    scenarios: list = field(default_factory=lambda: list(SCENARIOS))
    seed: int = 0


class QGJointGenerator:
    def __init__(self, p: QGParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

    def sample_row(self, scenario):
        p, rng = self.p, self.rng
        coords = np.array([rng.uniform(0, p.x_range),
                           *rng.uniform(-p.x_range, p.x_range, 3)])
        r = np.linalg.norm(coords[1:])

        # ---- the single set of physical draws for this row ----
        M = rng.uniform(*p.mass_range) if scenario in ('black_hole', 'hawking') else 0.0
        T = np.zeros((4, 4))
        if scenario == 'dust':
            T[0, 0] = 0.1 * np.exp(-0.5 * (r / 5) ** 2)
        elif scenario == 'radiation':
            rho = 0.08 * np.exp(-0.3 * (r / 5) ** 2)
            T[0, 0], T[1, 1], T[2, 2], T[3, 3] = rho, rho / 3, rho / 3, rho / 3
        elif scenario == 'vacuum':
            L = 0.01
            T[0, 0], T[1, 1], T[2, 2], T[3, 3] = -L, L, L, L
        elif scenario == 'black_hole':
            rho = M / (r ** 3 + 0.5)
            T[0, 0], T[1, 1], T[2, 2], T[3, 3] = rho, 0.1 * rho, 0.1 * rho, 0.1 * rho
        elif scenario == 'wormhole':
            rho = -abs(rng.standard_normal()) * 0.2
            T[0, 0], T[1, 1], T[2, 2], T[3, 3] = rho, -rho, -0.5 * rho, -0.5 * rho
        else:  # hawking: thermal radiation from a hole of mass M
            T_H_ = p.hbar * p.c ** 3 / (8 * np.pi * p.G * M)
            rho = T_H_ ** 4
            T[0, 0], T[1, 1], T[2, 2], T[3, 3] = rho, rho / 3, rho / 3, rho / 3

        # ---- GR block, derived from T ----
        T_tr = -T[0, 0] + T[1, 1] + T[2, 2] + T[3, 3]
        R = 8 * np.pi * p.G * T_tr
        h = np.diag([-0.1 * 8 * np.pi * p.G * T[0, 0],
                     *(0.1 * 8 * np.pi * p.G * T[i, i] for i in (1, 2, 3))])
        dg = rng.standard_normal((4, 4)) * p.metric_fluctuation
        dg = (dg + dg.T) / 2
        g = np.diag([-1.0, 1, 1, 1]) + h + dg

        # ---- QM block on that metric ----
        sqrt_det = np.sqrt(max(abs(np.linalg.det(g)), 1e-10))
        psi = np.exp(-0.5 * r ** 2) * np.exp(1j * r) / sqrt_det

        # ---- horizon quantities from the SAME M ----
        if M > 0:
            r_s = 2 * p.G * M / p.c ** 2
            T_H = p.hbar * p.c ** 3 / (8 * np.pi * p.G * M)
            S = 4 * np.pi * p.G * M ** 2          # A/4G
        else:
            r_s, T_H, S = 0.0, 0.0, abs(R) * 0.1

        dx = max(abs(coords[1]), 0.1)
        dt = p.l_planck ** 2 / dx * (1 + rng.exponential(0.5))
        row = [*coords, *np.diag(g),
               dg[0, 1], dg[0, 2], dg[0, 3], dg[1, 2], dg[1, 3], dg[2, 3], dg[1, 1], dg[2, 2],
               psi.real, psi.imag, R, *np.diag(T),
               M, r_s, T_H,
               1.0 if scenario == 'wormhole' else 0.0,
               dx * dt, S,
               1.0 if scenario in ('black_hole', 'wormhole', 'hawking') else 0.0]
        return row

    def generate(self, shuffle=False, drop_cols=()):
        n_per = self.p.n_samples // len(self.p.scenarios)
        blocks, labels = [], []
        for k, sc in enumerate(self.p.scenarios):
            b = np.array([self.sample_row(sc) for _ in range(n_per)])
            if shuffle:
                for j in range(b.shape[1]):
                    b[:, j] = self.rng.permutation(b[:, j])
            blocks.append(b)
            labels += [SCENARIOS.index(sc)] * n_per
        data = np.vstack(blocks).astype(np.float64)
        keep = [c for c in COLS if c not in drop_cols]
        data = data[:, [COLS.index(c) for c in keep]]
        return data.astype(np.float32), np.array(labels), keep


if __name__ == '__main__':
    d, lab, cols = QGJointGenerator(QGParams()).generate()
    i = {c: k for k, c in enumerate(cols)}
    viol = np.abs(d[:, i['R']] - 8 * np.pi * (-d[:, i['T_00']] + d[:, i['T_11']] + d[:, i['T_22']] + d[:, i['T_33']]))
    bh = d[:, i['M_BH']] > 0
    print('shape', d.shape, '| Einstein max violation', viol.max(),
          '| 8*pi*M*T_H on horizons', np.abs(8 * np.pi * d[bh, i['M_BH']] * d[bh, i['T_H']] - 1).max())
    np.save('qg_toy_data_v5.npy', d)
