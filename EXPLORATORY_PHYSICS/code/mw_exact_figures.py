#!/usr/bin/env python3
"""Figures for EXPLORATORY_PHYSICS (reads ../results/exact_geometry*.json/npz)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
PAP = os.path.join(HERE, '..', 'papers'); os.makedirs(PAP, exist_ok=True)
SC = ['dust', 'radiation', 'vacuum', 'black_hole', 'wormhole', 'hawking']
GR = ['coordinates', 'gravity', 'quantum', 'horizon', 'planck_topology']
LAB = {'coordinates': 'coordinates (<1%)', 'gravity': 'gravity (g, R, T)', 'quantum': 'quantum (ψ, Δx·Δt)',
       'horizon': 'horizon (M, r_s, T_H, S)', 'planck_topology': 'Planck/topology'}
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # categorical slots 1-5, fixed order
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3de'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6})

js = json.load(open(os.path.join(RES, 'exact_geometry.json')))
pts = np.load(os.path.join(RES, 'exact_geometry_points.npz'), allow_pickle=True)
A = 'A_full_prior'

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))

# (a) the field equation holds: Gauss and Bianchi residuals vs the legacy proxy scale
a = ax[0, 0]
data = [pts[f'{A}__gauss'], pts[f'{A}__bian'], np.abs(pts[f'{A}__legacy'])]
names = ['Gauss:\nRiem(Γ) vs Riem(II)', 'Bianchi:\n∇ᵃG_ab = 0', 'legacy "R" proxy\n(not a curvature)']
for i, (dd, c) in enumerate(zip(data, [COL[0], COL[0], MUTED])):
    y = np.log10(dd)
    a.scatter(np.full_like(y, i) + np.random.default_rng(i).uniform(-.18, .18, len(y)), y, s=6, color=c, alpha=.5, lw=0)
    a.plot([i - .25, i + .25], [np.median(y)] * 2, color=INK, lw=2)
a.set_xticks(range(3)); a.set_xticklabels(names, fontsize=7.5)
a.set_ylabel('log₁₀ value'); a.set_title('(a) Einstein-form identity holds to round-off', fontsize=9, loc='left')

# (b) Ricci scalar per regime
b = ax[0, 1]
R, lab = pts[f'{A}__R'], pts[f'{A}__lab']
for k, sc in enumerate(SC):
    y = R[lab == k]
    b.scatter(np.full_like(y, k) + np.random.default_rng(k).uniform(-.2, .2, len(y)), y, s=6, color=COL[0], alpha=.5, lw=0)
    b.plot([k - .28, k + .28], [np.median(y)] * 2, color=INK, lw=2)
b.set_yscale('symlog', linthresh=10); b.set_xticks(range(6))
b.set_xticklabels([s.replace('_', ' ') for s in SC], rotation=25, fontsize=7.5)
b.set_ylabel('Ricci scalar R (exact)'); b.set_title('(b) Curvature by regime (240 test states)', fontsize=9, loc='left')

# (c) source shares: which columns curve the manifold
c = ax[1, 0]
share = pts[f'{A}__share']
left = np.zeros(6)
for j, g in enumerate(GR):
    v = np.array([share[lab == k, j].mean() for k in range(6)])
    c.barh(range(6), v, left=left, color=COL[j], height=.62, edgecolor='white', linewidth=1.5, label=LAB[g])
    left += v
c.set_yticks(range(6)); c.set_yticklabels([s.replace('_', ' ') for s in SC], fontsize=7.5); c.invert_yaxis()
c.set_xlim(0, 1); c.set_xlabel('share of |R| contributed by each column group')
c.set_title('(c) One equation, additive sources: G = κ ΣT⁽ˢ⁾', fontsize=9, loc='left'); c.grid(axis='y', visible=False)
c.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

# (d) geodesic between a light and a heavy black hole
dax = ax[1, 1]
gpath = np.load(os.path.join(RES, f'geodesic_path_{A}.npz'))
ci = {'M_BH': 23, 'T_H': 25}
for key, lbl, col, ls in [('geo', 'geodesic', COL[0], '-'), ('line', 'straight latent line', MUTED, '--')]:
    x = gpath[key]; s = np.linspace(0, 1, len(x))
    dax.plot(s, x[:, ci['M_BH']], color=col, lw=2, ls=ls, label=f'M along {lbl}')
dax.set_xlabel('arc parameter s'); dax.set_ylabel('decoded mass M')
g = js[A]['geodesic']
dax.set_title('(d) Geodesic: black hole M≈0.6 → 1.9', fontsize=9, loc='left')
dax.text(0.03, 0.97, f'acceleration normal to manifold:\n geodesic {100 * (1 - g["tangential_accel_fraction_geodesic"]["median"]):.0f}%'
         f'  vs line {100 * (1 - g["tangential_accel_fraction_line"]["median"]):.0f}%\n'
         f'|8πM·T_H − 1| on geodesic: {g["geodesic"]["hawking_abs_8piMTH_minus_1_mean"]:.2f}',
         transform=dax.transAxes, va='top', fontsize=7, color=INK)
dax.legend(fontsize=7, loc='lower right', frameon=False)

fig.tight_layout()
for p in [os.path.join(RES, 'fig_exploratory.png'), os.path.join(PAP, 'fig_exploratory.pdf')]:
    fig.savefig(p, dpi=200)
print('wrote fig_exploratory.png / papers/fig_exploratory.pdf')
