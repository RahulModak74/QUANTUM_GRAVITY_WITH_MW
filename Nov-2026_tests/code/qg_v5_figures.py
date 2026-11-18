#!/usr/bin/env python3
"""
Figures for the Nov-2026 revision.
Trains runs A (MW priors) and B (plain VAE) with the same seed/epochs
as qg_v5_experiments.py, saves checkpoints, and plots ../results/fig_v5.png (+ .pdf
copy in ../paper/). Numbers are recomputed from the saved models; with the fixed seed
they reproduce results.json.
"""
import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qg_toy_data_generator_v5 import QGParams, QGJointGenerator, SCENARIOS
from qg_toy_vae_trainer_v5 import train, QGPhysicsVAE

HERE = os.path.dirname(os.path.abspath(__file__))
RES, PAPER = os.path.join(HERE, '..', 'results'), os.path.join(HERE, '..', 'paper')
CK = os.path.join(RES, 'checkpoints'); os.makedirs(CK, exist_ok=True)
EPOCHS = int(os.environ.get('EPOCHS', 150))
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']  # validated categorical order
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'


def gen(seed, drop=(), mass=(0.5, 2.0), scen=None, n=30000):
    p = QGParams(seed=seed, mass_range=mass, n_samples=n)
    if scen: p.scenarios = scen
    return QGJointGenerator(p).generate(drop_cols=drop)


def get_model(name, drop, priors):
    tr, _, cols = gen(1, drop)
    path = os.path.join(CK, f'{name}.pt')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(path):
        X = torch.tensor(tr); m, s = X.mean(0), X.std(0) + 1e-8
        vae = QGPhysicsVAE(len(cols), 20, m.to(dev), s.to(dev)).to(dev)
        vae.load_state_dict(torch.load(path, map_location=dev)); vae.eval()
    else:
        print(f'training {name}')
        vae = train(tr, cols, use_priors=priors, epochs=EPOCHS)
        torch.save(vae.state_dict(), path)
    return vae, {c: i for i, c in enumerate(cols)}


@torch.no_grad()
def enc_dec(vae, d):
    x = torch.tensor(d, device=vae.mean.device)
    mu, _ = vae.encode((x - vae.mean) / vae.std)
    return vae.denormalize(vae.decode(mu)).cpu().numpy(), mu


def style(ax, title, xl, yl):
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
    ax.set_xlabel(xl, color=MUTED, fontsize=9); ax.set_ylabel(yl, color=MUTED, fontsize=9)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


A, ciA = get_model('A_full_prior', (), True)
B, ciB = get_model('B_full_noprior', (), False)
te, lab, _ = gen(2)
rA, muA = enc_dec(A, te); rB, _ = enc_dec(B, te)
ci = ciA
res = json.load(open(os.path.join(RES, 'results.json')))
fig, axd = plt.subplot_mosaic([['a', 'a', 'b', 'b', 'c', 'c'], ['d', 'd', 'd', 'e', 'e', 'e']],
                              figsize=(15, 9.2), facecolor='#fcfcfb')
plt.subplots_adjust(hspace=0.42, wspace=0.9)

# (a) latent space by scenario
ax = axd['a']
mu = muA.cpu().numpy(); mu = mu - mu.mean(0)
pc = mu @ np.linalg.svd(mu, full_matrices=False)[2][:2].T
idx = np.random.default_rng(0).permutation(len(pc))[:6000]
for k, sc in enumerate(SCENARIOS):
    m = idx[lab[idx] == k]
    ax.scatter(pc[m, 0], pc[m, 1], s=8, c=SERIES[k], edgecolors='#fcfcfb', linewidths=0.3, label=sc.replace('_', ' '))
ax.legend(fontsize=7.5, frameon=False, markerscale=1.6, loc='best')
style(ax, '(a) One latent space, six regimes', 'latent PC1', 'latent PC2')

# (b) Einstein on decoded states
ax = axd['b']
def tr_(d): return 8 * np.pi * (-d[:, ci['T_00']] + d[:, ci['T_11']] + d[:, ci['T_22']] + d[:, ci['T_33']])
rhs, R = tr_(rA), rA[:, ci['R']]
rel = np.mean(np.abs(R - rhs)) / np.mean(np.abs(rhs))
ax.scatter(rhs[idx], R[idx], s=8, c=SERIES[0], edgecolors='#fcfcfb', linewidths=0.3)
lim = [min(rhs.min(), R.min()), max(rhs.max(), R.max())]
ax.plot(lim, lim, color=MUTED, lw=1.2, ls='--', label='R = 8πG·T')
ax.text(0.04, 0.92, f'relative residual {100 * rel:.1f}%', transform=ax.transAxes, fontsize=9, color=INK)
ax.legend(fontsize=8, frameon=False, loc='lower right')
style(ax, '(b) GR: Einstein trace on decoded states', '8πG·T (decoded)', 'R (decoded)')

# (c) Hawking scaling, MW priors vs plain VAE
ax = axd['c']
def hw(r):
    h = (r[:, ci['M_BH']] > 0.1) & (r[:, ci['T_H']] > 1e-6)
    M, T = r[h, ci['M_BH']], r[h, ci['T_H']]
    s, b = np.polyfit(np.log(M), np.log(T), 1)
    r2 = 1 - np.var(np.log(T) - (s * np.log(M) + b)) / np.var(np.log(T))
    return M, T, s, r2
fig_nums = {'fig_einstein_rel': float(rel)}
for r, col, nm, key in ((rB, SERIES[1], 'plain VAE (B)', 'B'), (rA, SERIES[0], 'MW priors (A)', 'A')):
    M, T, s, r2 = hw(r); j = np.random.default_rng(1).permutation(len(M))[:2500]
    fig_nums[f'fig_hawking_{key}'] = {'slope': float(s), 'r2': float(r2)}
    ax.scatter(M[j], T[j], s=8, c=col, edgecolors='#fcfcfb', linewidths=0.3, label=f'{nm}: slope {s:.2f}, R² {r2:.2f}')
mm = np.linspace(0.5, 2.0, 50); ax.plot(mm, 1 / (8 * np.pi * mm), color=INK, lw=1.4, ls='--', label='T_H = 1/(8πM)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(fontsize=7.5, frameon=False, loc='lower left')
style(ax, '(c) Hawking scaling on decoded states', 'M (decoded)', 'T_H (decoded)')

RUNS3 = [('A_full_prior', 'A  MW priors', SERIES[0]), ('B_full_noprior', 'B  plain VAE', SERIES[1]),
         ('D_shuffled', 'D  rows shuffled', MUTED)]
def bars(ax, vals, fmt, off):
    bb = ax.bar([n for _, n, _ in RUNS3], vals, color=[c for _, _, c in RUNS3], width=0.55,
                edgecolor='#fcfcfb', linewidth=2)
    for b_, v in zip(bb, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + off, fmt.format(v), ha='center', fontsize=9, color=INK)

# (d) Hawking R2 across runs
ax = axd['d']
bars(ax, [res[k]['recon_hawking']['r2'] for k, _, _ in RUNS3], '{:.2f}', 0.02)
ax.set_ylim(0, 1)
style(ax, '(d) Hawking-relation fidelity: MW priors vs plain VAE vs shuffled', '', 'R²  (log T_H vs log M)'); ax.grid(False, axis='x')

# (e) Einstein residual across runs
ax = axd['e']
bars(ax, [100 * res[k]['recon_einstein_rel'] for k, _, _ in RUNS3], '{:.1f}%', 1.2)
ax.set_ylim(0, 70)
style(ax, '(e) Einstein residual: joint rows vs shuffled rows', '', 'relative residual (%)'); ax.grid(False, axis='x')

fig.savefig(os.path.join(RES, 'fig_v5.png'), dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
fig.savefig(os.path.join(PAPER, 'fig_v5.pdf'), bbox_inches='tight', facecolor=fig.get_facecolor())
json.dump(fig_nums, open(os.path.join(RES, 'figure_numbers.json'), 'w'), indent=2)
print('saved figures')
