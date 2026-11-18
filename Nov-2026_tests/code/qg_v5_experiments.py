#!/usr/bin/env python3
"""
V5 emergence experiments
========================
Runs (same seed, same architecture, same epochs):
  A  full_prior     all 30 columns, priors C1-C6
  B  full_noprior   all 30 columns, plain VAE
  C  heldout_TH     T_H column removed from data (no prior ever used T_H)
  D  shuffled       each column permuted within its scenario (marginals kept, coupling destroyed)

Measures, on fresh test data from the same generator (different seed):
  1. Einstein relative residual on reconstructions
  2. Hawking T_H vs M on reconstructions (log-log slope; 8*pi*M*T_H - 1)
  3. Out-of-distribution: horizons with M in [2, 4] (training saw [0.5, 2])
  4. Latent geometry vs M on horizon rows: does posterior variance or decoder
     volume scale like 1/M (slope -1) when T_H is never shown? (run C)
Writes ../results/results.json and ../results/log.txt
"""
import json, os, sys, time
import numpy as np
import torch
from qg_toy_data_generator_v5 import QGParams, QGJointGenerator
from qg_toy_vae_trainer_v5 import train

EPOCHS = int(os.environ.get('EPOCHS', 150))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)
logf = open(f'{OUT}/log.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); logf.write(s + '\n'); logf.flush()


def gen(seed, shuffle=False, drop=(), mass=(0.5, 2.0), scen=None, n=30000):
    p = QGParams(seed=seed, mass_range=mass, n_samples=n)
    if scen: p.scenarios = scen
    return QGJointGenerator(p).generate(shuffle=shuffle, drop_cols=drop)


@torch.no_grad()
def recon(vae, data):
    x = torch.tensor(np.nan_to_num(data), dtype=torch.float32, device=vae.mean.device)
    mu, logvar = vae.encode((x - vae.mean) / vae.std)
    return vae.denormalize(vae.decode(mu)).cpu().numpy(), mu, logvar


def einstein_rel(d, ci):
    rhs = 8 * np.pi * (-d[:, ci['T_00']] + d[:, ci['T_11']] + d[:, ci['T_22']] + d[:, ci['T_33']])
    return float(np.mean(np.abs(d[:, ci['R']] - rhs)) / np.mean(np.abs(rhs)))


def loglog(x, y):
    m = (x > 1e-6) & (y > 1e-6)
    if m.sum() < 10: return {'slope': None, 'r2': None, 'n': int(m.sum())}
    lx, ly = np.log(x[m]), np.log(y[m])
    s, b = np.polyfit(lx, ly, 1)
    r2 = 1 - np.var(ly - (s * lx + b)) / np.var(ly)
    return {'slope': round(float(s), 3), 'r2': round(float(r2), 3), 'n': int(m.sum())}


def hawking(d, ci):
    M, TH = d[:, ci['M_BH']], d[:, ci['T_H']]
    h = M > 0.1
    return {**loglog(M[h], TH[h]), 'mean_|8piMT_H-1|': round(float(np.mean(np.abs(8 * np.pi * M[h] * TH[h] - 1))), 3)}


def latent_geometry(vae, data, ci, true_M):
    """Per horizon row: encoder posterior variance and decoder volume log det(J^T J)."""
    h = true_M > 0.1
    rows = np.where(h)[0][:800]
    _, mu, logvar = recon(vae, data[rows])
    var = logvar.exp().mean(1).cpu().numpy()
    vols = []
    for z in mu:
        J = torch.autograd.functional.jacobian(vae.decode, z.unsqueeze(0)).squeeze()
        vols.append(torch.logdet(J.T @ J + 1e-6 * torch.eye(J.shape[1], device=J.device)).item())
    vol = np.exp(np.array(vols) / mu.shape[1])  # geometric-mean stretch per latent dim
    M = true_M[rows]
    return {'posterior_var_vs_M': loglog(M, var), 'decoder_stretch_vs_M': loglog(M, vol)}


RUNS = {
    'A_full_prior':   dict(shuffle=False, drop=(), priors=True),
    'B_full_noprior': dict(shuffle=False, drop=(), priors=False),
    'C_heldout_TH':   dict(shuffle=False, drop=('T_H',), priors=True),
    'D_shuffled':     dict(shuffle=True, drop=(), priors=True),
}

results = {'epochs': EPOCHS}
only = sys.argv[1:] or list(RUNS)
for name in only:
    cfg = RUNS[name]
    log(f'\n=== {name} ===')
    tr, _, cols = gen(seed=1, shuffle=cfg['shuffle'], drop=cfg['drop'])
    te, _, _ = gen(seed=2, shuffle=cfg['shuffle'], drop=cfg['drop'])
    ood, _, _ = gen(seed=3, drop=cfg['drop'], mass=(2.0, 4.0), scen=['black_hole', 'hawking'], n=4000)
    ci = {c: i for i, c in enumerate(cols)}
    t0 = time.time()
    vae = train(tr, cols, use_priors=cfg['priors'], epochs=EPOCHS, log=log)
    r = {'train_minutes': round((time.time() - t0) / 60, 1)}

    rt, _, _ = recon(vae, te)
    r['data_einstein_rel'] = einstein_rel(te, ci)
    r['recon_einstein_rel'] = einstein_rel(rt, ci)
    if 'T_H' in ci:
        r['data_hawking'] = hawking(te, ci)
        r['recon_hawking'] = hawking(rt, ci)
    ro, _, _ = recon(vae, ood)
    r['ood_recon_einstein_rel'] = einstein_rel(ro, ci)
    r['ood_M_rel_err'] = float(np.mean(np.abs(ro[:, ci['M_BH']] - ood[:, ci['M_BH']]) / ood[:, ci['M_BH']]))
    if 'T_H' in ci:
        r['ood_recon_hawking'] = hawking(ro, ci)
    if not cfg['shuffle']:
        r['latent_geometry'] = latent_geometry(vae, te, ci, te[:, ci['M_BH']])
    results[name] = r
    log(json.dumps(r, indent=2))
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=2)
log('\nDONE')
