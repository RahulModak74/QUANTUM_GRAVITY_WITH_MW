#!/usr/bin/env python3
"""
nu > 1 falsification test (see ../results/nu_test_PREREGISTRATION.md).

Retrains the v5 physics-prior VAE with the Student-t measurement prior at nu in {0.8, 1.5, 3.0}
and training seeds {0, 1}, on the run-A data (train seed 1, test seed 2), and scores the GR
sector and the Planck/topology structure. Writes ../results/nu_test.json (updated after every
run, so a partial grid is never lost) and ../results/nu_test_log.txt.
Usage: python3 mw_nu_falsification.py [nu:seed ...]     (default: the full grid)
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.join(HERE, '..', '..', 'Nov-2026_tests', 'code'); sys.path.insert(0, V5)
from qg_toy_vae_trainer_v5 import train                                          # noqa: E402
from qg_toy_data_generator_v5 import QGJointGenerator, QGParams, SCENARIOS       # noqa: E402

OUT = os.path.join(HERE, '..', 'results')
CK = os.path.join(OUT, 'nu_checkpoints'); os.makedirs(CK, exist_ok=True)
JS = os.path.join(OUT, 'nu_test.json')
logf = open(os.path.join(OUT, 'nu_test_log.txt'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()


def gen(seed):
    return QGJointGenerator(QGParams(seed=seed)).generate()


def einstein_rel(d, ci):
    rhs = 8 * np.pi * (-d[:, ci['T_00']] + d[:, ci['T_11']] + d[:, ci['T_22']] + d[:, ci['T_33']])
    return float(np.mean(np.abs(d[:, ci['R']] - rhs)) / np.mean(np.abs(rhs)))


def hawking(d, ci):
    M, TH = d[:, ci['M_BH']], d[:, ci['T_H']]
    h = (M > 0.1) & (TH > 1e-6)
    lx, ly = np.log(M[h]), np.log(TH[h]); s, b = np.polyfit(lx, ly, 1)
    return {'slope': float(s), 'r2': float(1 - np.var(ly - (s * lx + b)) / np.var(ly))}


def knn_recall(Z, lab, target, k=10, n=3000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(len(Z), n, replace=False)
    Zs, ls = torch.tensor(Z[idx]), lab[idx]
    D = torch.cdist(Zs, Zs); D.fill_diagonal_(float('inf'))
    nb = D.topk(k, largest=False).indices.numpy()
    pred = np.array([np.bincount(ls[r], minlength=6).argmax() for r in nb])
    m = ls == target
    return float((pred[m] == target).mean()), float((pred == ls).mean())


def score(vae, te, lab, ci):
    with torch.no_grad():
        x = torch.tensor(te, device=vae.mean.device)
        mu, _ = vae.encode((x - vae.mean) / vae.std)
        rt = vae.denormalize(vae.decode(mu)).cpu().numpy()
    fl = [ci['topology'], ci['disc_flag']]
    acc = float(((rt[:, fl] > 0.5) == (te[:, fl] > 0.5)).mean())
    mae = float(np.abs(rt[:, fl] - te[:, fl]).mean())
    wr, allacc = knn_recall(mu.cpu().numpy(), lab, SCENARIOS.index('wormhole'))
    return {'einstein_rel': einstein_rel(rt, ci), 'hawking': hawking(rt, ci),
            'P1_flag_accuracy': acc, 'P2_flag_mae': mae, 'P3_wormhole_knn_recall': wr,
            'knn_scenario_accuracy': allacc}


def main():
    grid = [tuple(a.split(':')) for a in sys.argv[1:]] or [(n, s) for n in ('0.8', '1.5', '3.0') for s in ('0', '1')]
    res = json.load(open(JS)) if os.path.exists(JS) else {}
    tr, _, cols = gen(1); te, lab, _ = gen(2)
    ci = {c: i for i, c in enumerate(cols)}
    for nu, seed in grid:
        key = f'nu{nu}_seed{seed}'
        if key in res:
            log(f'skip {key} (done)'); continue
        log(f'\n=== {key} ===')
        t0 = time.time()
        vae = train(tr, cols, use_priors=True, seed=int(seed), nu=float(nu), log=log)
        torch.save(vae.state_dict(), os.path.join(CK, f'{key}.pt'))
        r = score(vae, te, lab, ci); r['train_minutes'] = round((time.time() - t0) / 60, 1)
        res[key] = r
        json.dump(res, open(JS, 'w'), indent=1)
        log(json.dumps(r))
    log('DONE')


if __name__ == '__main__':
    main()
