#!/usr/bin/env python3
"""
Kerr v2 step 1+2: exact geodesic data, then the MW VAE.

Data: timelike Kerr geodesics integrated from Hamilton's equations
   H = 1/2 g^{mu nu} p_mu p_nu = -1/2,  dx/dlam = dH/dp,  dp/dlam = -dH/dx
(batched RK4 in float64, dlam = 0.02 M). Initial conditions perturb circular
equatorial orbits (inclined, eccentric, some plunging); points with r < 1.15 r_+
or r > 40 M are dropped. |H + 1/2| is reported as the integration check.
The v1 generator wrote circles with random radial jitter and linear-in-time theta
tracks, which are not geodesics.

Model: Pyro VAE (8-D latent) on x = (r, theta, cos phi, sin phi), fitted internally on
(log r, theta, cos phi, sin phi) for uniform relative precision in r, with
  * Gaussian likelihood on standardised features (sigma 3e-3),
  * N(0, I) latent prior weighted beta = 1e-3 (beta-VAE), and
  * MW physics prior: the Kerr metric at the reconstructed point must match the
    metric at the data point (relative Gaussian, sigma 1e-2), i.e. the decoder is
    asked to preserve the geometry, not only the coordinates.
(v1 computed g_true and g_recon in its model but never used them.)

Usage: python3 kerr_v2_data_train.py [--steps 40000]
Writes data/kerr_geodesics_v2.npz, model/kerr_mw_v2.pt, results/train_log.txt
"""
import argparse, os, time, json
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from kerr_v2_common import A_SPIN, R_PLUS, inverse_metric, metric, omega_exact, energy_exact, KerrMWVAE

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('data', 'model', 'results'):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
torch.set_default_dtype(torch.float64)
LOG = open(os.path.join(HERE, 'results', 'train_log.txt'), 'w')


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def hamiltonian(q, p):
    gtt, gtp, gpp, grr, gthth = inverse_metric(q[:, 1], q[:, 2])
    return 0.5 * (gtt * p[:, 0] ** 2 + 2 * gtp * p[:, 0] * p[:, 3] + gpp * p[:, 3] ** 2
                  + grr * p[:, 1] ** 2 + gthth * p[:, 2] ** 2)


def rhs(q, p):
    q = q.detach().requires_grad_(True); p = p.detach().requires_grad_(True)
    H = hamiltonian(q, p).sum()
    dq, dp = torch.autograd.grad(H, (p, q))
    return dq, -dp


def initial_conditions(n, rng):
    r0 = rng.uniform(2.4, 25.0, n)
    th0 = np.pi / 2 + rng.uniform(-1.1, 1.1, n)
    ph0 = rng.uniform(0, 2 * np.pi, n)
    E = energy_exact(r0) * rng.uniform(0.995, 1.01, n)
    L = (r0 ** 2 - 2 * A_SPIN * np.sqrt(r0) + A_SPIN ** 2) / (r0 ** 0.75 * np.sqrt(r0 ** 1.5 - 3 * np.sqrt(r0) + 2 * A_SPIN))
    L = L * rng.uniform(0.75, 1.05, n)
    q = torch.tensor(np.stack([np.zeros(n), r0, th0, ph0], 1))
    pr = torch.tensor(rng.normal(0, 0.05, n)); pth = torch.tensor(rng.normal(0, 0.5, n)) * torch.tensor(np.sqrt(r0))
    pphi = torch.tensor(L)
    # solve H = -1/2 for p_t (future-directed: p_t < 0)
    gtt, gtp, gpp, grr, gthth = inverse_metric(q[:, 1], q[:, 2])
    A_, B_ = gtt, 2 * gtp * pphi
    C_ = gpp * pphi ** 2 + grr * pr ** 2 + gthth * pth ** 2 + 1.0
    disc = B_ ** 2 - 4 * A_ * C_
    pt = (-B_ + torch.sqrt(disc.clamp_min(0))) / (2 * A_)       # A_<0 outside horizon -> this root is negative
    p = torch.stack([pt, pr, pth, pphi], 1)
    return q, p


def make_data(n_geo=3000, n_steps=6000, dlam=0.02, every=25, seed=42):
    rng = np.random.default_rng(seed)
    q, p = initial_conditions(n_geo, rng)
    alive = torch.ones(n_geo, dtype=torch.bool)
    H0 = hamiltonian(q, p).detach()
    pts, herr = [], []
    t0 = time.time()
    for s in range(n_steps):
        k1q, k1p = rhs(q, p)
        k2q, k2p = rhs(q + 0.5 * dlam * k1q, p + 0.5 * dlam * k1p)
        k3q, k3p = rhs(q + 0.5 * dlam * k2q, p + 0.5 * dlam * k2p)
        k4q, k4p = rhs(q + dlam * k3q, p + dlam * k3p)
        q = (q + dlam / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)).detach()
        p = (p + dlam / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)).detach()
        alive &= (q[:, 1] > 1.15 * R_PLUS) & (q[:, 1] < 40) & torch.isfinite(q).all(1)
        if s % every == 0 and s > 0:
            m = alive
            pts.append(torch.cat([q[m], p[m]], 1).numpy())
            herr.append((hamiltonian(q[m], p[m]) - H0[m]).abs().numpy())
        if s % 1000 == 0:
            log(f'  geodesic step {s}/{n_steps}  alive {int(alive.sum())}  ({time.time() - t0:.0f}s)')
    P = np.concatenate(pts); herr = np.concatenate(herr)
    log(f'  points {len(P)};  |H + 1/2| max {herr.max():.2e}, median {np.median(herr):.2e}')
    return P, herr


def train(X, steps, seed=0, latent_dim=8, batch=1024, beta=1e-3, g_sigma=1e-2, lr=1e-3, obs_sigma=3e-3):
    torch.manual_seed(seed); pyro.set_rng_seed(seed); pyro.clear_param_store()
    Xt = torch.tensor(X)
    F = KerrMWVAE.feat(Xt)
    mean, std = F.mean(0), F.std(0)
    vae = KerrMWVAE(latent_dim, mean, std).double()

    def gfeat(x):
        gtt, gtp, gpp, grr, gthth = metric(x[:, 0], x[:, 1])
        return torch.stack([gtt, gtp, gpp, torch.log(grr), torch.log(gthth)], 1)

    def model(x):
        pyro.module('vae', vae)
        with pyro.plate('data', x.shape[0]):
            with pyro.poutine.scale(scale=beta):                     # beta-VAE weight on the latent prior
                z = pyro.sample('z', dist.Normal(torch.zeros(x.shape[0], latent_dim), 1.).to_event(1))
            xr = vae.decode(z)
            pyro.sample('obs', dist.Normal((vae.feat(xr) - mean) / std, obs_sigma).to_event(1), obs=(vae.feat(x) - mean) / std)
            g_true, g_rec = gfeat(x), gfeat(xr)
            scale = g_sigma * (g_true.abs() + 1e-2)
            pyro.sample('metric_prior', dist.Normal(g_rec, scale).to_event(1), obs=g_true)   # MW physics prior

    def guide(x):
        pyro.module('vae', vae)
        with pyro.plate('data', x.shape[0]):
            mu, lv = vae.encode(x)
            with pyro.poutine.scale(scale=beta):
                pyro.sample('z', dist.Normal(mu, torch.exp(0.5 * lv)).to_event(1))

    sched = pyro.optim.ClippedAdam({'lr': lr, 'clip_norm': 10.0, 'lrd': (1e-2) ** (1 / steps)})
    svi = SVI(model, guide, sched, loss=Trace_ELBO())
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, len(Xt), (batch,))
        loss = svi.step(Xt[idx])
        if s % 2000 == 0 or s == steps - 1:
            with torch.no_grad():
                xr = vae.roundtrip(Xt[:4000])
                er = (xr[:, 0] - Xt[:4000, 0]).abs()
            log(f'  step {s:6d}  loss/pt {loss / batch:10.3f}  |dr| median {er.median():.2e} p95 {er.quantile(.95):.2e}  ({time.time() - t0:.0f}s)')
    return vae.eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--beta', type=float, default=1e-3); ap.add_argument('--g-sigma', type=float, default=1e-2)
    a = ap.parse_args()
    torch.set_num_threads(6)
    dpath = os.path.join(HERE, 'data', 'kerr_geodesics_v2.npz')
    if not os.path.exists(dpath):
        log('Integrating exact Kerr geodesics (a = 0.9) ...')
        P, herr = make_data()
        np.savez_compressed(dpath, qp=P, h_err=herr)
    P = np.load(dpath)['qp']
    X = np.stack([P[:, 1], P[:, 2], np.cos(P[:, 3]), np.sin(P[:, 3])], 1)
    rng = np.random.default_rng(0); perm = rng.permutation(len(X))
    n_te = min(20000, len(X) // 10)
    Xte, Xtr = X[perm[:n_te]], X[perm[n_te:]]
    log(f'train {len(Xtr)}, test {len(Xte)};  r in [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}], '
        f'theta in [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]')
    vae = train(Xtr, a.steps, beta=a.beta, g_sigma=a.g_sigma)
    with torch.no_grad():
        xr = vae.roundtrip(torch.tensor(Xte))
    Xte_t = torch.tensor(Xte)
    dr = (xr[:, 0] - Xte_t[:, 0]).abs().numpy(); dth = (xr[:, 1] - Xte_t[:, 1]).abs().numpy()
    dphi = np.abs(np.angle((xr[:, 2] + 1j * xr[:, 3]).numpy() * np.conj(Xte[:, 2] + 1j * Xte[:, 3])))
    stats = {k: {'median': float(np.median(v)), 'p95': float(np.quantile(v, .95)), 'max': float(v.max())}
             for k, v in [('abs_dr', dr), ('abs_dtheta', dth), ('abs_dphi', dphi)]}
    log('held-out round-trip errors: ' + json.dumps(stats))
    torch.save({'state_dict': vae.state_dict(), 'latent_dim': vae.latent_dim, 'mean': vae.mean, 'std': vae.std},
               os.path.join(HERE, 'model', 'kerr_mw_v2.pt'))
    json.dump({'roundtrip_test': stats, 'n_train': len(Xtr), 'n_test': len(Xte), 'steps': a.steps},
              open(os.path.join(HERE, 'results', 'train_summary.json'), 'w'), indent=1)
    log('saved model/kerr_mw_v2.pt')


if __name__ == '__main__':
    main()
