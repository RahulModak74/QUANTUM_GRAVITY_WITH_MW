#!/usr/bin/env python3
"""
Kerr v3: one MW manifold for the whole Kerr family, spin a in [0, 0.95].

v2 learned one black hole (a = 0.9). v3 conditions the manifold on spin: encoder and decoder both receive a,
so a single trained model covers every spin in the range, including spins it never saw.

  data   exact timelike geodesics (Hamilton's equations, batched RK4, float64), each with its own a ~ U(0, 0.95)
  model  conditional Pyro VAE, 8-D latent, on x = (r, theta, cos phi, sin phi) given a; priors as in v2
         (Gaussian likelihood, beta-weighted N(0, I) latent prior, MW metric prior: g(x_hat, a) = g(x, a))
  eval   spins never used as training values (a = 0, 0.3, 0.5, 0.7, 0.9, 0.95 are measure-zero in U(0, 0.95)):
         ISCO(a), Omega(r; a), E(r; a) through the manifold vs Bardeen-Press-Teukolsky, and aLIGO-weighted
         inspiral matches at M = 60 Msun, eta = 0.01 (from ../v2_nov2026/kerr_v2_ligo.py)

Usage:
  python3 kerr_v3_spin.py data            # ~geodesics -> data/kerr_family_v3.npz
  python3 kerr_v3_spin.py train --steps 100000
  python3 kerr_v3_spin.py eval            # -> results/spin_family.json, results/fig_spin_family.png
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, '..', 'v2_nov2026'); sys.path.insert(0, V2)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
torch.set_default_dtype(torch.float64)
from kerr_v2_common import metric, inverse_metric, omega_exact, energy_exact, isco_exact, circular_from_metric  # noqa

for d in ('data', 'model', 'results'):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)
A_MAX = 0.95
DPATH = os.path.join(HERE, 'data', 'kerr_family_v3.npz')
MPATH = os.path.join(HERE, 'model', 'kerr_mw_v3_spin.pt')
LOG = None


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    if LOG: LOG.write(s + '\n'); LOG.flush()


def r_horizon(a): return 1 + np.sqrt(1 - a ** 2)
def r_photon(a): return 2 * (1 + np.cos(2 / 3 * np.arccos(-a)))       # prograde circular photon orbit


# ------------------------------------------------------------------ data
def hamiltonian(q, p, a):
    gtt, gtp, gpp, grr, gthth = inverse_metric(q[:, 1], q[:, 2], a)
    return 0.5 * (gtt * p[:, 0] ** 2 + 2 * gtp * p[:, 0] * p[:, 3] + gpp * p[:, 3] ** 2 + grr * p[:, 1] ** 2 + gthth * p[:, 2] ** 2)


def rhs(q, p, a):
    q = q.detach().requires_grad_(True); p = p.detach().requires_grad_(True)
    dq, dp = torch.autograd.grad(hamiltonian(q, p, a).sum(), (p, q))
    return dq, -dp


def make_data(n_geo=6000, n_steps=6000, dlam=0.02, every=25, seed=7):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, A_MAX, n_geo)
    r0 = rng.uniform(1.15 * r_photon(a), 25.0)
    r0 = np.maximum(r0, 1.25 * r_photon(a))
    th0 = np.pi / 2 + rng.uniform(-1.1, 1.1, n_geo); ph0 = rng.uniform(0, 2 * np.pi, n_geo)
    L = (r0 ** 2 - 2 * a * np.sqrt(r0) + a ** 2) / (r0 ** 0.75 * np.sqrt(r0 ** 1.5 - 3 * np.sqrt(r0) + 2 * a))
    L = L * rng.uniform(0.75, 1.05, n_geo)
    at = torch.tensor(a)
    q = torch.tensor(np.stack([np.zeros(n_geo), r0, th0, ph0], 1))
    pr = torch.tensor(rng.normal(0, 0.05, n_geo)); pth = torch.tensor(rng.normal(0, 0.5, n_geo) * np.sqrt(r0))
    pphi = torch.tensor(L)
    gtt, gtp, gpp, grr, gthth = inverse_metric(q[:, 1], q[:, 2], at)
    B = 2 * gtp * pphi; C = gpp * pphi ** 2 + grr * pr ** 2 + gthth * pth ** 2 + 1.0
    pt = (-B + torch.sqrt((B ** 2 - 4 * gtt * C).clamp_min(0))) / (2 * gtt)
    p = torch.stack([pt, pr, pth, pphi], 1)
    H0 = hamiltonian(q, p, at).detach()
    rcut = torch.tensor(1.15 * r_horizon(a))
    alive = torch.ones(n_geo, dtype=torch.bool)
    pts, herr = [], []
    t0 = time.time()
    for s in range(n_steps):
        k1q, k1p = rhs(q, p, at)
        k2q, k2p = rhs(q + 0.5 * dlam * k1q, p + 0.5 * dlam * k1p, at)
        k3q, k3p = rhs(q + 0.5 * dlam * k2q, p + 0.5 * dlam * k2p, at)
        k4q, k4p = rhs(q + dlam * k3q, p + dlam * k3p, at)
        q = (q + dlam / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)).detach()
        p = (p + dlam / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)).detach()
        alive &= (q[:, 1] > rcut) & (q[:, 1] < 40) & torch.isfinite(q).all(1) & torch.isfinite(p).all(1)
        if s % every == 0 and s > 0:
            m = alive
            pts.append(torch.cat([q[m], p[m], at[m, None]], 1).numpy())
            herr.append((hamiltonian(q[m], p[m], at[m]) - H0[m]).abs().numpy())
        if s % 1000 == 0:
            log(f'  geodesic step {s}/{n_steps}  alive {int(alive.sum())}  ({time.time() - t0:.0f}s)')
    P = np.concatenate(pts); herr = np.concatenate(herr)
    log(f'  points {len(P)};  |H + 1/2| max {herr.max():.2e}, median {np.median(herr):.2e};  data wall {time.time() - t0:.0f}s')
    np.savez_compressed(DPATH, qpa=P, h_err=herr, wall_s=time.time() - t0)


# ------------------------------------------------------------------ model
class KerrSpinVAE(nn.Module):
    """Conditional encoder/decoder over (r, theta, cos phi, sin phi) given spin a."""

    def __init__(self, latent_dim=8, mean=None, std=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('mean', torch.zeros(4) if mean is None else mean)
        self.register_buffer('std', torch.ones(4) if std is None else std)
        act = nn.Tanh
        self.enc = nn.Sequential(nn.Linear(5, 160), act(), nn.Linear(160, 160), act(), nn.Linear(160, 64), act())
        self.fc_mu, self.fc_lv = nn.Linear(64, latent_dim), nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim + 1, 64), act(), nn.Linear(64, 160), act(), nn.Linear(160, 160),
                                 act(), nn.Linear(160, 4))

    @staticmethod
    def feat(x): return torch.cat([torch.log(x[:, :1]), x[:, 1:]], 1)
    @staticmethod
    def ain(a): return ((a - A_MAX / 2) / (A_MAX / 2))[:, None]

    def encode(self, x, a):
        h = self.enc(torch.cat([(self.feat(x) - self.mean) / self.std, self.ain(a)], 1))
        return self.fc_mu(h), self.fc_lv(h).clamp(-14, 6)

    def decode(self, z, a):
        o = self.dec(torch.cat([z, self.ain(a)], 1))
        r_min = 1.05 * (1 + torch.sqrt(1 - a ** 2))
        r = torch.maximum(torch.exp(o[:, 0] * self.std[0] + self.mean[0]), r_min)
        th = torch.pi * torch.sigmoid(o[:, 1])
        cs = o[:, 2:4] / o[:, 2:4].norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.stack([r, th, cs[:, 0], cs[:, 1]], 1)

    def roundtrip(self, x, a): return self.decode(self.encode(x, a)[0], a)


def load_xa():
    P = np.load(DPATH)['qpa']
    X = np.stack([P[:, 1], P[:, 2], np.cos(P[:, 3]), np.sin(P[:, 3])], 1)
    return X, P[:, 8]


def train(steps, latent_dim=8, batch=1024, beta=1e-3, g_sigma=1e-2, lr=1e-3, obs_sigma=3e-3, seed=0):
    import pyro, pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    X, A = load_xa()
    rng = np.random.default_rng(0); perm = rng.permutation(len(X)); n_te = min(20000, len(X) // 10)
    Xt, At = torch.tensor(X[perm[n_te:]]), torch.tensor(A[perm[n_te:]])
    Xte, Ate = torch.tensor(X[perm[:n_te]]), torch.tensor(A[perm[:n_te]])
    log(f'train {len(Xt)}, test {n_te};  a in [{A.min():.3f}, {A.max():.3f}], r in [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]')
    torch.manual_seed(seed); pyro.set_rng_seed(seed); pyro.clear_param_store()
    F = KerrSpinVAE.feat(Xt); mean, std = F.mean(0), F.std(0)
    vae = KerrSpinVAE(latent_dim, mean, std).double()

    def gfeat(x, a):
        gtt, gtp, gpp, grr, gthth = metric(x[:, 0], x[:, 1], a)
        return torch.stack([gtt, gtp, gpp, torch.log(grr), torch.log(gthth)], 1)

    def model(x, a):
        pyro.module('vae', vae)
        with pyro.plate('data', x.shape[0]):
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample('z', dist.Normal(torch.zeros(x.shape[0], latent_dim), 1.).to_event(1))
            xr = vae.decode(z, a)
            pyro.sample('obs', dist.Normal((vae.feat(xr) - mean) / std, obs_sigma).to_event(1), obs=(vae.feat(x) - mean) / std)
            gt, gr = gfeat(x, a), gfeat(xr, a)
            pyro.sample('metric_prior', dist.Normal(gr, g_sigma * (gt.abs() + 1e-2)).to_event(1), obs=gt)

    def guide(x, a):
        pyro.module('vae', vae)
        with pyro.plate('data', x.shape[0]):
            mu, lv = vae.encode(x, a)
            with pyro.poutine.scale(scale=beta):
                pyro.sample('z', dist.Normal(mu, torch.exp(0.5 * lv)).to_event(1))

    opt = pyro.optim.ClippedAdam({'lr': lr, 'clip_norm': 10.0, 'lrd': (1e-2) ** (1 / steps)})
    svi = SVI(model, guide, opt, loss=Trace_ELBO())
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, len(Xt), (batch,))
        loss = svi.step(Xt[idx], At[idx])
        if s % 5000 == 0 or s == steps - 1:
            with torch.no_grad():
                er = (vae.roundtrip(Xte[:4000], Ate[:4000])[:, 0] - Xte[:4000, 0]).abs()
            log(f'  step {s:6d}  loss/pt {loss / batch:10.3f}  held-out |dr| median {er.median():.2e} p95 {er.quantile(.95):.2e}  ({time.time() - t0:.0f}s)')
    with torch.no_grad():
        xr = vae.roundtrip(Xte, Ate)
    dr = (xr[:, 0] - Xte[:, 0]).abs().numpy()
    torch.save({'state_dict': vae.state_dict(), 'latent_dim': latent_dim, 'mean': mean, 'std': std,
                'train_wall_s': time.time() - t0, 'steps': steps,
                'heldout_abs_dr': {'median': float(np.median(dr)), 'p95': float(np.quantile(dr, .95))}}, MPATH)
    log(f'saved {MPATH}; held-out |dr| median {np.median(dr):.2e}; train wall {time.time() - t0:.0f}s')


# ------------------------------------------------------------------ eval
def load_spin_model():
    ck = torch.load(MPATH, map_location='cpu')
    m = KerrSpinVAE(ck['latent_dim'], ck['mean'], ck['std']).double()
    m.load_state_dict(ck['state_dict']); return m.eval(), ck


def manifold_circular(vae, r_grid, a, n_phi=8):
    Oms, Es = [], []
    for ph in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
        r = torch.tensor(r_grid, requires_grad=True); at = torch.full_like(r, a)
        x = torch.stack([r, torch.full_like(r, np.pi / 2), torch.full_like(r, np.cos(ph)), torch.full_like(r, np.sin(ph))], 1)
        xh = vae.roundtrip(x, at.detach())
        gtt, gtp, gpp, _, _ = metric(xh[:, 0], xh[:, 1], a)
        d = [torch.autograd.grad(c.sum(), r, retain_graph=True)[0] for c in (gtt, gtp, gpp)]
        Om, E, _ = circular_from_metric(gtt, gtp, gpp, *d)
        Oms.append(Om.detach().numpy()); Es.append(E.detach().numpy())
    return np.mean(Oms, 0), np.mean(Es, 0)


def evaluate():
    from scipy.interpolate import UnivariateSpline
    import kerr_v2_gw as G, kerr_v2_ligo as L
    vae, ck = load_spin_model()
    psd_f, psd = np.loadtxt(os.path.join(V2, 'results', 'psd_aLIGO_ZDHP.txt'), unpack=True)
    out = {'train': {k: ck[k] for k in ('steps', 'train_wall_s', 'heldout_abs_dr')},
           'model_parameters': sum(p.numel() for p in vae.parameters()), 'spins': {}}
    dz = np.load(DPATH); out['data'] = {'points': int(len(dz['qpa'])), 'h_err_max': float(dz['h_err'].max()),
                                        'wall_s': float(dz['wall_s'])}
    curves = {}
    t_geo = []
    for a in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95):
        ri = isco_exact(a)
        r = np.linspace(0.9 * ri, 12.0, 1200)
        t0 = time.time(); OmM, EM = manifold_circular(vae, r, a); t_geo.append(time.time() - t0)
        OmX, EX = omega_exact(r, a), energy_exact(r, a)
        sE = UnivariateSpline(r, EM, s=0, k=4); fg = np.linspace(0.92 * ri, 1.15 * ri, 20001)
        riM = float(fg[np.argmin(sE(fg))])
        ok = r > ri
        res = {'isco_exact': float(ri), 'isco_manifold': riM, 'isco_rel_err': riM / ri - 1,
               'omega_rel_err_max': float(np.abs(OmM[ok] / OmX[ok] - 1).max()),
               'omega_rel_err_median': float(np.median(np.abs(OmM[ok] / OmX[ok] - 1))),
               'energy_rel_err_max': float(np.abs(EM[ok] / EX[ok] - 1).max())}
        # aLIGO-weighted inspiral match, M = 60 Msun, eta = 0.01 (exact uses exact ISCO; manifold its own)
        okX = np.isfinite(OmX) & np.isfinite(EX)
        hX, dX, cyc = L.waveform(r[okX], OmX[okX], EX[okX], 0.01, 1.03 * ri, 60.0)
        hM, dM, _ = L.waveform(r, OmM, EM, 0.01, 1.03 * riM, 60.0)
        dt = 1 / 8192
        res['ligo_match_M60_eta0.01'] = L.nw_match(L.resample(hX, dX, dt), L.resample(hM, dM, dt), dt, psd_f, psd)
        res['gw_cycles'] = float(cyc)
        out['spins'][f'a={a}'] = res
        curves[a] = (r, OmM / OmX - 1, ri, riM)
        log(f'a={a}: ' + json.dumps(res))
    out['geometry_extraction_s_per_spin_median'] = float(np.median(t_geo))
    # ISCO across the whole family (fine scan)
    a_scan = np.linspace(0, A_MAX, 39); is_X, is_M = [], []
    for a in a_scan:
        ri = isco_exact(a); r = np.linspace(0.85 * ri, 1.3 * ri, 300)
        _, EM = manifold_circular(vae, r, a, n_phi=4)
        sE = UnivariateSpline(r, EM, s=0, k=4); fg = np.linspace(0.9 * ri, 1.2 * ri, 8001)
        is_X.append(ri); is_M.append(float(fg[np.argmin(sE(fg))]))
    is_X, is_M = np.array(is_X), np.array(is_M)
    out['isco_scan'] = {'a': a_scan.tolist(), 'exact': is_X.tolist(), 'manifold': is_M.tolist(),
                        'rel_err_max': float(np.abs(is_M / is_X - 1).max()),
                        'rel_err_median': float(np.median(np.abs(is_M / is_X - 1)))}
    log('ISCO scan: max rel err %.2e, median %.2e' % (out['isco_scan']['rel_err_max'], out['isco_scan']['rel_err_median']))
    json.dump(out, open(os.path.join(HERE, 'results', 'spin_family.json'), 'w'), indent=1)
    figure(out, curves, a_scan, is_X, is_M)


def figure(out, curves, a_scan, is_X, is_M):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    MUTED = '#6b6a64'
    cols = ['#cfe0f5', '#9cc1ea', '#69a1de', '#3a80cf', '#2a62a8', '#1b4580']
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e4e3de'})
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    a0 = ax[0]
    a0.plot(a_scan, is_X, color=MUTED, lw=2, ls='--', label='exact Kerr (BPT)')
    a0.plot(a_scan, is_M, color='#2a78d6', lw=1.8, label='MW spin-family manifold')
    a0.set_xlabel('spin a'); a0.set_ylabel('r_ISCO / M'); a0.legend(frameon=False, fontsize=7)
    a0.set_title(f"(a) ISCO across the family (max err {100 * out['isco_scan']['rel_err_max']:.2f}%)", loc='left', fontsize=9)
    a1 = ax[1]
    for c, (a, (r, e, ri, riM)) in zip(cols, curves.items()):
        m = r > ri
        a1.plot(r[m], 100 * e[m], color=c, lw=1.6, label=f'a = {a}')
    a1.axhline(0, color=MUTED, lw=1); a1.set_xlabel('r / M'); a1.set_ylabel('Ω manifold − exact (%)')
    a1.legend(frameon=False, fontsize=6.5, ncol=2); a1.set_title('(b) Orbital frequency error', loc='left', fontsize=9)
    a2 = ax[2]
    sp = list(out['spins'].keys()); mm = [1 - out['spins'][k]['ligo_match_M60_eta0.01'] for k in sp]
    a2.bar(range(len(sp)), mm, color='#2a78d6', width=0.6)
    a2.axhline(0.03, color='#eb6834', lw=1.2, ls='--'); a2.text(len(sp) - 0.5, 0.03, ' 0.97 bank\n threshold', fontsize=6.5, va='center', color='#eb6834')
    a2.set_yscale('log'); a2.set_xticks(range(len(sp))); a2.set_xticklabels([k.replace('a=', '') for k in sp])
    a2.set_xlabel('spin a (held out)'); a2.set_ylabel('1 − match (aLIGO)')
    a2.set_title('(c) Mismatch, M = 60 M☉, η = 0.01', loc='left', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'results', 'fig_spin_family.png'), dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['data', 'train', 'eval'])
    ap.add_argument('--steps', type=int, default=100000); ap.add_argument('--n-geo', type=int, default=6000)
    a = ap.parse_args()
    torch.set_num_threads(6)
    LOG = open(os.path.join(HERE, 'results', f'{a.cmd}_log.txt'), 'w')
    if a.cmd == 'data': make_data(n_geo=a.n_geo)
    elif a.cmd == 'train': train(a.steps)
    else: evaluate()
