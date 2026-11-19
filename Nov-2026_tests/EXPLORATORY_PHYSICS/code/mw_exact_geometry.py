#!/usr/bin/env python3
"""
MW manifold -> field equations in Einstein's form
=================================================

Reverse direction of the usual workflow: the trained M-W manifold (Nov-2026 v5
checkpoints) is the object, and this script reads off the differential equations that
object satisfies.

The decoder f : U -> R^30 (U = the active latent axes; output in standardised data units,
Euclidean ambient metric) is an immersion. f, df and d2f come from mw_taylor.decoder_jet
(exact, analytic chain rule, float64):

  g_ab        = <d_a f, d_b f>                         pullback (M-W) metric
  Gamma^c_ab  = g^{cd} <d_d f, d_a d_b f>              Levi-Civita connection (exact)
  II_ab       = P_perp d_a d_b f                       second fundamental form (exact)

Field equation (Gauss equation, contracted):
  G_ab = R_ab - 1/2 g_ab R = kappa T^MW_ab
  kappa T^MW_ab = <H, II_ab> - g^{cd}<II_ac, II_bd> - 1/2 g_ab (|H|^2 - |II|^2),   H = g^{ab} II_ab
<.,.> is a sum over data columns, so T^MW splits exactly into column groups
(gravity / quantum / horizon / coordinates / Planck-topology).

Independent checks:
  1. Gauss: Riemann from the connection (d Gamma by 4th-order central differences of the
     exact Gamma, plus Gamma*Gamma) vs Riemann from II. Two different formulas, one object.
  2. Contracted Bianchi: nabla^a G_ab = 0 (hence nabla^a T^MW_ab = 0, conservation).
  3. Geodesic equation on a minimum-energy path between a light and a heavy black hole,
     and the decoded Hawking / Einstein relations along it.

Usage:  python3 mw_exact_geometry.py [--per-scenario 40] [--selftest]
Writes ../results/exact_geometry.json, ../results/exact_geometry_log.txt,
       ../results/exact_geometry_points.npz
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'code'))
sys.path.insert(0, HERE)
from qg_toy_vae_trainer_v5 import QGPhysicsVAE                                       # noqa: E402
from qg_toy_data_generator_v5 import QGJointGenerator, QGParams, COLS, SCENARIOS     # noqa: E402
from mw_taylor import decoder_jet                                                    # noqa: E402

torch.set_default_dtype(torch.float64)
CK = os.path.join(HERE, '..', '..', 'results', 'checkpoints')
OUT = os.path.join(HERE, '..', 'results'); os.makedirs(OUT, exist_ok=True)

GROUPS = {
    'coordinates': ['t', 'x', 'y', 'z'],
    'gravity': ['g_00', 'g_11', 'g_22', 'g_33', 'dg_01', 'dg_02', 'dg_03', 'dg_12',
                'dg_13', 'dg_23', 'dg_11', 'dg_22', 'R', 'T_00', 'T_11', 'T_22', 'T_33'],
    'quantum': ['psi_re', 'psi_im', 'dxdt'],
    'horizon': ['M_BH', 'r_s', 'T_H', 'S_ent'],
    'planck_topology': ['topology', 'disc_flag'],
}
CI = {c: i for i, c in enumerate(COLS)}
logf = None


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True)
    if logf: logf.write(s + '\n'); logf.flush()


def load(name):
    sd = torch.load(os.path.join(CK, f'{name}.pt'), map_location='cpu')
    v = QGPhysicsVAE(30, 20, sd['mean'], sd['std']); v.load_state_dict(sd)
    return v.eval().double()


# ---------------------------------------------------------------- geometry (batched)
def geom(vae, Z, active):
    f, J, H = decoder_jet(vae, Z, active)
    g = J.transpose(1, 2) @ J
    gi = torch.linalg.inv(g)
    Gam = torch.einsum('bcd,bkd,bkae->bcae', gi, J, H)                 # Gamma^c_ae
    P = torch.eye(J.shape[1]) - J @ gi @ J.transpose(1, 2)
    II = torch.einsum('bkl,blae->bkae', P, H)
    return dict(f=f, J=J, H=H, g=g, gi=gi, Gam=Gam, II=II)


def riem_gauss(II):
    return torch.einsum('Bkac,Bkbd->Babcd', II, II) - torch.einsum('Bkad,Bkbc->Babcd', II, II)


def einstein(II, g, gi, cols=None):
    if cols is not None:
        II = II[:, cols]
    Hv = torch.einsum('bae,bkae->bk', gi, II)
    Ric = torch.einsum('bk,bkcd->bcd', Hv, II) - torch.einsum('bae,bkac,bked->bcd', gi, II, II)
    R = torch.einsum('bcd,bcd->b', gi, Ric)
    return Ric - 0.5 * g * R[:, None, None], Ric, R


def shifted(Z, active, h):
    """4th-order central-difference stencil along each active axis: [B, n, 4, 20]."""
    n = len(active); offs = torch.tensor([2, 1, -1, -2], dtype=Z.dtype) * h
    S = Z[:, None, None, :].repeat(1, n, 4, 1)
    for a, ax in enumerate(active):
        S[:, a, :, ax] += offs
    return S


def fd(vals, h):
    """vals [B, n, 4, ...] at offsets (+2h,+h,-h,-2h) -> derivative [B, ..., n] (last axis = direction)."""
    d = (-vals[:, :, 0] + 8 * vals[:, :, 1] - 8 * vals[:, :, 2] + vals[:, :, 3]) / (12 * h)
    return d.movedim(1, -1)


def riem_intrinsic(Gam, dGam, g):
    # dGam[b,c,a,e,x] = d_x Gamma^c_ae ;  R^a_bcd = d_c G^a_db - d_d G^a_cb + G^a_ce G^e_db - G^a_de G^e_cb
    t1 = torch.einsum('Badbc->Babcd', dGam)
    t2 = torch.einsum('Bacbd->Babcd', dGam)
    t3 = torch.einsum('Bace,Bedb->Babcd', Gam, Gam)
    t4 = torch.einsum('Bade,Becb->Babcd', Gam, Gam)
    return torch.einsum('Bae,Bebcd->Babcd', g, t1 - t2 + t3 - t4)


def checks_batch(vae, Z, active, h=1e-3):
    """Gauss + Bianchi for a batch of points (uses 4n shifted jets per point)."""
    B, n = Z.shape[0], len(active)
    base = geom(vae, Z, active)
    S = shifted(Z, active, h).reshape(-1, Z.shape[1])
    sh = geom(vae, S, active)
    Gam_s = sh['Gam'].reshape(B, n, 4, n, n, n)
    G_s = einstein(sh['II'], sh['g'], sh['gi'])[0].reshape(B, n, 4, n, n)
    dGam = fd(Gam_s, h)                                                  # [B,c,a,e,x]
    Rm_int = riem_intrinsic(base['Gam'], dGam, base['g'])
    Rm_ext = riem_gauss(base['II'])
    gauss = (torch.linalg.norm(Rm_int - Rm_ext, dim=(1, 2)).norm(dim=(1, 2)) /
             torch.linalg.norm(Rm_ext, dim=(1, 2)).norm(dim=(1, 2)))
    G = einstein(base['II'], base['g'], base['gi'])[0]
    dG = fd(G_s, h)                                                      # [B,a,b,c] = d_c G_ab
    Gam = base['Gam']
    nab = dG - torch.einsum('Bdca,Bdb->Babc', Gam, G) - torch.einsum('Bdcb,Bad->Babc', Gam, G)
    div = torch.einsum('Bac,Babc->Bb', base['gi'], nab)
    scale = torch.einsum('Bac,Babc->Bb', base['gi'].abs(), nab.abs())
    bian = torch.linalg.norm(div, dim=1) / torch.linalg.norm(scale, dim=1)
    Ric_int = torch.einsum('Bac,Babcd->Bbd', base['gi'], Rm_int)
    asym = torch.linalg.norm(Ric_int - Ric_int.transpose(1, 2), dim=(1, 2)) / torch.linalg.norm(Ric_int, dim=(1, 2))
    R_int = torch.einsum('Bbd,Bbd->B', base['gi'], Ric_int)
    return gauss, bian, asym, R_int


def selftest():
    from torch.func import jacrev
    vae = load('A_full_prior'); act = [i for i in range(20) if i not in (3, 8)]
    z = torch.randn(1, 20) * 0.7
    f, J, H = decoder_jet(vae, z, act)
    def fu(u):
        zz = z[0].clone(); zz[act] = u; return vae.decode(zz.unsqueeze(0)).squeeze(0)
    u = z[0, act]
    print('jet vs reverse-mode autodiff:  J', ((J[0] - jacrev(fu)(u)).norm() / J.norm()).item(),
          ' H', ((H[0] - jacrev(jacrev(fu))(u)).norm() / H.norm()).item())


# ---------------------------------------------------------------- main study
def q(v):
    v = np.asarray(v, dtype=float)
    return {'median': float(np.median(v)), 'p05': float(np.quantile(v, .05)), 'p95': float(np.quantile(v, .95))}


def run_model(name, d, lab, n_per, rng, res, store):
    vae = load(name)
    X = (torch.tensor(d, dtype=torch.float64) - vae.mean) / vae.std
    with torch.no_grad():
        mu, lv = vae.encode(X)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean(0)
    active = [i for i in range(20) if kl[i] > 0.1]
    log(f'\n=== {name}: latent 20D; active axes (KL>0.1) = {len(active)}; inactive = '
        f'{[i for i in range(20) if i not in active]}')
    idx = np.concatenate([rng.choice(np.where(lab == k)[0], n_per, replace=False) for k in range(6)])
    Z = mu[idx].clone()
    t0 = time.time()
    with torch.no_grad():
        base = geom(vae, Z, active)
        G, Ric, R = einstein(base['II'], base['g'], base['gi'])
        parts = {k: einstein(base['II'], base['g'], base['gi'], [CI[c] for c in cols]) for k, cols in GROUPS.items()}
        split_err = torch.linalg.norm(sum(p[0] for p in parts.values()) - G, dim=(1, 2)) / torch.linalg.norm(G, dim=(1, 2))
        gauss, bian, asym, R_int = [], [], [], []
        for s in range(0, len(Z), 20):
            a, b, c, r = checks_batch(vae, Z[s:s + 20], active)
            gauss.append(a); bian.append(b); asym.append(c); R_int.append(r)
        gauss, bian, asym, R_int = map(torch.cat, (gauss, bian, asym, R_int))
        ev = torch.linalg.eigvalsh(base['g'])
        Rm = riem_gauss(base['II'])
        n = len(active); iu = torch.triu_indices(n, n, 1)
        gaa = torch.diagonal(base['g'], dim1=1, dim2=2)
        K = Rm[:, iu[0], iu[1], iu[0], iu[1]] / (gaa[:, iu[0]] * gaa[:, iu[1]] - base['g'][:, iu[0], iu[1]] ** 2)
        legacy = 0.5 * (torch.logdet(base['g']) + torch.diagonal(base['gi'], dim1=1, dim2=2).sum(1) - n)
    log(f'  geometry + checks for {len(Z)} points: {time.time() - t0:.0f}s')

    out = {'latent_dim': 20, 'active_dims': len(active), 'points': int(len(idx)),
           'gauss_rel_err': q(gauss), 'bianchi_rel': q(bian), 'ricci_asymmetry_rel': q(asym),
           'R_connection_vs_R_gauss_rel': q(((R_int - R).abs() / R.abs()).numpy()),
           'source_split_rel_err': q(split_err), 'metric_eig_min': q(ev[:, 0]), 'metric_eig_max': q(ev[:, -1]),
           'metric_condition': q(ev[:, -1] / ev[:, 0]), 'legacy_R_proxy': q(legacy), 'ricci_scalar': q(R),
           'abs_ricci_scalar': q(R.abs()), 'einstein_norm': q(torch.linalg.norm(G, dim=(1, 2))),
           'sectional_K': q(K.flatten()), 'sectional_K_neg_frac': float((K < 0).double().mean()),
           'per_scenario': {}}
    absR = torch.stack([p[2].abs() for p in parts.values()], 1)
    share = absR / absR.sum(1, keepdim=True)
    for k, sc in enumerate(SCENARIOS):
        m = torch.tensor(lab[idx] == k)
        out['per_scenario'][sc] = {
            'R': q(R[m]), 'einstein_norm': q(torch.linalg.norm(G[m], dim=(1, 2))),
            'R_by_source_median': {g_: float(parts[g_][2][m].median()) for g_ in GROUPS},
            'source_share_mean': {g_: float(share[m, j].mean()) for j, g_ in enumerate(GROUPS)}}
    store[name] = dict(R=R.numpy(), lab=lab[idx], share=share.numpy(), K=K.numpy(), gauss=gauss.numpy(),
                       bian=bian.numpy(), legacy=legacy.numpy(), eig=ev.numpy(),
                       R_parts=np.stack([parts[g_][2].numpy() for g_ in GROUPS], 1))
    log(f'  Gauss    |Riem(connection) - Riem(II)| / |Riem|  median {out["gauss_rel_err"]["median"]:.1e}  '
        f'p95 {out["gauss_rel_err"]["p95"]:.1e}')
    log(f'  Bianchi  |nabla^a G_ab| / no-cancellation scale    median {out["bianchi_rel"]["median"]:.1e}  '
        f'p95 {out["bianchi_rel"]["p95"]:.1e}')
    log(f'  Ricci asymmetry median {out["ricci_asymmetry_rel"]["median"]:.1e}; source-split error '
        f'{out["source_split_rel_err"]["median"]:.1e}')
    log(f'  metric eig min median {out["metric_eig_min"]["median"]:.2e}, condition median '
        f'{out["metric_condition"]["median"]:.0f}')
    log(f'  Ricci scalar median {out["ricci_scalar"]["median"]:.2f} [p05 {out["ricci_scalar"]["p05"]:.1f}, '
        f'p95 {out["ricci_scalar"]["p95"]:.1f}];  legacy "R" proxy median {out["legacy_R_proxy"]["median"]:.3e}')
    log(f'  sectional K (coordinate planes) median {out["sectional_K"]["median"]:.3f} '
        f'[{out["sectional_K"]["p05"]:.3f}, {out["sectional_K"]["p95"]:.3f}], negative fraction '
        f'{out["sectional_K_neg_frac"]:.2f}')
    for sc, v in out['per_scenario'].items():
        sh = ', '.join(f'{k} {100 * s:.0f}%' for k, s in v['source_share_mean'].items())
        log(f'   {sc:10s} R median {v["R"]["median"]:9.2f} | |R| share: {sh}')
    out['geodesic'] = geodesic_test(vae, mu, d, lab, active, name)
    res[name] = out


def geodesic_test(vae, mu, d, lab, active, name, n_int=80, steps=3000):
    """Minimum-energy path (discrete geodesic) between a light and a heavy black hole, both
    inside the training mass range; compare with the straight latent line."""
    bh = np.where(lab == SCENARIOS.index('black_hole'))[0]
    M = d[bh, CI['M_BH']]
    a, b = bh[np.argmin(np.abs(M - 0.6))], bh[np.argmin(np.abs(M - 1.9))]
    za, zb = mu[a].clone(), mu[b].clone()
    inactive = [i for i in range(mu.shape[1]) if i not in active]
    za[inactive] = 0.0; zb[inactive] = 0.0        # keep the whole path inside the active-axis chart
    act = torch.tensor(active)
    t = torch.linspace(0, 1, n_int + 2)[:, None]
    line = (1 - t) * za + t * zb
    inner = line[1:-1, act].clone().requires_grad_(True)
    def path(inn):
        p = line.clone(); p[1:-1, act] = inn; return p
    opt = torch.optim.Adam([inner], lr=5e-3)                  # coarse
    for _ in range(steps):
        opt.zero_grad(); x = vae.decode(path(inner))
        ((x[1:] - x[:-1]) ** 2).sum().backward(); opt.step()
    opt = torch.optim.LBFGS([inner], lr=1, max_iter=2000, tolerance_grad=1e-12, tolerance_change=1e-15,
                            history_size=50, line_search_fn='strong_wolfe')   # polish to the critical point

    def closure():
        opt.zero_grad(); x = vae.decode(path(inner)); e = ((x[1:] - x[:-1]) ** 2).sum(); e.backward(); return e
    opt.step(closure)
    with torch.no_grad():
        geo = path(inner)
        xg_n, xl_n = vae.decode(geo), vae.decode(line)
        E = lambda x: ((x[1:] - x[:-1]) ** 2).sum().item() * (n_int + 1)
        L = lambda x: torch.linalg.norm(x[1:] - x[:-1], dim=1).sum().item()
        h = 1.0 / (n_int + 1)
        U = geo[:, act]
        up = (U[2:] - U[:-2]) / (2 * h); upp = (U[2:] - 2 * U[1:-1] + U[:-2]) / h ** 2
        Gam = geom(vae, geo[1:-1], active)['Gam']
        acc = torch.einsum('bcae,ba,be->bc', Gam, up, up)
        resid = (torch.linalg.norm(upp + acc, dim=1) / (torch.linalg.norm(upp, dim=1) + torch.linalg.norm(acc, dim=1)))
        resid = resid[2:-2]

        def tangential_fraction(P):
            # geodesic <=> data-space acceleration is normal to the manifold (P_tan x'' = 0)
            x = vae.decode(P); xpp = (x[2:] - 2 * x[1:-1] + x[:-2]) / h ** 2
            gg = geom(vae, P[1:-1], active)
            Pt = gg['J'] @ gg['gi'] @ gg['J'].transpose(1, 2)
            tf = torch.linalg.norm(torch.einsum('bkl,bl->bk', Pt, xpp), dim=1) / torch.linalg.norm(xpp, dim=1)
            return tf[2:-2].numpy()
        tf_geo, tf_line = tangential_fraction(geo), tangential_fraction(line)
        seg = torch.linalg.norm(xg_n[1:] - xg_n[:-1], dim=1)
        xg, xl = vae.denormalize(xg_n).numpy(), vae.denormalize(xl_n).numpy()

    def rel(x):
        Mx, TH = x[:, CI['M_BH']], x[:, CI['T_H']]
        tr = 8 * np.pi * (-x[:, CI['T_00']] + x[:, CI['T_11']] + x[:, CI['T_22']] + x[:, CI['T_33']])
        return {'hawking_abs_8piMTH_minus_1_mean': float(np.mean(np.abs(8 * np.pi * Mx * TH - 1))),
                'einstein_rel_resid': float(np.mean(np.abs(x[:, CI['R']] - tr)) / np.mean(np.abs(tr))),
                'M_along_path': [round(float(m), 3) for m in Mx[::4]]}
    out = {'M_start': float(d[a, CI['M_BH']]), 'M_end': float(d[b, CI['M_BH']]),
           'length_geodesic': L(xg_n), 'length_line': L(xl_n), 'energy_geodesic': E(xg_n), 'energy_line': E(xl_n),
           'ode_resid': q(resid.numpy()), 'speed_cv': float(seg.std() / seg.mean()),
           'tangential_accel_fraction_geodesic': q(tf_geo), 'tangential_accel_fraction_line': q(tf_line),
           'geodesic': rel(xg), 'straight_line': rel(xl)}
    log(f'  geodesic, black hole M {out["M_start"]:.2f} -> {out["M_end"]:.2f}: length {out["length_geodesic"]:.3f} '
        f'vs straight latent line {out["length_line"]:.3f}; geodesic-ODE residual median '
        f'{out["ode_resid"]["median"]:.2e}; speed CV {out["speed_cv"]:.3f}')
    log(f'    tangential share of acceleration |P_tan x\'\'|/|x\'\'|: geodesic median '
        f'{out["tangential_accel_fraction_geodesic"]["median"]:.3f} vs straight latent line '
        f'{out["tangential_accel_fraction_line"]["median"]:.3f}')
    log(f'    along geodesic: |8 pi M T_H - 1| = {out["geodesic"]["hawking_abs_8piMTH_minus_1_mean"]:.3f}, '
        f'Einstein residual {out["geodesic"]["einstein_rel_resid"]:.3f}; M path {out["geodesic"]["M_along_path"]}')
    log(f'    along straight line: |8 pi M T_H - 1| = {out["straight_line"]["hawking_abs_8piMTH_minus_1_mean"]:.3f}, '
        f'Einstein residual {out["straight_line"]["einstein_rel_resid"]:.3f}')
    np.savez(os.path.join(OUT, f'geodesic_path_{name}.npz'), geo=xg, line=xl)
    return out


def main():
    global logf
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-scenario', type=int, default=40)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.selftest:
        selftest(); return
    logf = open(os.path.join(OUT, 'exact_geometry_log.txt'), 'w')
    d, lab, cols = QGJointGenerator(QGParams(seed=123, n_samples=6000)).generate()
    assert cols == COLS
    res = {'test_data': 'v5 generator, seed 123 (not used in training), 6000 rows; latent points = encoder means',
           'ambient': 'decoder output in standardised data units (each column z-scored), Euclidean',
           'kappa': '8*pi*G with G = 1', 'groups': GROUPS}
    store = {}
    for name in ['A_full_prior', 'B_full_noprior']:
        run_model(name, d, lab, a.per_scenario, np.random.default_rng(7), res, store)
    json.dump(res, open(os.path.join(OUT, 'exact_geometry.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'exact_geometry_points.npz'),
             **{f'{m}__{k}': v for m, dct in store.items() for k, v in dct.items()}, groups=list(GROUPS))
    log('\nwrote results/exact_geometry.json, exact_geometry_points.npz')


if __name__ == '__main__':
    main()
