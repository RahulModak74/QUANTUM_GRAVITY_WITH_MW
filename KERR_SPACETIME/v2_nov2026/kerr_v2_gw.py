#!/usr/bin/env python3
"""
Kerr v2 step 3: gravitational waves from the MW manifold, with no amplitude calibration.

1. Diagnose v1 (gravitational_wave_inference_STABLE_9.py, kerr_vae.pth):
   - the sqrt|Van Vleck| factor that multiplies the quadrupole, and the auto-calibration factor;
   - the v1 VAE round trip on the orbit it analyses;
   - the v1 match with the VAE factor removed (shows how much of the match the geometry drives).
2. Geometry through the manifold: the Kerr metric is evaluated at the manifold's reconstruction
   x_hat = decode(encode(x)); d/dr flows through the encoder and decoder (autograd). From that metric:
   circular-orbit Omega(r), E(r), ISCO, compared with exact Kerr (Bardeen-Press-Teukolsky).
3. Amplitude: quadrupole h0 = 4 mu r^2 Omega^2 / D from a densely sampled orbit (numerical second
   derivative) against the analytic value; no calibration step.
4. Adiabatic circular inspiral: dE_orb/dt = -F, h = 4 mu v^2/D e^{-2i Phi}, v = Omega^{1/3},
   F = 32/5 eta^2 v^10 f(v) with f = 1 (quadrupole, 12 M -> 1.03 r_ISCO) or the test-mass Kerr 2.5PN
   series (12 M -> 6 M only: for a = 0.9 that series turns negative inside r ~ 4 M). Matches
   (flat noise, maximised over time and phase) between manifold and exact geometry.
Writes results/gw_v2.json and results/fig_kerr_v2.png
"""
import io, contextlib, json, os, sys
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from kerr_v2_common import (A_SPIN, metric, omega_exact, energy_exact, isco_exact,
                            circular_from_metric, load_model)

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, '..')
torch.set_default_dtype(torch.float64)
RES = os.path.join(HERE, 'results')
OUT = {}
D = 100.0


# ------------------------------------------------------------------ 1. v1 diagnosis
def diagnose_v1():
    torch.set_default_dtype(torch.float32)          # the v1 model is float32
    try:
        return _diagnose_v1()
    finally:
        torch.set_default_dtype(torch.float64)


def _diagnose_v1():
    sys.path.insert(0, V1)
    from gravitational_wave_inference_STABLE_9 import GravitationalWaveInferenceFIXED as G
    with contextlib.redirect_stdout(io.StringIO()):
        g = G(os.path.join(V1, 'kerr_vae.pth'))
        orb = g.generate_circular_orbit(r_orbit=g.ISCO * 1.5)
        hp, hc, t = g.compute_waveform_with_learned_geometry(orb, distance=D)
        pp, pc, tp = g.compute_exact_pn_waveform(orb, distance=D)
        m_plus = g.compute_match(hp, t, pp, tp)
        # same pipeline with the VAE factor set to 1 (pure analytic orbit)
        orig = g.vae.van_vleck_determinant
        g.vae.van_vleck_determinant = lambda a, b: (1.0, 1.0)
        hp1, hc1, t1 = g.compute_waveform_with_learned_geometry(orb, distance=D)
        m_plus_noVAE = g.compute_match(hp1, t1, pp, tp)
        g.vae.van_vleck_determinant = orig
    v = g.vae
    x = torch.tensor(orb[::50], dtype=torch.float32, device=g.device)
    deltas = []
    for xa in x[:10]:
        xb = xa.clone(); xb[1] += .01; xb[3] += .01
        with torch.enable_grad(), contextlib.redirect_stdout(io.StringIO()):
            deltas.append(v.van_vleck_determinant(xa, xb)[0])
    with torch.no_grad():
        xr = v.decode(v.encode(x)[0]).cpu().numpy()
    r_true = orb[0, 1]
    # what the v1 auto-calibration targeted, and the standard leading-order amplitude
    M, a, r = 1.0, g.a, r_true
    Om = 1 / (r ** 1.5 + a); mu = 0.1
    Mc = (mu ** 3 * M ** 2) ** 0.2
    v1_target = 2 * mu / D * (np.pi * Mc * Om / (2 * np.pi)) ** (2 / 3)
    out = {'orbit_r': float(r_true), 'v1_match_plus': float(m_plus), 'v1_match_plus_with_VAE_factor_removed': float(m_plus_noVAE),
           'sqrt_abs_van_vleck_median': float(np.median(np.sqrt(np.abs(deltas)))),
           'v1_target_amplitude': float(v1_target),
           'quadrupole_amplitude_4mu_r2_Om2_over_D': float(4 * mu * r ** 2 * Om ** 2 / D),
           'v1_roundtrip_r_on_orbit': {'true': float(r_true), 'reconstructed_min': float(xr[:, 1].min()),
                                       'reconstructed_max': float(xr[:, 1].max()),
                                       'median_abs_err': float(np.median(np.abs(xr[:, 1] - r_true)))}}
    print('v1 diagnosis:', json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ 2. manifold geometry
def manifold_circular(vae, r_grid, n_phi=8):
    """Omega, E, L of prograde circular equatorial orbits from the metric at the manifold reconstruction."""
    Om_all, E_all = [], []
    for ph in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
        r = torch.tensor(r_grid, requires_grad=True)
        x = torch.stack([r, torch.full_like(r, np.pi / 2), torch.full_like(r, np.cos(ph)), torch.full_like(r, np.sin(ph))], 1)
        xh = vae.roundtrip(x)
        gtt, gtp, gpp, _, _ = metric(xh[:, 0], xh[:, 1])
        d = [torch.autograd.grad(c.sum(), r, retain_graph=True)[0] for c in (gtt, gtp, gpp)]
        Om, E, _ = circular_from_metric(gtt, gtp, gpp, *d)
        Om_all.append(Om.detach().numpy()); E_all.append(E.detach().numpy())
    Om_all, E_all = np.array(Om_all), np.array(E_all)
    return Om_all.mean(0), E_all.mean(0), Om_all.std(0), E_all.std(0)


# ------------------------------------------------------------------ 4. inspiral
def pn_flux_factor(v, q=A_SPIN):
    return (1 - 1247 / 336 * v ** 2 + (4 * np.pi - 11 / 4 * q) * v ** 3 + (-44711 / 9072 + 33 / 16 * q ** 2) * v ** 4
            + (-8191 / 672 * np.pi - 59 / 16 * q) * v ** 5)


R_PN_STOP = 6.0   # the test-mass 2.5PN flux factor f(v) for a = 0.9 falls to 0 at r ~ 4 M and is negative inside,
                  # so PN-flux inspirals stop at 6 M (f = 0.54); quadrupole-flux inspirals run to 1.03 r_ISCO


def inspiral(r_grid, Om, E, eta, flux, r_stop):
    """Return uniformly sampled complex h(t) = h+ - i hx (face-on) for an adiabatic inspiral."""
    if flux == 'pn':
        r_stop = max(r_stop, R_PN_STOP)
    sE = UnivariateSpline(r_grid, E, s=0, k=4); sO = UnivariateSpline(r_grid, Om, s=0, k=4)
    rr = np.linspace(r_grid[-1], r_stop, 400000)
    Omr, dE = sO(rr), sE.derivative()(rr)
    v = Omr ** (1 / 3)
    f = pn_flux_factor(v) if flux == 'pn' else 1.0
    dtdr = -dE / (32 / 5 * eta * v ** 10 * f)                      # M = 1, mu = eta
    t = np.concatenate([[0], np.cumsum(0.5 * (dtdr[1:] + dtdr[:-1]) * np.diff(rr))])
    Phi = np.concatenate([[0], np.cumsum(0.5 * (Omr[1:] * dtdr[1:] + Omr[:-1] * dtdr[:-1]) * np.diff(rr))])
    dt = (2 * np.pi / (2 * Omr.max())) / 24                        # 24 samples per GW cycle at the end
    tu = np.arange(0, t[-1], dt)
    Phu, vu = np.interp(tu, t, Phi), np.interp(tu, t, v)
    h = 4 * eta * vu ** 2 / D * np.exp(-2j * Phu)
    return tu, h, float(Phi[-1] / np.pi), dt                       # GW cycles = Phi/pi


def match(h1, h2):
    n = 1 << int(np.ceil(np.log2(len(h1) + len(h2))))
    c = np.fft.ifft(np.fft.fft(h1, n) * np.conj(np.fft.fft(h2, n)))
    return float(np.abs(c).max() / (np.linalg.norm(h1) * np.linalg.norm(h2)))


def main():
    os.makedirs(RES, exist_ok=True)
    OUT['v1_diagnosis'] = diagnose_v1()
    vae = load_model(os.path.join(HERE, 'model', 'kerr_mw_v2.pt'))
    risco = isco_exact()
    r_grid = np.linspace(1.9, 12.0, 1200)
    Om_M, E_M, Om_sd, E_sd = manifold_circular(vae, r_grid)
    Om_X, E_X = omega_exact(r_grid), energy_exact(r_grid)
    ok = r_grid > risco
    sE = UnivariateSpline(r_grid, E_M, s=0, k=4)
    fine = np.linspace(2.0, 3.0, 20001)
    risco_M = float(fine[np.argmin(sE(fine))])
    sel = lambda rv: int(np.argmin(np.abs(r_grid - rv)))
    OUT['geometry'] = {
        'isco_exact': float(risco), 'isco_manifold': risco_M, 'isco_rel_err': risco_M / risco - 1,
        'omega_rel_err_max_outside_isco': float(np.abs(Om_M[ok] / Om_X[ok] - 1).max()),
        'omega_rel_err_median': float(np.median(np.abs(Om_M[ok] / Om_X[ok] - 1))),
        'energy_rel_err_max_outside_isco': float(np.abs(E_M[ok] / E_X[ok] - 1).max()),
        'omega_phi_scatter_max': float((Om_sd[ok] / Om_X[ok]).max()),
        'table': {f'r={rv}': {'Omega_exact': float(omega_exact(rv)), 'Omega_manifold': float(Om_M[sel(rv)]),
                              'E_exact': float(energy_exact(rv)), 'E_manifold': float(E_M[sel(rv)])}
                  for rv in (2.5, 3.48, 5.0, 8.0, 12.0)}}
    print('geometry:', json.dumps(OUT['geometry'], indent=1))

    # 3. amplitude, no calibration: dense quadrupole second derivative vs analytic
    r0 = 1.5 * risco; mu = 0.1
    Om_r0_M = float(np.interp(r0, r_grid, Om_M)); Om_r0_X = omega_exact(r0)
    t = np.linspace(0, 5 * 2 * np.pi / Om_r0_M, 20001)
    Ph = Om_r0_M * t
    Ixx_m_Iyy = mu * r0 ** 2 * np.cos(2 * Ph)
    hplus = np.gradient(np.gradient(Ixx_m_Iyy, t), t) / D
    h0_num = float(np.abs(hplus[100:-100]).max())
    OUT['amplitude'] = {'r': r0, 'h0_numerical_manifold': h0_num,
                        'h0_analytic_manifold_4mu_r2_Om2_over_D': 4 * mu * r0 ** 2 * Om_r0_M ** 2 / D,
                        'h0_analytic_exact_kerr': 4 * mu * r0 ** 2 * Om_r0_X ** 2 / D,
                        'calibration_factor_applied': 1.0}
    OUT['amplitude']['ratio_manifold_to_exact'] = OUT['amplitude']['h0_numerical_manifold'] / OUT['amplitude']['h0_analytic_exact_kerr']
    print('amplitude:', OUT['amplitude'])

    # 4. inspirals
    OUT['inspiral'] = {}
    r_stop_X, r_stop_M = 1.03 * risco, 1.03 * risco_M
    for eta in (0.1, 0.01):
        res = {}
        hX = {fl: inspiral(r_grid, Om_X, E_X, eta, fl, r_stop_X) for fl in ('quad', 'pn')}
        hM = {fl: inspiral(r_grid, Om_M, E_M, eta, fl, r_stop_M) for fl in ('quad', 'pn')}
        for fl in ('quad', 'pn'):
            # resample manifold waveform onto the exact one's time step
            tM, h, _, _ = hM[fl]; tX, hx, cyc, dt = hX[fl]
            hMr = np.interp(np.arange(0, tM[-1], dt), tM, h.real) + 1j * np.interp(np.arange(0, tM[-1], dt), tM, h.imag)
            res[f'match_manifold_vs_exact_{fl}_flux'] = match(hMr, hx)
            res[f'gw_cycles_exact_{fl}'] = cyc
            res[f'gw_cycles_manifold_{fl}'] = hM[fl][2]
            res[f'amplitude_ratio_end_{fl}'] = float(np.abs(hMr[-1]) / np.abs(hx[-1]))
        OUT['inspiral'][f'eta={eta}'] = res
        print(f'inspiral eta={eta}:', json.dumps(res, indent=1))
        if eta == 0.1:
            FIG = dict(tX=hX['quad'][0], hX=hX['quad'][1], tM=hM['quad'][0], hM=hM['quad'][1])
    json.dump(OUT, open(os.path.join(RES, 'gw_v2.json'), 'w'), indent=1)
    figure(r_grid, Om_M, Om_X, E_M, E_X, risco, risco_M, FIG)


def figure(r, OmM, OmX, EM, EX, risco, riscoM, F):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    C1, C2, MUTED = '#2a78d6', '#eb6834', '#6b6a64'
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e4e3de'})
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6))
    ok = r > risco
    a = ax[0, 0]
    a.plot(r[ok], 100 * (OmM[ok] / OmX[ok] - 1), color=C1, lw=2, label='Ω (orbital frequency)')
    a.plot(r[ok], 100 * (EM[ok] / EX[ok] - 1), color=C2, lw=2, label='E (orbital energy)')
    a.axhline(0, color=MUTED, lw=1)
    a.set_xlabel('r / M'); a.set_ylabel('manifold − exact (%)'); a.legend(frameon=False, fontsize=7)
    a.set_title('(a) Kerr circular orbits through the MW manifold', loc='left', fontsize=9)
    b = ax[0, 1]
    m = (r > 2.0) & (r < 6)
    b.plot(r[m], EX[m], color=MUTED, lw=2, ls='--', label='exact Kerr')
    b.plot(r[m], EM[m], color=C1, lw=2, label='MW manifold')
    b.axvline(risco, color=MUTED, lw=1, ls=':'); b.axvline(riscoM, color=C1, lw=1, ls=':')
    b.set_xlabel('r / M'); b.set_ylabel('E per unit mass'); b.legend(frameon=False, fontsize=7)
    b.set_title(f'(b) ISCO: exact {risco:.3f} M, manifold {riscoM:.3f} M', loc='left', fontsize=9)
    c = ax[1, 0]
    n = int(6 * 24)
    c.plot(F['tX'][-n:] - F['tX'][-1], F['hX'][-n:].real * 1e3, color=MUTED, lw=2, ls='--', label='exact Kerr geometry')
    hM_al = F['hM'] * np.exp(1j * (np.angle(F['hX'][-1]) - np.angle(F['hM'][-1])))   # same alignment freedom as the match
    c.plot(F['tM'][-n:] - F['tM'][-1], hM_al[-n:].real * 1e3, color=C1, lw=1.6, label='MW manifold geometry')
    c.set_xlabel('t − t_end (M)'); c.set_ylabel('h₊ × 10³  (D = 100 M)'); c.legend(frameon=False, fontsize=7)
    c.set_title('(c) Last cycles to ISCO, aligned (η=0.1, quadrupole flux)', loc='left', fontsize=9)
    d = ax[1, 1]
    d1 = OUT['v1_diagnosis']
    vals = [d1['quadrupole_amplitude_4mu_r2_Om2_over_D'] * d1['sqrt_abs_van_vleck_median'],
            OUT['amplitude']['h0_numerical_manifold'], OUT['amplitude']['h0_analytic_exact_kerr']]
    d.bar(range(3), vals, color=[MUTED, C1, C2], width=0.6)
    d.set_yscale('log'); d.set_xticks(range(3))
    d.set_xticklabels(['v1 raw\n(×√|Δ| artefact)', 'v2 manifold\n(no calibration)', 'exact Kerr\nquadrupole'], fontsize=7.5)
    d.set_ylabel('h₀ at r = 1.5 r_ISCO'); d.set_title('(d) Amplitude without calibration', loc='left', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_kerr_v2.png'), dpi=200)


if __name__ == '__main__':
    main()
