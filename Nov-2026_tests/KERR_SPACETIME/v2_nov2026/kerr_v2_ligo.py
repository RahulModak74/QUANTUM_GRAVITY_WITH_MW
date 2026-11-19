#!/usr/bin/env python3
"""
Kerr v2: matches weighted by the Advanced LIGO design noise curve, in physical units.

For a total mass M (solar masses) and symmetric mass ratio eta, build the quadrupole-flux adiabatic inspiral
12 M -> 1.03 r_ISCO twice: from exact Kerr geometry and from the MW manifold's geometry. Convert to seconds,
FFT, and compute the noise-weighted match
    match = max_{t0, phi0} <h_X, h_M> / sqrt(<h_X,h_X><h_M,h_M>),  <a,b> = 4 Re sum a~ b~* / S_n(f) df
over f in [20 Hz, 2048 Hz], with S_n = aLIGO zero-detuned high power (results/psd_aLIGO_ZDHP.txt, from
kerr_v2_lal_reference.py). The waveforms carry both polarisations (face-on), so |complex overlap| maximises
over phase. A 0.97 minimal match is the usual template-bank standard in LIGO searches.

Writes results/ligo_match.json
"""
import json, os
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from scipy.signal.windows import tukey

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
torch.set_default_dtype(torch.float64)
from kerr_v2_common import load_model, omega_exact, energy_exact, isco_exact  # noqa: E402
import kerr_v2_gw as G  # noqa: E402

MSUN_S = 4.925490947641267e-06   # G M_sun / c^3 in seconds
F_LO, F_HI = 20.0, 2048.0


def waveform(r, Om, E, eta, r_stop, M_sun):
    t, h, cyc, dt = G.inspiral(r, Om, E, eta, 'quad', r_stop)
    return h, dt * M_sun * MSUN_S, cyc


def resample(h, dt_in, dt_out):
    t_in = np.arange(len(h)) * dt_in
    t_out = np.arange(0, t_in[-1], dt_out)
    return np.interp(t_out, t_in, h.real) + 1j * np.interp(t_out, t_in, h.imag)


def nw_match(h1, h2, dt, psd_f, psd):
    n = 1 << int(np.ceil(np.log2(2 * max(len(h1), len(h2)))))
    w1, w2 = tukey(len(h1), 0.05), tukey(len(h2), 0.05)
    H1, H2 = np.fft.fft(h1 * w1, n) * dt, np.fft.fft(h2 * w2, n) * dt
    f = np.abs(np.fft.fftfreq(n, dt))
    band = (f >= F_LO) & (f <= F_HI)
    S = np.interp(f, psd_f, psd); W = np.where(band, 1 / S, 0.0)
    c = np.fft.ifft(H1 * np.conj(H2) * W)
    n1, n2 = np.sum(np.abs(H1) ** 2 * W), np.sum(np.abs(H2) ** 2 * W)
    return float(np.abs(c).max() * n / np.sqrt(n1 * n2))


def main():
    psd_f, psd = np.loadtxt(os.path.join(HERE, 'results', 'psd_aLIGO_ZDHP.txt'), unpack=True)
    vae = load_model(os.path.join(HERE, 'model', 'kerr_mw_v2.pt'))
    r = np.linspace(1.9, 12.0, 1200)
    Om_M, E_M, _, _ = G.manifold_circular(vae, r)
    Om_X, E_X = omega_exact(r), energy_exact(r)
    sE = UnivariateSpline(r, E_M, s=0, k=4); fg = np.linspace(2.0, 3.0, 20001)
    rs_M, rs_X = 1.03 * float(fg[np.argmin(sE(fg))]), 1.03 * isco_exact()
    out = {'psd': 'aLIGO ZDHP (lalsimulation)', 'band_Hz': [F_LO, F_HI], 'spin': 0.9, 'cases': {}}
    for M in (20.0, 60.0, 100.0):
        for eta in (0.1, 0.01):
            hX, dX, cyc = waveform(r, Om_X, E_X, eta, rs_X, M)
            hM, dM, _ = waveform(r, Om_M, E_M, eta, rs_M, M)
            dt = 1 / 8192
            hX, hM = resample(hX, dX, dt), resample(hM, dM, dt)
            f_start = omega_exact(12.0) / np.pi / (M * MSUN_S)
            f_isco = omega_exact(isco_exact()) / np.pi / (M * MSUN_S)
            m = nw_match(hX, hM, dt, psd_f, psd)
            key = f'M={M:.0f}Msun_eta={eta}'
            out['cases'][key] = {'match_aLIGO': m, 'mismatch': 1 - m, 'duration_s': len(hX) * dt,
                                 'gw_cycles': cyc, 'f_start_Hz': f_start, 'f_isco_Hz': f_isco}
            print(key, json.dumps(out['cases'][key]))
    json.dump(out, open(os.path.join(HERE, 'results', 'ligo_match.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
