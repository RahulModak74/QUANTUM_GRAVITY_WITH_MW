#!/usr/bin/env python3
"""
Kerr v2: cost and representation-size benchmark (CPU only).

Measures on the machine it runs on:
  * one-time cost: load the MW model; extract circular-orbit geometry Omega(r), E(r) from the manifold;
  * per-template cost: full adiabatic inspiral 12 M -> 1.03 r_ISCO (quadrupole flux), for a range of
    mass ratios, single-threaded and with the default thread count;
  * representation size: geodesic training set vs the trained model.
Published NR costs are quoted (not measured) in the README with sources.

Writes results/benchmark.json
"""
import json, os, platform, time
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
torch.set_default_dtype(torch.float64)
from kerr_v2_common import load_model, KerrMWVAE  # noqa: E402
import kerr_v2_gw as G  # noqa: E402


def cpu_name():
    try:
        for l in open('/proc/cpuinfo'):
            if l.startswith('model name'):
                return l.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def time_templates(r, Om, E, r_stop, eta_lo, eta_hi, n, rng):
    ts, cyc = [], []
    for _ in range(n):
        eta = rng.uniform(eta_lo, eta_hi)
        t0 = time.perf_counter()
        _, h, c, _ = G.inspiral(r, Om, E, eta, 'quad', r_stop)
        ts.append(time.perf_counter() - t0); cyc.append(c)
    ts = np.array(ts)
    return {'eta_range': [eta_lo, eta_hi], 'n': n, 'ms_median': 1e3 * float(np.median(ts)),
            'ms_p95': 1e3 * float(np.quantile(ts, .95)), 'templates_per_min': 60 / float(np.median(ts)),
            'gw_cycles_median': float(np.median(cyc)), 'gw_cycles_max': float(np.max(cyc)),
            'minutes_for_25000': 25000 * float(np.median(ts)) / 60}


def main():
    out = {'cpu': cpu_name(), 'logical_cpus': os.cpu_count(), 'torch': torch.__version__, 'gpu_used': False}
    t0 = time.perf_counter(); vae = load_model(os.path.join(HERE, 'model', 'kerr_mw_v2.pt'))
    out['model_load_s'] = time.perf_counter() - t0

    r = np.linspace(1.9, 12.0, 1200)
    for th in (1, 6):
        torch.set_num_threads(th)
        t0 = time.perf_counter(); Om, E, _, _ = G.manifold_circular(vae, r)
        out[f'geometry_extraction_s_{th}thread'] = time.perf_counter() - t0
    sE = UnivariateSpline(r, E, s=0, k=4); f = np.linspace(2.0, 3.0, 20001)
    r_stop = 1.03 * float(f[np.argmin(sE(f))])

    # raw geometry query rate: metric + d/dr through the manifold, batched
    x = torch.tensor(np.stack([np.linspace(2.5, 30, 100000), np.full(100000, np.pi / 2),
                               np.ones(100000), np.zeros(100000)], 1))
    torch.set_num_threads(6)
    t0 = time.perf_counter()
    with torch.no_grad():
        vae.roundtrip(x)
    out['manifold_queries_per_s'] = 100000 / (time.perf_counter() - t0)

    rng = np.random.default_rng(0)
    out['templates'] = {
        'eta_0.05-0.25': time_templates(r, Om, E, r_stop, 0.05, 0.25, 200, rng),
        'eta_0.01-0.05': time_templates(r, Om, E, r_stop, 0.01, 0.05, 200, rng),
        'eta_0.001-0.01': time_templates(r, Om, E, r_stop, 0.001, 0.01, 50, rng)}

    n_par = sum(p.numel() for p in vae.parameters())
    dpath = os.path.join(HERE, 'data', 'kerr_geodesics_v2.npz')
    n_pts = int(np.load(dpath)['qp'].shape[0]) if os.path.exists(dpath) else 674651
    out['representation'] = {
        'training_points': n_pts, 'floats_per_point_phase_space': 8,
        'training_set_floats': n_pts * 8, 'model_parameters': n_par,
        'compression_training_set_to_model': n_pts * 8 / n_par,
        'model_file_bytes': os.path.getsize(os.path.join(HERE, 'model', 'kerr_mw_v2.pt')),
        'latent_dim': vae.latent_dim}
    summ = json.load(open(os.path.join(HERE, 'results', 'train_summary.json')))
    out['training'] = {'steps': summ['steps'], 'wall_s_from_log': 3689, 'device': 'CPU, 6 threads'}
    json.dump(out, open(os.path.join(HERE, 'results', 'benchmark.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
