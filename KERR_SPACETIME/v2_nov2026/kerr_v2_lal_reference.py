#!/usr/bin/env python3
"""
Kerr v2: reference data from LIGO's own library (LALSuite), run in a separate env:
    ~/.venvs/gw/bin/python kerr_v2_lal_reference.py        (pip install lalsuite numpy scipy)

1. Advanced LIGO design noise curve (zero-detuned high power, lalsimulation) -> results/psd_aLIGO_ZDHP.txt
2. Wall time of standard LIGO waveform approximants on this CPU, same masses and band as the MW
   templates, for context: these are hand-derived models (PN / EOB / phenomenological) that already
   generate in milliseconds. Writes results/lal_reference_timing.json
"""
import json, os, time
import numpy as np
import lal
import lalsimulation as ls

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

f = np.arange(5.0, 4096.0, 0.25)
psd = np.array([ls.SimNoisePSDaLIGOZeroDetHighPower(x) for x in f])
np.savetxt(os.path.join(RES, 'psd_aLIGO_ZDHP.txt'), np.c_[f, psd], header='f_Hz  S_n(f) [1/Hz]  aLIGO ZDHP (lalsimulation)')


def td(approx, m1, m2, chi1, flow, dt=1 / 4096):
    p = lal.CreateDict()
    t0 = time.perf_counter()
    hp, hc = ls.SimInspiralChooseTDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, chi1, 0, 0, 0,
                                            100e6 * lal.PC_SI, 0, 0, 0, 0, 0, dt, flow, flow, p,
                                            ls.GetApproximantFromString(approx))
    return time.perf_counter() - t0, hp.data.length * dt


def fd(approx, m1, m2, chi1, flow, df=1 / 64):
    p = lal.CreateDict()
    t0 = time.perf_counter()
    ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, chi1, 0, 0, 0,
                                   100e6 * lal.PC_SI, 0, 0, 0, 0, 0, df, flow, 2048., flow, p,
                                   ls.GetApproximantFromString(approx))
    return time.perf_counter() - t0, None


out = {'lalsuite': lal.__version__, 'note': 'median of 5 calls; aligned spin chi1 = 0.9 on the heavier body; f_low = 20 Hz. SEOBNRv4 is NR-calibrated to mass ratio 1:8 (plus the test-particle limit) and IMRPhenomD to 1:18; at eta 0.0099 (1:99) both run outside their NR-calibrated range (timing only)',
       'cases': {}}
M = 60.0
for eta in (0.1, 0.0099):
    m1 = M * (1 + np.sqrt(1 - 4 * eta)) / 2; m2 = M - m1
    case = {'m1': m1, 'm2': m2}
    for name, fn in (('SpinTaylorT4', td), ('SEOBNRv4_opt', td), ('IMRPhenomD', fd)):
        try:
            ts = [fn(name, m1, m2, 0.9, 20.0)[0] for _ in range(5)]
            case[name + '_ms'] = 1e3 * float(np.median(ts))
        except Exception as e:  # approximant outside its validity range
            case[name + '_ms'] = None; case[name + '_error'] = str(e)[:120]
    out['cases'][f'M={M:.0f}Msun_eta={eta}'] = case
json.dump(out, open(os.path.join(RES, 'lal_reference_timing.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
