#!/usr/bin/env python3
"""Apply the pre-registered decision rule (../results/nu_test_PREREGISTRATION.md) to
../results/nu_test.json. Prints a table and writes ../results/nu_test_verdict.json."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
r = json.load(open(os.path.join(RES, 'nu_test.json')))
NUS, SEEDS = ['0.8', '1.5', '3.0'], ['0', '1']
# direction: True = higher is better
P = {'P1_flag_accuracy': True, 'P2_flag_mae': False, 'P3_wormhole_knn_recall': True}

print(f'{"run":16s} {"Einstein":>9s} {"H slope":>8s} {"H R2":>6s} {"P1 acc":>8s} {"P2 mae":>8s} {"P3 recall":>9s}')
for nu in NUS:
    for s in SEEDS:
        k = f'nu{nu}_seed{s}'; v = r[k]
        print(f'{k:16s} {100 * v["einstein_rel"]:8.2f}% {v["hawking"]["slope"]:8.3f} {v["hawking"]["r2"]:6.3f} '
              f'{v["P1_flag_accuracy"]:8.4f} {v["P2_flag_mae"]:8.4f} {v["P3_wormhole_knn_recall"]:9.4f}')

base = [r[f'nu0.8_seed{s}'] for s in SEEDS]
ein_cap = 1.5 * max(b['einstein_rel'] for b in base)
out = {'einstein_cap': ein_cap, 'per_nu': {}}
gr_ok_all, degraded_any = True, False
for nu in NUS[1:]:
    runs = [r[f'nu{nu}_seed{s}'] for s in SEEDS]
    gr_ok = all(x['einstein_rel'] <= ein_cap for x in runs)
    worse = {}
    for m, hi in P.items():
        band = [b[m] for b in base]
        worst_base = min(band) if hi else max(band)
        worse[m] = all((x[m] < worst_base) if hi else (x[m] > worst_base) for x in runs)
    degraded = sum(worse.values()) >= 2
    out['per_nu'][nu] = {'gr_preserved': gr_ok, 'worse_than_nu08_band': worse, 'planck_degraded': degraded}
    gr_ok_all &= gr_ok; degraded_any |= degraded
out['gr_preserved'] = gr_ok_all
out['planck_degraded'] = degraded_any
out['verdict'] = 'SUPPORTED' if (gr_ok_all and degraded_any) else 'FALSIFIED'
out['literal_zero_einstein_claim'] = {nu: [r[f'nu{nu}_seed{s}']['einstein_rel'] for s in SEEDS] for nu in NUS[1:]}
json.dump(out, open(os.path.join(RES, 'nu_test_verdict.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
