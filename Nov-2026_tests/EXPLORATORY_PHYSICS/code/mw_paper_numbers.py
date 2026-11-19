#!/usr/bin/env python3
"""Write ../papers/numbers.tex (LaTeX macros) from the results JSON, so every number in the
three papers comes straight from a run."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
js = json.load(open(os.path.join(RES, 'exact_geometry.json')))
be = json.load(open(os.path.join(RES, 'berry_check.json')))
A, B = js['A_full_prior'], js['B_full_noprior']


def sci(x, p=1):
    m, e = f'{x:.{p}e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def fx(x, p=2):
    return f'{x:.{p}f}'


L = []
def mac(name, val): L.append(f'\\newcommand{{\\{name}}}{{{val}}}')

for tag, r in [('A', A), ('B', B)]:
    mac(f'Act{tag}', r['active_dims']); mac(f'Npts{tag}', r['points'])
    mac(f'Gauss{tag}', sci(r['gauss_rel_err']['median'])); mac(f'GaussHi{tag}', sci(r['gauss_rel_err']['p95']))
    mac(f'Bian{tag}', sci(r['bianchi_rel']['median'])); mac(f'BianHi{tag}', sci(r['bianchi_rel']['p95']))
    mac(f'Asym{tag}', sci(r['ricci_asymmetry_rel']['median'])); mac(f'Split{tag}', sci(r['source_split_rel_err']['median']))
    mac(f'RconnVsGauss{tag}', sci(r['R_connection_vs_R_gauss_rel']['median']))
    mac(f'EigMin{tag}', sci(r['metric_eig_min']['median'], 1)); mac(f'EigMax{tag}', fx(r['metric_eig_max']['median']))
    mac(f'Cond{tag}', f"{r['metric_condition']['median']:.0f}")
    mac(f'Legacy{tag}', sci(r['legacy_R_proxy']['median'], 2))
    mac(f'Rmed{tag}', fx(r['ricci_scalar']['median'], 1)); mac(f'Rlo{tag}', fx(r['ricci_scalar']['p05'], 0))
    mac(f'Rhi{tag}', fx(r['ricci_scalar']['p95'], 0)); mac(f'AbsR{tag}', fx(r['abs_ricci_scalar']['median'], 0))
    mac(f'Kmed{tag}', sci(r['sectional_K']['median'], 1)); mac(f'Klo{tag}', fx(r['sectional_K']['p05'], 3))
    mac(f'Khi{tag}', fx(r['sectional_K']['p95'], 3)); mac(f'KNeg{tag}', f"{100 * r['sectional_K_neg_frac']:.0f}")
    g = r['geodesic']
    mac(f'GeoMs{tag}', fx(g['M_start'])); mac(f'GeoMe{tag}', fx(g['M_end']))
    mac(f'GeoLen{tag}', fx(g['length_geodesic'], 3)); mac(f'LineLen{tag}', fx(g['length_line'], 3))
    mac(f'GeoNormal{tag}', f"{100 * (1 - g['tangential_accel_fraction_geodesic']['median']):.0f}")
    mac(f'LineNormal{tag}', f"{100 * (1 - g['tangential_accel_fraction_line']['median']):.0f}")
    mac(f'GeoHawk{tag}', fx(g['geodesic']['hawking_abs_8piMTH_minus_1_mean'], 3))
    mac(f'LineHawk{tag}', fx(g['straight_line']['hawking_abs_8piMTH_minus_1_mean'], 3))
    mac(f'GeoEin{tag}', fx(g['geodesic']['einstein_rel_resid'], 3))
    mac(f'LineEin{tag}', fx(g['straight_line']['einstein_rel_resid'], 3))
    mac(f'GeoSpeed{tag}', sci(max(g['speed_cv'], 1e-6), 0))
    for sc, v in r['per_scenario'].items():
        k = sc.replace('_', '').capitalize()
        mac(f'R{k}{tag}', fx(v['R']['median'], 1))
        for grp, s in v['source_share_mean'].items():
            gk = {'coordinates': 'Co', 'gravity': 'Gr', 'quantum': 'Qu', 'horizon': 'Ho', 'planck_topology': 'Pl'}[grp]
            mac(f'Sh{gk}{k}{tag}', '$<$1' if 100 * s < 0.5 else f'{100 * s:.0f}')
    mac(f'Berry{tag}', sci(be[f'{"A_full_prior" if tag == "A" else "B_full_noprior"}']['F_over_dA_max'], 0))
open(os.path.join(HERE, '..', 'papers', 'numbers.tex'), 'w').write('\n'.join(L) + '\n')
print(f'wrote papers/numbers.tex ({len(L)} macros)')
