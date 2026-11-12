"""Trois scénarios (35 000, 45 000, transition 70 000 → 50 000 en 2035), 2026-2060 : effectif des RNP, population,
passages à la RP, structure par âge, effet de composition sur le PIB par habitant (profils d'intégration par statut,
comme integration_statut.py) et poids du Québec dans le Canada (projection M1 de Statistique Canada, 17-10-0057,
recalée sur la population canadienne observée de 2026). Sorties : output/scenarios_70k_*.csv."""
import csv
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immsim2026 import immsim2026 as M  # noqa: E402
from prep_intrants_2026 import stc_by_age  # noqa: E402
from chemins_70k import SCENARIOS, ENTREES  # noqa: E402

NPR25 = stc_by_age('17100158.csv', 2025)
END = 2060
P_RP = dict(e0=0.60, he=6, r0=0.50, hr=10)
P_NPR = dict(e0=0.80, he=10, r0=0.85, hr=3)
P_ETU = dict(e0=0.50, he=6, r0=0.60, hr=6)
P_ASI = dict(e0=0.50, he=8, r0=0.70, hr=8)
MIX = {'travail': (217501 + 24191) / 531821, 'etudiants': (58401 + 41001) / 531821, 'asile': 190727 / 531821}


def prof(p, speed=1.0):
    q = dict(p); q['he'] *= speed; q['hr'] *= speed
    return M.profile(**q)


def prof_npr(speed=1.0):
    return MIX['travail'] * prof(P_NPR, speed) + MIX['etudiants'] * prof(P_ETU, speed) + MIX['asile'] * prof(P_ASI, speed)


# Canada : projection M1 (et LG, HG) recalée sur la population observée de juillet 2026
proj = pd.read_csv(os.path.join(ROOT, 'data', 'stc', '17100057_proj_canada_qc.csv'))
obs = pd.read_csv(os.path.join(ROOT, 'data', 'stc', 'population_canada_qc_juillet.csv')).set_index('annee')
can = {}
for sc, lab in [(3, 'M1'), (2, 'LG'), (7, 'HG')]:
    s = proj[(proj.geo == 1) & (proj.scenario == sc)].set_index('annee').pop_milliers * 1000
    can[lab] = s / s.loc[2026] * obs.loc[2026, 'canada']
qc_m1 = proj[(proj.geo == 6) & (proj.scenario == 3)].set_index('annee').pop_milliers * 1000

traj, resume = {}, []
for nom, (rp, etr) in SCENARIOS.items():
    for fast in (False, True):
        m = M().calibrate(NPR25.values, 0.175, 0.061)
        m.set_npr_policy(entry_cap=ENTREES)
        m.set_imm_policy(entry_cap=rp, pr_from_abroad=etr)
        m.proj(END)
        yrs = list(range(2026, END + 1))
        sp = lambda t: 1.0 if (not fast or t <= 2026) else max(0.5, 1 - 0.5 * (t - 2026) / 5)
        eff = pd.Series({t: m.effective_workers_by_status(t, prof_npr(sp(t)), prof(P_RP, sp(t))) for t in yrs})
        pop = m.pop_sim.sum(axis=1).loc[yrs]
        wa = m.pop_sim.loc[yrs, 15:64].sum(axis=1)
        old = m.pop_sim.loc[yrs, 65:].sum(axis=1)
        n = m.npr_sim_age.sum(axis=1).loc[yrs]
        f = m.flows
        k = nom + (' + vitesse' if fast else '')
        traj[k] = pd.DataFrame({'rnp': n, 'pop': pop, 'wa': wa, 'old': old, 'eff': eff, 'y': eff / pop,
                                'part_qc_M1': pop / can['M1'].reindex(yrs) * 100,
                                'part_qc_LG': pop / can['LG'].reindex(yrs) * 100,
                                'part_qc_HG': pop / can['HG'].reindex(yrs) * 100,
                                'passages': f.rnp_vers_rp.reindex(yrs), 'etranger': f.rp_etranger.reindex(yrs),
                                'taux_rp': f.taux_rp.reindex(yrs), 'departs': f.rnp_departs.reindex(yrs)})
        if fast:
            continue
        t = traj[k]
        resume.append({'scenario': nom,
                       'adm_27_35': round(f.loc[2027:2035, 'rp_total'].sum()), 'passages_27_35': round(f.loc[2027:2035, 'rnp_vers_rp'].sum()),
                       'passages_27_30': round(f.loc[2027:2030, 'rnp_vers_rp'].sum()),
                       'taux_rp_max': round(f.loc[2027:2035, 'taux_rp'].max(), 3), 'passages_2027': round(f.loc[2027, 'rnp_vers_rp']),
                       **{f'rnp_{y}': round(n[y]) for y in (2028, 2030, 2035, 2041, 2050)},
                       **{f'pop_{y}': round(pop[y]) for y in (2030, 2035, 2041, 2050, 2060)},
                       'var_pop_26_30': round(pop[2030] - pop[2026]),
                       **{f'part15_64_{y}': round(wa[y] / pop[y] * 100, 2) for y in (2026, 2030, 2041, 2060)},
                       **{f'dep65_{y}': round(old[y] / wa[y] * 100, 1) for y in (2026, 2041, 2060)},
                       **{f'y_cum_{y}': round((t.y[y] / t.y[2026] - 1) * 100, 2) for y in (2030, 2035, 2040, 2050, 2060)},
                       **{f'part_qc_{y}': round(t.part_qc_M1[y], 2) for y in (2030, 2041, 2050, 2060)}})
out = os.path.join(ROOT, 'output')
pd.concat(traj, axis=1).to_csv(os.path.join(out, 'scenarios_70k_trajectoires.csv'))
r = pd.DataFrame(resume).set_index('scenario').T
pd.set_option('display.width', 220)
print(r.to_string())
r.to_csv(os.path.join(out, 'scenarios_70k_resume.csv'))
v = {k: (traj[k].y / traj[k].y[2026] - 1) * 100 for k in traj}
print(pd.DataFrame(v).loc[[2030, 2035, 2040, 2050, 2060]].round(2).to_string())
print('Part QC observée 2026 :', round(obs.loc[2026, 'quebec'] / obs.loc[2026, 'canada'] * 100, 2),
      '| STC M1 Québec/Canada :', {y: round(qc_m1[y] / (proj[(proj.geo == 1) & (proj.scenario == 3)].set_index('annee').pop_milliers[y] * 1000) * 100, 2) for y in (2030, 2041, 2050, 2060)})
