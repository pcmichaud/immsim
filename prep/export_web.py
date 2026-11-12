"""Exporte les intrants d'immsim2026 pour le simulateur web (docs/intrants.json) et des résultats de référence
calculés en Python (tests/reference_web.json), qui servent à vérifier que la version JavaScript donne les mêmes
résultats. Usage : python prep/export_web.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immsim2026 import immsim2026 as M  # noqa: E402
from prep_intrants_2026 import stc_by_age  # noqa: E402
import marge_quebec as mq  # noqa: E402

# mêmes paramètres que prep/scenarios_70k.py (non importé : ce script exécute les scénarios)
P_RP = dict(e0=0.60, he=6, r0=0.50, hr=10)
P_NPR = dict(e0=0.80, he=10, r0=0.85, hr=3)
P_ETU = dict(e0=0.50, he=6, r0=0.60, hr=6)
P_ASI = dict(e0=0.50, he=8, r0=0.70, hr=8)
MIX = {'travail': (217501 + 24191) / 531821, 'etudiants': (58401 + 41001) / 531821, 'asile': 190727 / 531821}
proj = pd.read_csv(os.path.join(ROOT, 'data', 'stc', '17100057_proj_canada_qc.csv'))
obs = pd.read_csv(os.path.join(ROOT, 'data', 'stc', 'population_canada_qc_juillet.csv')).set_index('annee')
can = {}
for sc_, lab in [(3, 'M1'), (2, 'LG'), (7, 'HG')]:
    s_ = proj[(proj.geo == 1) & (proj.scenario == sc_)].set_index('annee').pop_milliers * 1000
    can[lab] = s_ / s_.loc[2026] * obs.loc[2026, 'canada']

END = 2060
NPR25 = stc_by_age('17100158.csv', 2025)
m = M().calibrate(NPR25.values, 0.175, 0.061)
YRS = list(range(2026, END + 1))


def r(a, k=6):
    """Arrondi à k chiffres significatifs, pour alléger le JSON."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = np.where(a == 0, 0, np.floor(np.log10(np.abs(np.where(a == 0, 1, a)))))
    f = 10.0 ** (k - 1 - mag)
    return (np.round(a * f) / f).tolist()


yrs_rates = list(range(2027, END + 1))
intrants = {
    'annee_depart': 2026, 'annee_fin': END,
    'mx': {t: r(m.mx.loc[t].values) for t in yrs_rates},
    'ex': {t: r(m.ex.loc[t].values) for t in yrs_rates},
    'interpx': {t: r(m.interpx.loc[t].values) for t in yrs_rates},
    'share_ix': {t: r(m.share_ix.loc[t].values) for t in yrs_rates},
    'bx': {t: float(m.bx.loc[t]) for t in yrs_rates},
    'npr_exit_rates': r(m.npr_exit_rates),
    'npr_apply_rates': r(m.npr_apply_rates),
    'npr_entry_profile': r(m.npr_entry_profile),
    'start_pop': r(m.start_pop, 9),
    'start_npr': r(m.start_npr, 9),
    'start_pr': r(m.start_pr, 9),
    'exit_factor': m.exit_factor, 'apply_factor': m.apply_factor,
    # entrées de 2025 par palier (modèle à trois groupes) et facteur de mise à l'échelle des demandes d'asile
    'entrees_2025': {'quebec': mq.eQ, 'pmi': mq.eP, 'asile': mq.eA}, 'echelle_asile': mq.sc,
    'profils': {'rp': P_RP, 'travail': P_NPR, 'etudiants': P_ETU, 'asile': P_ASI}, 'mix': MIX,
    'canada': {k: {int(t): float(v) for t, v in s.reindex(YRS).items()} for k, s in can.items()},
    'isq2019': {2026: 8830200, 2031: 9039500, 2036: 9209300, 2041: 9350200},
}
with open(os.path.join(ROOT, 'docs', 'intrants.json'), 'w') as f:
    json.dump(intrants, f, separators=(',', ':'))


# ---------------------------------------------------------------- résultats de référence
def prof(p, speed=1.0):
    q = dict(p); q['he'] *= speed; q['hr'] *= speed
    return M.profile(**q)


def prof_npr(speed=1.0):
    return sum(MIX[k] * prof(p, speed) for k, p in (('travail', P_NPR), ('etudiants', P_ETU), ('asile', P_ASI)))


def run(rp, etr, entrees=90111.0, kappa=1.0, vitesse=1.0):
    mm = M().calibrate(NPR25.values, 0.175, 0.061)
    mm.set_npr_policy(entry_cap=entrees, exit_multiplier=kappa)
    mm.set_imm_policy(entry_cap=rp, pr_from_abroad=etr)
    mm.proj(END)
    sp = lambda t: 1.0 if t <= 2026 else max(1 / vitesse, 1 - (1 - 1 / vitesse) * (t - 2026) / 5)
    eff = np.array([mm.effective_workers_by_status(t, prof_npr(sp(t)), prof(P_RP, sp(t))) for t in YRS])
    pop = mm.pop_sim.sum(axis=1).loc[YRS].values
    y = eff / pop
    return {'rnp': mm.npr_sim_age.sum(axis=1).loc[YRS].round(1).tolist(), 'pop': pop.round(1).tolist(),
            'y': ((y / y[0] - 1) * 100).round(4).tolist(),
            'passages': [0.0] + mm.flows.loc[2027:END, 'rnp_vers_rp'].round(1).tolist()}


trans = {2027 + k: 70000 - 2500 * k for k in range(9)}
cas = {
    'transition': dict(args=dict(rp=trans, etr=20000), params=dict(rp0=70000, rp1=50000, an1=2035, etr=20000)),
    'plan45': dict(args=dict(rp=45000, etr=15750), params=dict(rp0=45000, rp1=45000, an1=2027, etr=15750)),
    'cible35': dict(args=dict(rp=35000, etr=12000), params=dict(rp0=35000, rp1=35000, an1=2027, etr=12000)),
    'cible35_vitesse': dict(args=dict(rp=35000, etr=12000, vitesse=2.0),
                            params=dict(rp0=35000, rp1=35000, an1=2027, etr=12000, vitesse=2.0)),
    'test_leviers': dict(args=dict(rp={2027 + k: 60000 - 2000 * k for k in range(6)}, etr=18000, entrees=70000.0,
                                   kappa=1.5, vitesse=1.5),
                         params=dict(rp0=60000, rp1=50000, an1=2032, etr=18000, entrees=70000.0, kappa=1.5,
                                     vitesse=1.5)),
}
ref = {k: {'params': v['params'], 'resultats': run(**v['args'])} for k, v in cas.items()}
os.makedirs(os.path.join(ROOT, 'tests'), exist_ok=True)
with open(os.path.join(ROOT, 'tests', 'reference_web.json'), 'w') as f:
    json.dump(ref, f, separators=(',', ':'))
print('intrants :', os.path.getsize(os.path.join(ROOT, 'docs', 'intrants.json')) // 1024, 'Ko')
print('2030 RNP :', {k: round(v['resultats']['rnp'][4]) for k, v in ref.items()})
