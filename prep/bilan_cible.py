"""Bilan comptable 2027-2030 de l'effectif des RNP (remplace la Figure 7, frontière entrées x départs).
Effectif 2026 + entrées − départs − passages à la RP = effectif 2030. RP 35 000, dont 12 000 de l'étranger
(23 000 passages), comme dans le reste du rapport. Trois colonnes : rythme de 2026 ; cible de 250 000 au taux de
départ actuel (entrées requises) ; cible de 250 000 avec les entrées de 2026 (départs requis)."""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immsim2026 import immsim2026  # noqa: E402
from prep_intrants_2026 import stc_by_age  # noqa: E402

NPR25 = stc_by_age('17100158.csv', 2025)
E26, CIBLE = 90111, 250e3


def run(E, m=1.0):
    mod = immsim2026().calibrate(NPR25.values, 0.175, 0.061)
    mod.set_npr_policy(entry_cap=E, exit_multiplier=m)
    mod.set_imm_policy(entry_cap=35000, pr_from_abroad=12000)
    return mod.proj(2030)


def bisect(f, lo, hi, n=40):
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


stock = lambda mod: mod.npr_sim_age.loc[2030].sum()
E_req = bisect(lambda E: CIBLE - stock(run(E)), 0, E26)            # stock croît avec E
m_req = bisect(lambda m: stock(run(E26, m)) - CIBLE, 1.0, 10.0)    # stock décroît avec m
rows = {}
for nom, (E, m) in {'Rythme de 2026': (E26, 1.0), '250 000 : moins d\'entrées': (E_req, 1.0),
                    '250 000 : plus de départs': (E26, m_req)}.items():
    mod = run(E, m); f = mod.flows.loc[2027:2030]; p = mod.pop_sim.sum(axis=1)
    rows[nom] = {'effectif_2026': mod.npr_sim_age.loc[2026].sum(), 'entrees': f.rnp_entrees.sum(),
                 'departs': f.rnp_departs.sum(), 'passages': f.rnp_vers_rp.sum(), 'effectif_2030': stock(mod),
                 'entrees_an': E, 'departs_an': f.rnp_departs.mean(), 'taux_depart_2027': f.loc[2027, 'taux_depart'],
                 'taux_depart_moy': (f.rnp_departs / (f.rnp_departs / f.taux_depart)).mean(),
                 'pop_2030': p[2030], 'var_pop': p[2030] - p[2026], 'm': m}
out = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(out.round(3).to_string())
out.to_csv(os.path.join(ROOT, 'output', 'bilan_cible.csv'))
