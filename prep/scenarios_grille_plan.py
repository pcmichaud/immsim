"""
Grille Québec x fédéral avec le plan québécois en vigueur : 45 000 admissions RP par an, dont 65 % issues de
personnes déjà au Québec (15 750 admissions de l'étranger). Entrées de RNP = somme des blocs (marge_quebec.py) ;
projection immsim2026. Remplace, dans la section « Une cible d'effectif qui dépend d'Ottawa », la grille
calculée avec la trajectoire proposée (scenarios_degressif_grille.py), qui n'est motivée que plus loin.
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immsim2026 import immsim2026  # noqa: E402
from prep_intrants_2026 import stc_by_age  # noqa: E402
import marge_quebec as mq  # noqa: E402

NPR25 = stc_by_age('17100158.csv', 2025)
RP_PLAN = 45000
ETR_PLAN = round(45000 * 0.35)   # plan québécois : 65 % des admissions issues de personnes déjà au Québec
QUEBEC = {'Entrées de 2025': mq.eQ, 'Plafonds du plan québécois': mq.eQ * 0.90, 'Réduction de 30 %': mq.eQ * 0.70}
FED = {'Flux de 2025': (mq.eP, mq.eA), 'Asile au rythme de 2026': (mq.eP, 15000 * mq.sc),
       'Asile 2026 et PMI −30 %': (mq.eP * 0.70, 15000 * mq.sc)}

rows = []
for qn, eq in QUEBEC.items():
    for fn, (ep, ea) in FED.items():
        E = eq + ep + ea
        m = immsim2026().calibrate(NPR25.values, 0.175, 0.061)
        m.set_npr_policy(entry_cap=E)
        m.set_imm_policy(entry_cap=RP_PLAN, pr_from_abroad=ETR_PLAN)
        m.proj(2041)
        n = m.npr_sim_age.sum(axis=1); p = m.pop_sim.sum(axis=1)
        rows.append({'quebec': qn, 'federal': fn, 'entrees': round(E), 'rnp_2030': round(n[2030]),
                     'rnp_2035': round(n[2035]), 'rnp_2041': round(n[2041]),
                     'var_pop_26_30': round(p[2030] - p[2026]), 'pop_2041': round(p[2041]),
                     'part_rnp_2030': round(n[2030] / p[2030] * 100, 1)})
out = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(out.to_string())
out.to_csv(os.path.join(ROOT, 'output', 'grille_plan45000.csv'), index=False)
