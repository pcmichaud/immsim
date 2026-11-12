"""Test de non-régression : reproduit les chiffres du Tableau 3 du rapport « Atterrir en douceur » (novembre 2026).
Usage : python tests/test_regression.py"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'prep'))
from immsim2026 import immsim2026 as M  # noqa: E402
from prep_intrants_2026 import stc_by_age  # noqa: E402
from chemins_70k import SCENARIOS, ENTREES  # noqa: E402

# effectif des RNP en 2030 et 2041, population en 2041 (Tableau 3)
ATTENDU = {'35 000 : cible du gouvernement': (454567, 407140, 9014268),
           '45 000 : plan 2026-2029': (435638, 374922, 9138983),
           'Transition : 70 000 → 50 000': (385968, 362119, 9301483)}

NPR25 = stc_by_age('17100158.csv', 2025)
ok = True
for nom, (rp, etr) in SCENARIOS.items():
    m = M().calibrate(NPR25.values, 0.175, 0.061)
    m.set_npr_policy(entry_cap=ENTREES)
    m.set_imm_policy(entry_cap=rp, pr_from_abroad=etr)
    m.proj(2041)
    n = m.npr_sim_age.sum(axis=1); p = m.pop_sim.sum(axis=1)
    obtenu = (round(n[2030]), round(n[2041]), round(p[2041]))
    bon = all(abs(a - b) <= 1 for a, b in zip(obtenu, ATTENDU[nom]))
    ok &= bon
    print(f"{'OK ' if bon else 'ÉCHEC'} {nom:28s} RNP 2030 {obtenu[0]:>8,} | RNP 2041 {obtenu[1]:>8,} | pop 2041 {obtenu[2]:>10,}")
sys.exit(0 if ok else 1)
