"""Trajectoires d'admissions RP des trois scénarios (2026-11-12) : 35 000, 45 000 et transition.

Transition (révisé le 2026-11-12) : 70 000 en 2027, baisse linéaire jusqu'à 50 000 en 2035 (−2 500 par an), puis 50 000.
Admissions de l'étranger maintenues à 20 000 par an (soit 40 % des 50 000 de long terme, au-dessus du plan CAQ) : la
hausse est comblée par des RNP déjà au Québec, sans cannibaliser les admissions de l'étranger.
(Version précédente, fin à 40 000 avec 16 000 de l'étranger : chemins_70k_v1_fin40000.py.)
CAQ : plan 2026-2029 prolongé, 45 000 par an, dont 65 % issues de personnes déjà au Québec (15 750 de l'étranger).
PQ : 35 000 par an dès 2027 ; hors RNP, seulement la réunification familiale et les réfugiés (12 000 de l'étranger).
"""
PROPOSE = {2027 + k: 70000 - 2500 * k for k in range(9)}          # 70 000 (2027) ... 50 000 (2035)
ETR_PROPOSE = 20000
CAQ, ETR_CAQ = 45000, 15750
PQ, ETR_PQ = 35000, 12000
ENTREES = 90111      # entrées de RNP : plafonds du plan québécois, asile au rythme de 2026, PMI inchangé
SCENARIOS = {'35 000 : cible du gouvernement': (PQ, ETR_PQ), '45 000 : plan 2026-2029': (CAQ, ETR_CAQ), 'Transition : 70 000 → 50 000': (PROPOSE, ETR_PROPOSE)}
