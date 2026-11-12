"""
Marge du Québec : décomposition agrégée (sans âge) de l'effectif des RNP en trois stocks selon
le palier qui contrôle les entrées. Calcul d'appoint à la frontière (frontiere_pq.py).

  Québec  : PTET + étudiants (CAQ requis) et leurs familles
  PMI     : Programme de mobilité internationale (fédéral)
  Asile   : demandeurs d'asile et groupes apparentés (fédéral)

Sources : STC 17-10-0121 (stocks au 1er juil. 2026) ; MIFI, stocks au 31 déc. 2024 (répartition
des titulaires de permis) ; entrées 2025 imputées = premiers permis MIFI 2024 x (permis en vigueur
IRCC 2025 / 2024), demandes d'asile IRCC 2025, ramenées au total STC 2025 (116 435) ;
sortie de l'asile 11 % (tableau 4, 2025) ; sortie des titulaires de permis = résidu pour que la
sortie totale (départs + RP) soit de 23,6 %.
"""
PERMIS, ASILE, TOTAL = 341094, 190727, 531821
MIFI = {'PTET': 72405, 'PMI': 125200, 'PEE': 128090}
ENTREES_STC_2025 = 116435
S_ASILE, S_TOTAL = 0.11, 0.236

tot = sum(MIFI.values())
Q0 = PERMIS * (MIFI['PTET'] + MIFI['PEE']) / tot
P0 = PERMIS * MIFI['PMI'] / tot
A0 = ASILE

ptet, pmi, pee, asl = 22465 * 46650 / 56285, 28110 * 61990 / 83160, 45815 * 61075 / 72780, 34800
sc = ENTREES_STC_2025 / (ptet + pmi + pee + asl)
eQ, eP, eA = (ptet + pee) * sc, pmi * sc, asl * sc
s_permis = (S_TOTAL * TOTAL - S_ASILE * A0) / (Q0 + P0)


def projeter(eQ, eP, eA, kQ=1.0, annees=4):
    Q, P, A = Q0, P0, A0
    for _ in range(annees):
        Q = Q * (1 - min(s_permis * kQ, 1)) + eQ
        P = P * (1 - s_permis) + eP
        A = A * (1 - S_ASILE) + eA
    return Q, P, A


if __name__ == '__main__':
    print(f'Stocks 2026 : Québec {Q0:,.0f} ({Q0/TOTAL:.1%}), PMI {P0:,.0f} ({P0/TOTAL:.1%}), asile {A0:,.0f} ({A0/TOTAL:.1%})')
    print(f'Entrées 2025 : Québec {eQ:,.0f} ({eQ/ENTREES_STC_2025:.1%}), PMI {eP:,.0f}, asile {eA:,.0f}')
    print(f'Sortie des titulaires de permis (résidu) : {s_permis:.1%}')
    for lab, ea in [('asile au niveau 2025', eA), ('asile au rythme 2026 (~15 000)', 15000 * sc)]:
        for kQ, eq, nom in [(1, eQ, 'rien ne change'), (1, 0, 'Québec : zéro entrée'),
                            (2, 0, 'zéro entrée + départs doublés'), (100, 0, 'stock québécois vidé')]:
            Q, P, A = projeter(eq, eP, ea, kQ)
            print(f'{lab:32s} {nom:32s} 2030 : Québec {Q:8,.0f}  PMI {P:8,.0f}  asile {A:8,.0f}  total {Q+P+A:8,.0f}')
