"""
Préparation des intrants d'immsim calés sur la mise à jour 2026 des perspectives de l'ISQ
(scénario Référence A2026, 9 juillet 2026) et sur les estimations de Statistique Canada
au 1er juillet 2026.

Les composantes par âge de l'ISQ ne sont pas publiques. On garde donc les profils par âge de
l'édition 2024 (data/tx_isq.xlsx) et on recale chaque composante, année par année, sur les
totaux du fichier de composantes ISQ 2026 :
  - naissances : tx_naissance recalculé = naissances ISQ / population ISQ 15-45 ans ;
  - décès : mx(2024) x facteur annuel, pour reproduire les décès ISQ ;
  - émigration nette et migration interprovinciale nette : taux(2024) x facteur annuel,
    appliqués à la population hors RNP, pour reproduire les totaux ISQ ;
  - immigrants permanents : total ISQ, répartition par âge de l'édition 2024 ;
  - RNP : effectif total ISQ, répartition par âge des RNP de STC au 1er juillet 2026.

Convention : l'année t du modèle est la population au 1er juillet t. Les flux de l'année t
(dans le modèle) sont ceux de la période ISQ (t-1, t), soit la ligne « Année = t-1 » du fichier
de composantes.

Sortie : data/tx_isq_2026.xlsx (mêmes feuilles que tx_isq.xlsx, plus base_2026 et isq2026_compo).
"""
import csv
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data')
AGES = list(range(0, 101))
LAST_YEAR = 2070


def stc_by_age(fname, year, geo='Québec'):
    """Effectif par âge simple (0 à 99, 100+) au 1er juillet, Total - genre."""
    out = {}
    with open(os.path.join(DATA, 'stc', fname), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f, delimiter=';'):
            if r['GÉO'] != geo or r['PÉRIODE DE RÉFÉRENCE'] != str(year) or r['Genre'] != 'Total - genre':
                continue
            g = r["Groupe d'âge"]
            if g == '100 ans et plus':
                out[100] = float(r['VALEUR'])
            elif g.split()[-1] in ('an', 'ans') and len(g.split()) == 2 and g.split()[0].isdigit():
                out[int(g.split()[0])] = float(r['VALEUR'])
    s = pd.Series(out).reindex(AGES)
    assert s.notna().all(), f'âges manquants dans {fname}'
    return s


def isq_compo():
    c = pd.read_excel(os.path.join(DATA, 'isq2026', 'isq2026_compo-qc-base.xlsx'), header=None)
    c = c[c[0] == 'Référence (A2026)']
    cols = {2: 'annee', 3: 'pop_t', 5: 'naissances', 6: 'isf', 8: 'deces', 15: 'immigrants',
            16: 'emigration_nette', 17: 'rnp_t', 18: 'rnp_t1', 19: 'solde_rnp',
            24: 'solde_interprov', 30: 'pop_t1'}
    c = c[list(cols)].rename(columns=cols).apply(pd.to_numeric, errors='coerce')  # « ... » = non disponible
    c['annee'] = c['annee'].astype(int)
    return c.set_index('annee')


def isq_pop_by_age():
    d = pd.read_excel(os.path.join(DATA, 'isq2026', 'isq2026_pop-as-qc-base.xlsx'), "Années d'âge", header=None)
    d = d[(d[0] == 'Référence (A2026)') & (d[4] == 3)]
    p = d[[2] + list(range(12, 113))].copy()
    p.columns = ['year'] + AGES
    p['year'] = p['year'].astype(int)
    return p.set_index('year').astype(float)


def main():
    old = pd.ExcelFile(os.path.join(DATA, 'tx_isq.xlsx'))
    tx = pd.read_excel(old, 'tx_by_age').drop(columns=['Unnamed: 0'], errors='ignore')
    compo = isq_compo()
    pop_isq = isq_pop_by_age()

    pop26 = stc_by_age('17100005.csv', 2026)
    npr26 = stc_by_age('17100158.csv', 2026)
    npr_profile = npr26 / npr26.sum()

    rows = []
    for t in range(2026, LAST_YEAR + 1):
        src = tx[tx.year == min(t, tx.year.max())].set_index('age').reindex(AGES)
        p_prev = pop_isq.loc[t - 1]                                   # population ISQ au 1er juillet t-1
        aged = pd.Series(np.r_[0.0, p_prev.values[:-1]], index=AGES)  # vieillie d'un an
        aged.iloc[-1] += p_prev.iloc[-1]                               # les 100+ restent 100+
        npr_prev = compo.loc[t - 1, 'rnp_t'] * npr_profile
        npr_aged = pd.Series(np.r_[0.0, npr_prev.values[:-1]], index=AGES)
        non_npr = (aged - npr_aged).clip(lower=0)

        f_mx = compo.loc[t - 1, 'deces'] / (src['mx'] * aged).sum()
        f_ex = compo.loc[t - 1, 'emigration_nette'] / (src['tx_net_emmigrant'] * non_npr).sum()
        f_ip = compo.loc[t - 1, 'solde_interprov'] / (src['tx_net_interp_migration'] * non_npr).sum()

        prop = src['prop_immig'].fillna(0)
        prop = prop / prop.sum()
        for a in AGES:
            rows.append({
                'year': t, 'age': a,
                'population': pop_isq.loc[t, a] if t in pop_isq.index else np.nan,
                'mx': src.loc[a, 'mx'] * f_mx,
                'tx_net_emmigrant': src.loc[a, 'tx_net_emmigrant'] * f_ex,
                'tx_net_interp_migration': src.loc[a, 'tx_net_interp_migration'] * f_ip,
                'NPR': compo.loc[t, 'rnp_t'] * npr_profile[a] if t in compo.index else np.nan,
                'tot_immig': compo.loc[t - 1, 'immigrants'],
                'prop_immig': prop[a],
            })
    new = pd.DataFrame(rows)

    births = []
    for t in range(2026, LAST_YEAR + 1):
        p15_45 = pop_isq.loc[t - 1, list(range(15, 46))].sum()
        births.append({'year': t, 'tx_naissance': compo.loc[t - 1, 'naissances'] / p15_45})
    births = pd.DataFrame(births)

    base = pd.DataFrame({'age': AGES, 'population': pop26.values, 'npr': npr26.values})

    out = os.path.join(DATA, 'tx_isq_2026.xlsx')
    with pd.ExcelWriter(out) as w:
        new.to_excel(w, sheet_name='tx_by_age', index=False)
        births.to_excel(w, sheet_name='tx_naissance', index=False)
        for s in ['npr_exits', 'npr_apply_pr', 'npr_proportion', 'immig_pop']:
            pd.read_excel(old, s, index_col=0).to_excel(w, sheet_name=s)
        base.to_excel(w, sheet_name='base_2026', index=False)
        compo.reset_index().to_excel(w, sheet_name='isq2026_compo', index=False)
    print('écrit :', out)
    print('population STC 2026 :', int(pop26.sum()), '| RNP STC 2026 :', int(npr26.sum()))


if __name__ == '__main__':
    main()
