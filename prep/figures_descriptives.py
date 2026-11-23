"""Figures descriptives du rapport (hors scénarios projetés) -> output/figures/.

- f2_immigration_permanente.png : entrées d'immigrants permanents pour 1 000 habitants (OCDE, 2024) ;
- f3_effectif_rnp.png : effectif des RNP au Québec, titulaires de permis et demandeurs d'asile ;
- f4b_population_observee.png : population observée et prévue par l'ISQ en 2019 ;
- f6_marge_quebec.png : effectif des RNP en 2030 selon les choix du Québec (modèle simplifié, annexe C).

Sources : OCDE (DSD_MIG_INT@DF_MIG_INT_PER) et Banque mondiale (population) pour la Figure 2 ;
Statistique Canada, tableaux 17-10-0121-01 et 17-10-0009-01 ; ISQ, Perspectives démographiques 2016-2066,
édition 2019, scénario Référence (tableau A2.2).
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marge_quebec as mq  # noqa: E402

OUT = os.path.join(ROOT, 'output', 'figures'); os.makedirs(OUT, exist_ok=True)
BLEU, GRIS, GRIS_C, NOIR = '#1f5fa8', '#8c8c8c', '#c8c8c8', '#222222'
plt.rcParams.update({'font.family': 'Palatino', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6',
                     'axes.axisbelow': True, 'figure.dpi': 300})
fr = lambda v, d=0: f'{v:,.{d}f}'.replace(',', ' ').replace('.', ',')  # noqa: E731


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(fig)


# --- Figure 2 : immigration permanente pour 1 000 habitants (2024). Québec : admissions et population au 1er juillet.
perm = [('Suisse', 15.1), ('Canada', 11.7), ('Nouvelle-Zélande', 10.0), ('Australie', 8.8), ('Allemagne', 7.0),
        ('Québec 2024 (59 440 RP)', 6.6), ('Royaume-Uni', 6.3), ('Québec, 45 000 RP', 5.0), ("Ensemble de l'OCDE", 4.7),
        ('France', 4.3), ('États-Unis', 4.2), ('Québec, 35 000 RP', 3.9)]
fig, ax = plt.subplots(figsize=(6.3, 3.6))
for i, (p, v) in enumerate(perm):
    ax.barh(i, v, color=BLEU if p.startswith('Québec') else (GRIS if 'OCDE' in p else GRIS_C),
            alpha=.6 if '35 000' in p else 1)
    ax.text(v + .2, i, fr(v, 1), va='center', fontsize=9)
ax.set_yticks(range(len(perm)), [p[0] for p in perm]); ax.invert_yaxis()
ax.set_xlabel("Entrées d'immigrants permanents pour 1 000 habitants (2024)")
save(fig, 'f2_immigration_permanente.png')

# --- Figure 3 : effectif des RNP, titulaires de permis et demandeurs d'asile
s = {}
with open(os.path.join(ROOT, 'data', 'stc', '17100121.csv'), encoding='utf-8-sig') as f:
    for r in csv.DictReader(f, delimiter=';'):
        if r['GÉO'] == 'Québec' and r['PÉRIODE DE RÉFÉRENCE'] >= '2021-07':
            s.setdefault(r['PÉRIODE DE RÉFÉRENCE'][:7], {})[r['Types de résidents non permanents']] = int(r['VALEUR'])
q = sorted(s)
perm_ = [s[k]['Total, titulaires de permis et membres de leur famille'] for k in q]
asl = [s[k]["Total, demandeurs d'asile, personnes protégées et groupes apparentés"] for k in q]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
x = np.arange(len(q))
ax.bar(x, np.array(perm_) / 1e3, color=GRIS_C, label='Titulaires de permis et famille')
ax.bar(x, np.array(asl) / 1e3, bottom=np.array(perm_) / 1e3, color=BLEU, label="Demandeurs d'asile")
ax.set_xticks([i for i, k in enumerate(q) if k.endswith('-01')], [k[:4] for k in q if k.endswith('-01')])
ax.set_ylabel('milliers'); ax.legend(frameon=False, loc='upper left')
save(fig, 'f3_effectif_rnp.png')

# --- Population observée et prévue par l'ISQ en 2019
obs = {}
with open(os.path.join(ROOT, 'data', 'stc', '17100009.csv'), encoding='utf-8-sig') as f:
    for r in csv.DictReader(f, delimiter=';'):
        if r['GÉO'] == 'Québec' and r['PÉRIODE DE RÉFÉRENCE'].endswith('-07-01'):
            obs[int(r['PÉRIODE DE RÉFÉRENCE'][:4])] = int(r['VALEUR'])
obs = pd.Series(obs).loc[2016:2026]
isq19 = pd.Series({2016: 8225948, 2021: 8568174, 2026: 8830192, 2031: 9039495})
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.plot(obs.index, obs / 1e6, color=NOIR, lw=2, label='Observée')
ax.plot(isq19.index, isq19 / 1e6, color=BLEU, ls='--', lw=1.8, label="Prévue par l'ISQ en 2019")
ax.annotate('', xy=(2026, obs.loc[2026] / 1e6), xytext=(2026, isq19.loc[2026] / 1e6),
            arrowprops=dict(arrowstyle='<->', color=GRIS, lw=1))
ax.text(2026.2, (obs.loc[2026] + isq19.loc[2026]) / 2e6, '+' + fr(round(obs.loc[2026] - isq19.loc[2026], -3)),
        fontsize=8.5, va='center')
ax.set_ylabel("millions d'habitants"); ax.legend(frameon=False, fontsize=8.5, loc='upper left')
save(fig, 'f4b_population_observee.png')

# --- Marge du Québec (flux fédéraux inchangés), modèle simplifié à trois groupes
cas = [('Rien ne\nchange', 1, mq.eQ), ('Zéro\nentrée', 1, 0), ('Zéro entrée,\ndéparts ×2', 2, 0), ('Stock\nvidé', 100, 0)]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True)
for ax, (tit, ea) in zip(axs, [('Asile au niveau de 2025', mq.eA), ('Asile au rythme de 2026', 15000 * mq.sc)]):
    for i, (lab, k, eq) in enumerate(cas):
        Q, P, A = mq.projeter(eq, mq.eP, ea, k)
        ax.bar(i, A / 1e3, color=GRIS_C); ax.bar(i, P / 1e3, bottom=A / 1e3, color=GRIS)
        ax.bar(i, Q / 1e3, bottom=(A + P) / 1e3, color=BLEU)
        ax.text(i, (A + P + Q) / 1e3 + 8, fr((A + P + Q) / 1e3), ha='center', fontsize=8.5)
    ax.axhline(250, color=NOIR, ls='--', lw=1); ax.set_title(tit, fontsize=10)
    ax.set_xticks(range(4), [c[0] for c in cas], fontsize=8)
axs[0].set_ylabel('Effectif des RNP en 2030 (milliers)')
axs[0].bar(0, 0, color=BLEU, label='PTET et étudiants (Québec)'); axs[0].bar(0, 0, color=GRIS, label='PMI (fédéral)')
axs[0].bar(0, 0, color=GRIS_C, label='Asile (fédéral)')
axs[0].plot([], [], color=NOIR, ls='--', lw=1, label='Cible du gouvernement : 250 000')
axs[0].legend(frameon=False, fontsize=7.5, loc='upper right')
save(fig, 'f6_marge_quebec.png')
print('Figures écrites dans', os.path.abspath(OUT))
