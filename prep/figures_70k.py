"""Figures des trois scénarios (35 000, 45 000, transition 70 000 → 50 000) à partir de output/scenarios_70k_trajectoires.csv."""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'output', 'figures'); os.makedirs(OUT, exist_ok=True)
BLEU, GRIS, GRIS_C, NOIR, ROUGE = '#1f5fa8', '#8c8c8c', '#c8c8c8', '#222222', '#b2182b'
plt.rcParams.update({'font.family': 'Palatino', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'axes.axisbelow': True})
fr = lambda v, d=0: f'{v:,.{d}f}'.replace(',', ' ').replace('.', ',')
t = pd.read_csv(os.path.join(ROOT, 'output', 'scenarios_70k_trajectoires.csv'), header=[0, 1], index_col=0)
PQ, CAQ, PR = '35 000 : cible du gouvernement', '45 000 : plan 2026-2029', 'Transition : 70 000 → 50 000'
STY = {PR: (BLEU, '-', 2.4, 'Transition : 70 000, puis 50 000 en 2035'), CAQ: (GRIS, '--', 1.7, '45 000 : plan 2026-2029'),
       PQ: (GRIS, ':', 1.9, '35 000 : cible du gouvernement')}


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=300); plt.close(fig)


# Figure 1 : RNP et population, 2026-2041
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2))
for k, (c, ls, lw, lab) in STY.items():
    s = t[k].loc[2026:2041]
    axs[0].plot(s.index, s.rnp / 1e3, color=c, ls=ls, lw=lw, label=lab)
    axs[1].plot(s.index, s['pop'] / 1e6, color=c, ls=ls, lw=lw, label=lab)
_p19 = pd.Series({2026: 8830200, 2031: 9039500, 2036: 9209300, 2041: 9350200})
_y = np.arange(2026, 2042)
axs[1].plot(_y, np.interp(_y, _p19.index, _p19.values) / 1e6, color=ROUGE, ls='--', lw=1.3,
            label="Projection de l'ISQ en 2019")
axs[0].set_title('Effectif des RNP (milliers)', fontsize=10); axs[1].set_title('Population (millions)', fontsize=10)
axs[0].legend(frameon=False, fontsize=7.5)
axs[1].legend(handles=[axs[1].lines[-1]], frameon=False, fontsize=7.5, loc='lower right')
for ax in axs:
    ax.axvline(2030, color=GRIS_C, ls=':', lw=1)
save(fig, 'f1_trois_scenarios.png')

# Trajectoire de population vs ISQ 2019
obs = {}
with open(os.path.join(ROOT, 'data', 'stc', '17100009.csv'), encoding='utf-8-sig') as f:
    for r in csv.DictReader(f, delimiter=';'):
        if r['GÉO'] == 'Québec' and r['PÉRIODE DE RÉFÉRENCE'].endswith('-07-01'):
            obs[int(r['PÉRIODE DE RÉFÉRENCE'][:4])] = int(r['VALEUR'])
obs = pd.Series(obs).loc[2016:2026]
p19 = pd.Series({2016: 8226000, 2021: 8568200, 2026: 8830200, 2031: 9039500, 2036: 9209300, 2041: 9350200})
yrs = np.arange(2016, 2042); isq19 = pd.Series(np.interp(yrs, p19.index, p19.values), index=yrs)
fig, ax = plt.subplots(figsize=(6.3, 3.6))
ax.plot(obs.index, obs / 1e6, color=NOIR, lw=2.2, label='Observée')
ax.plot(isq19.index, isq19 / 1e6, color=ROUGE, ls='--', lw=1.4, label="Prévue par l'ISQ en 2019")
for k, (c, ls, lw, lab) in STY.items():
    s = t[k]['pop'].loc[2026:2041]
    ax.plot(s.index, s / 1e6, color=c, ls=ls, lw=lw, label=lab)
ax.set_ylabel("millions d'habitants"); ax.legend(frameon=False, fontsize=8, loc='upper left')
save(fig, 'f9_trajectoire_3scen.png')

# Éditions de l'ISQ (2041) et scénarios
ed = [(2009, 9.05, 8.14, 10.08), (2014, 9.556, 8.795, 10.310), (2019, 9.350, 8.452, 10.227), (2021, 9.485, None, None),
      (2022, 9.489, None, None), (2024, 9.622, 8.742, 10.512), (2025, 9.168, None, None), (2026, 9.147, 8.370, 9.943)]
fig, ax = plt.subplots(figsize=(6.3, 3.6))
for i, (e, r, lo, hi) in enumerate(ed):
    if lo:
        ax.plot([i, i], [lo, hi], color=GRIS, lw=1.2)
        ax.plot([i - .12, i + .12], [lo, lo], color=GRIS, lw=1.2); ax.plot([i - .12, i + .12], [hi, hi], color=GRIS, lw=1.2)
    ax.scatter(i, r, color=NOIR, s=30, zorder=3)
for k, (c, ls, lw, lab) in STY.items():
    v = t[k]['pop'].loc[2041] / 1e6
    ax.axhline(v, color=c, ls=ls, lw=1.3)
    ax.text(0.35, v + (.02 if k != PQ else -.07), f'{lab.split(" :")[0]} : {fr(v, 2)} M', fontsize=7.5, ha='left', color=c)
ax.set_xticks(range(len(ed)), [str(e[0]) for e in ed]); ax.set_xlabel("Édition des perspectives de l'ISQ")
ax.set_ylabel('Population projetée en 2041 (millions)')
save(fig, 'f10_editions_3scen.png')

# Effet cumulé sur le PIB par habitant : décomposition du proposé
cum = lambda s: (s / s.loc[2026] - 1) * 100
s = t[PR]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.plot(s.index, cum(s.wa / s['pop']), color=GRIS, lw=2, ls='--', label='Facteur démographique (part des 15-64 ans)')
ax.plot(s.index, cum(s.eff / s.wa), color=BLEU, lw=2, ls='--', label="Facteur d'intégration")
ax.plot(s.index, cum(s.y), color=NOIR, lw=2.4, label='PIB par habitant (effet de composition)')
ax.axhline(0, color=GRIS, lw=1)
ax.set_ylabel('Écart cumulé depuis 2026 (%)'); ax.legend(frameon=False, fontsize=7.8, loc='lower left')
save(fig, 'f11_croissance_base_70k.png')

# Effet cumulé : trois scénarios et vitesse
fig, ax = plt.subplots(figsize=(6.3, 3.4))
for k, (c, ls, lw, lab) in STY.items():
    ax.plot(t.index, cum(t[k].y), color=c, ls=ls, lw=lw, label=lab)
ax.plot(t.index, cum(t[PR + ' + vitesse'].y), color=BLEU, ls='-.', lw=1.6, label='Transition et intégration deux fois plus rapide')
ax.axhline(0, color=GRIS, lw=1)
ax.set_ylabel('Écart cumulé depuis 2026 (%)'); ax.legend(frameon=False, fontsize=7.8, loc='lower left')
save(fig, 'f12_croissance_3scen.png')

# Poids du Québec dans le Canada
h = pd.read_csv(os.path.join(ROOT, 'data', 'stc', 'population_canada_qc_juillet.csv')).set_index('annee')
fig, ax = plt.subplots(figsize=(6.3, 3.6))
ax.plot(h.index, h.part_qc, color=NOIR, lw=2.2, label='Observé')
ax.fill_between(t.index, t[PR].part_qc_HG, t[PR].part_qc_LG, color=BLEU, alpha=.12, lw=0,
                label='Transition : croissance canadienne faible à forte')
for k, (c, ls, lw, lab) in STY.items():
    ax.plot(t.index, t[k].part_qc_M1, color=c, ls=ls, lw=lw, label=lab)
ax.set_ylabel('Part du Québec dans la population canadienne (%)'); ax.set_ylim(15, 30)
ax.legend(frameon=False, fontsize=7.8, loc='upper right')
save(fig, 'f13_poids_quebec.png')
print('ok')

# Effet cumulé : trois scénarios et 35 000 avec intégration plus rapide (version rédigée)
fig, ax = plt.subplots(figsize=(6.3, 3.4))
for k, (c, ls, lw, lab) in STY.items():
    ax.plot(t.index, cum(t[k].y), color=c, ls=ls, lw=lw, label=lab)
ax.plot(t.index, cum(t[PQ + ' + vitesse'].y), color=ROUGE, ls='-.', lw=1.6,
        label='35 000, avec intégration deux fois plus rapide')
ax.axhline(0, color=GRIS, lw=1)
ax.set_ylabel('Écart cumulé depuis 2026 (%)'); ax.legend(frameon=False, fontsize=7.8, loc='lower left')
save(fig, 'f12b_croissance_35k_vitesse.png')
