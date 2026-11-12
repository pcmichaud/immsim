"""Figure de la grille Québec x fédéral avec le plan québécois (45 000 RP) : f7_grille_plan45000.png."""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(ROOT, 'output', 'figures'); os.makedirs(OUT, exist_ok=True)
BLEU, GRIS, GRIS_C, NOIR = '#1f5fa8', '#8c8c8c', '#c8c8c8', '#222222'
plt.rcParams.update({'font.family': 'Palatino', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6',
                     'axes.axisbelow': True, 'figure.dpi': 300})
fr = lambda v, d=0: f'{v:,.{d}f}'.replace(',', ' ').replace('.', ',')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300)
    plt.close(fig)


# --- Figure 9 : grille Québec x fédéral, plan québécois (45 000 RP dont 65 % de RNP), effectif 2030
g = pd.read_csv(os.path.join(ROOT, 'output', 'grille_plan45000.csv'))
qs = ['Entrées de 2025', 'Plafonds du plan québécois', 'Réduction de 30 %']
fs = ['Flux de 2025', 'Asile au rythme de 2026', 'Asile 2026 et PMI −30 %']
fig, ax = plt.subplots(figsize=(6.3, 3.6))
w = .26; cols = [GRIS_C, GRIS, BLEU]
labs_f = ['Fédéral comme en 2025', 'Asile au rythme de 2026', 'Asile 2026 et PMI −30 %']
for j, fn in enumerate(fs):
    vals = [g[(g.quebec == qn) & (g.federal == fn)].rnp_2030.iloc[0] / 1e3 for qn in qs]
    ax.bar(np.arange(3) + (j - 1) * w, vals, w, color=cols[j], label=labs_f[j])
    for i, v in enumerate(vals):
        ax.text(i + (j - 1) * w, v + 6, fr(v), ha='center', fontsize=7.5)
ax.axhline(531.821, color=NOIR, lw=1, label='Effectif en juillet 2026 : 532')
ax.axhline(375, color=BLEU, ls='--', lw=1, label='Trajectoire ISQ 2026 en 2030 : 375')
ax.set_xticks(range(3), ['Québec : entrées 2025', 'Québec : plafonds CAQ', 'Québec : −30 %'])
ax.set_ylabel('Effectif des RNP en 2030 (milliers)'); ax.set_ylim(0, 720); ax.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2)
save(fig, 'f7_grille_plan45000.png')
