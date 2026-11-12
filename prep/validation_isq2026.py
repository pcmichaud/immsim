"""
Validation : immsim2026 doit reproduire le scénario Référence ISQ 2026 quand on lui donne
les mêmes hypothèses (départ au 1er juillet 2025, RP 54 120 puis 45 000, effectif RNP ISQ).
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immsim2026 import immsim2026  # noqa: E402
from prep_intrants_2026 import isq_compo, isq_pop_by_age, stc_by_age  # noqa: E402

compo = isq_compo()
pop_isq = isq_pop_by_age()
npr25 = stc_by_age('17100158.csv', 2025)
npr25 = npr25 / npr25.sum() * compo.loc[2025, 'rnp_t']

m = immsim2026()
m.set_start(2025, pop_isq.loc[2025].values, npr25.values)
m.set_npr_policy(renewal_rate=0.825, npr_target={t: compo.loc[t - 1, 'rnp_t1'] for t in range(2026, 2071)})
m.set_imm_policy(entry_cap={t: compo.loc[t - 1, 'immigrants'] for t in range(2026, 2071)}, accept_rate=0.061)
m.proj(2070)

yrs = [2026, 2027, 2030, 2035, 2041, 2051, 2061, 2070]
s = m.summary(yrs)
ref = pd.DataFrame({
    'population': pop_isq.loc[yrs].sum(axis=1),
    '0-19': pop_isq.loc[yrs, 0:19].sum(axis=1),
    '20-64': pop_isq.loc[yrs, 20:64].sum(axis=1),
    '65+': pop_isq.loc[yrs, 65:].sum(axis=1),
})
cmp = pd.concat({'immsim': s[['population', '0-19', '20-64', '65+']], 'ISQ 2026': ref}, axis=1)
for c in ['population', '0-19', '20-64', '65+']:
    cmp[('écart %', c)] = (cmp[('immsim', c)] / cmp[('ISQ 2026', c)] - 1) * 100
pd.set_option('display.width', 200)
print(cmp.round(1).to_string())
print()
print(m.flows.loc[[2026, 2027, 2030, 2040, 2050, 2070]].round(0).to_string())
cmp.to_csv(os.path.join(ROOT, 'output', 'validation_isq2026.csv'))
