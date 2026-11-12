# immsim : modèle de simulation de scénarios d'immigration au Québec

Pierre-Carl Michaud, HEC Montréal (Chaire de recherche Jacques-Parizeau en politiques économiques)

**Version 2026** (branche `v2026`). Elle accompagne le rapport *Atterrir en douceur : mise à jour de l'analyse des scénarios d'immigration au Québec* (novembre 2026). La version de 2025 (`immsim.py`, rapport *Planifier un atterrissage en douceur*, mai 2025) reste disponible sur la branche `main`.

**Simulateur en ligne :** https://pcmichaud.github.io/immsim/ (dossier `docs/`)

## Ce que fait le modèle

immsim projette la population du Québec au 1er juillet de chaque année, par âge simple (0 à 100 ans et plus), de 2026 à 2070. Il suit trois stocks :
- la population totale ;
- l'effectif des résidents non permanents (RNP), par âge et par durée de séjour (0 à 10 ans et plus) ;
- les résidents permanents récents, selon leurs années au Québec et leur voie d'arrivée (directement de l'étranger, ou après un statut temporaire).

Les leviers sont :
- les entrées annuelles de RNP ;
- les admissions à la résidence permanente (total et part venue de l'étranger) ;
- un multiplicateur des risques de départ du Québec.

Les passages de RNP vers la résidence permanente comblent la différence entre les admissions totales et celles de l'étranger.

### Équations

Notation : $P_a(t)$ est la population, $N_{d,a}(t)$ l'effectif des RNP selon la durée $d$ et l'âge $a$, $\tilde{P}_a(t)=P_{a-1}(t-1)$ la population vieillie d'un an.

$$B(t)=b(t)\sum_{a=15}^{45}P_a(t-1)$$

$$P_a(t)=\tilde{P}_a(t)\bigl(1-m_a(t)\bigr)-\bigl(e_a(t)-i_a(t)\bigr)\bigl(\tilde{P}_a(t)-\tilde{N}_a(t)\bigr)+M(t)\,s_a-\tilde{X}_a(t)+RP^E(t)\,\omega_a(t)$$

$$N_{0,a}(t)=M(t)\,s_a,\qquad N_{d+1,a+1}(t)=N_{d,a}(t-1)-X_{d,a}(t)-A_{d,a}(t)$$

$$X_{d,a}(t)=\min\bigl(\kappa\,\phi_X\,x_{a,d},1\bigr)N_{d,a}(t-1),\qquad A_{d,a}(t)=\min\bigl(\phi_A(t)\,\pi_{a,d},1\bigr)N_{d,a}(t-1)$$

$$\sum_{d,a}A_{d,a}(t)=RP(t)-RP^E(t)$$

Les symboles :
- $m$ : mortalité ;
- $e$ : émigration nette ;
- $i$ : migration interprovinciale nette (ces deux taux s'appliquent à la population hors RNP) ;
- $M$ : entrées de RNP, réparties par âge selon $s_a$ ;
- $X$ : départs du Québec ;
- $A$ : passages à la résidence permanente ;
- $x_{a,d}$, $\pi_{a,d}$ : risques de base estimés sur la Base de données longitudinales sur l'immigration (BDIM) ;
- $RP$ et $RP^E$ : admissions permanentes totales et venues de l'étranger, réparties par âge selon $\omega_a$.

Le facteur $\phi_A(t)$ s'ajuste chaque année pour que la dernière condition tienne.

### Calibrage

- **Population et RNP au 1er juillet 2026 :** Statistique Canada, tableaux 17-10-0005-01 et 17-10-0158-01.
- **Fécondité, mortalité, émigration et migration interprovinciale :** mise à jour 2026 des perspectives de l'ISQ, scénario Référence.
- **Facteurs $\phi_X$ et $\phi_A$ :** calés une seule fois (bissection) pour que, appliqués à l'effectif par âge de juillet 2025, ils reproduisent les taux observés en 2025 (17,5 % de départs, 6,1 % de passages). $\phi_X$ est ensuite tenu fixe.
- **Validation :** avec les hypothèses de l'ISQ, le modèle reproduit la population de référence à 0,1 % près jusqu'en 2041.

L'annexe A du rapport présente le modèle en détail. L'annexe E décrit l'indice de composition utilisé pour le PIB par habitant.

## Structure

| Chemin | Contenu |
|---|---|
| `immsim2026.py` | le modèle (classe `immsim2026`) |
| `data/tx_isq_2026.xlsx` | intrants : taux de l'ISQ 2026, risques de la BDIM, stocks de départ |
| `data/stc/`, `data/isq2026/` | données publiques de Statistique Canada et de l'ISQ |
| `prep/prep_intrants_2026.py` | construit `tx_isq_2026.xlsx` (exige aussi `data/stc/17100005.csv`, à télécharger de Statistique Canada, non versé à cause de sa taille) |
| `prep/chemins_70k.py` | trajectoires d'admissions des trois scénarios du rapport |
| `prep/scenarios_70k.py`, `prep/figures_70k.py` | scénarios et figures du rapport |
| `prep/scenarios_grille_plan.py`, `prep/figure_grille_plan.py` | grille des politiques québécoise et fédérale (Figure 5) |
| `prep/bilan_cible.py`, `prep/marge_quebec.py` | bilan d'une cible de 250 000 RNP et modèle simplifié à trois groupes |
| `prep/validation_isq2026.py` | validation sur les projections de l'ISQ |
| `prep/export_web.py` | exporte les intrants du simulateur (`docs/intrants.json`) et des résultats de référence |
| `docs/` | simulateur web : `index.html` et `immsim.js` (transposition JavaScript du modèle) |
| `tests/` | test de non-régression (Tableau 3 du rapport) et résultats de référence du simulateur |

## Reproduire les résultats du rapport

```bash
pip install -r requirements.txt
python tests/test_regression.py        # Tableau 3 du rapport
python prep/scenarios_70k.py           # trajectoires -> output/
python prep/figures_70k.py             # figures -> output/figures/
```

Le simulateur web reproduit les résultats de Python à 10⁻⁶ près ; ouvrir `docs/index.html?test=1` depuis un serveur local (`python -m http.server`) pour afficher la vérification.

## Exemple

```python
from immsim2026 import immsim2026
from prep.prep_intrants_2026 import stc_by_age

m = immsim2026().calibrate(stc_by_age('17100158.csv', 2025).values, 0.175, 0.061)
m.set_npr_policy(entry_cap=90111)                       # entrées de RNP par année
m.set_imm_policy(entry_cap=45000, pr_from_abroad=15750)  # admissions permanentes, dont de l'étranger
m.proj(2041)
print(m.summary([2026, 2030, 2041]))
```

## Limites

Ce sont des projections mécaniques, pas des prévisions :
- les comportements de 2025 sont tenus constants ;
- les entrées de RNP sont exogènes et constantes ;
- le taux de départ est commun à tous les types de permis ;
- les profils d'intégration des étudiants et des demandeurs d'asile sont des hypothèses.

L'annexe F du rapport en présente la liste complète.

Pour toute question ou correction : Pierre-Carl Michaud, pierre-carl.michaud@hec.ca
