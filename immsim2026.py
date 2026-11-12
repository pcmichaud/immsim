"""
immsim2026 : version paramétrable d'immsim (v2, avec durée depuis l'arrivée des RNP).

Différences avec immsim.py (v2, oct. 2025) :
  - fichier d'intrants au choix (par défaut data/tx_isq_2026.xlsx, calé sur l'ISQ 2026) ;
  - année et effectifs de départ libres (par défaut : STC au 1er juillet 2026), plus de
    614 577 ni de « t > 2025 » codés en dur ;
  - politiques fixes ou variables par année (dict {année: valeur}) ;
  - option `npr_target` : effectif de RNP visé par année ; les entrées sont alors déduites
    (entrées = cible - effectif restant après sorties et passages à la RP) ;
  - décès appliqués à toute la population ; émigration et migration interprovinciale appliquées
    à la population hors RNP (dans v2, l'exclusion visait npr_sim[t], encore vide à ce moment) ;
  - calcul vectoriel (numpy) et calage des facteurs par bissection plutôt que Nelder-Mead.

Convention : l'année t est la population au 1er juillet t ; proj() part de l'année de départ
(start_year) et simule start_year+1, ..., end_year.
"""
import os

import numpy as np
import pandas as pd

module_dir = os.path.dirname(os.path.abspath(__file__))
AGES = np.arange(101)
NDUR = 11  # durées 0 à 10 (10 = 10 ans et plus)


def _by_year(x, t):
    """Valeur d'une politique pour l'année t : scalaire, ou dict {année: valeur} (dernière valeur connue)."""
    if isinstance(x, dict):
        keys = sorted(k for k in x if k <= t)
        return x[keys[-1]] if keys else x[min(x)]
    return x


def _calibrate_factor(stock, rates, target):
    """Facteur f tel que somme(stock * min(rates*f, 1)) / somme(stock) = target (bissection)."""
    total = stock.sum()
    if total <= 0 or target <= 0:
        return 0.0
    eff = lambda f: (stock * np.minimum(rates * f, 1.0)).sum() / total
    lo, hi = 0.0, 1.0
    while eff(hi) < target and hi < 1e6:
        hi *= 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if eff(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


class immsim2026:
    def __init__(self, input_file=os.path.join(module_dir, 'data', 'tx_isq_2026.xlsx')):
        x = pd.ExcelFile(input_file)
        d = pd.read_excel(x, 'tx_by_age')
        piv = lambda c: d.pivot(index='year', columns='age', values=c).astype('float64')
        self.mx = piv('mx')
        self.ex = piv('tx_net_emmigrant')
        self.interpx = piv('tx_net_interp_migration')
        self.share_ix = piv('prop_immig')
        self.pop = piv('population')          # population projetée par l'ISQ (référence)
        self.npr_isq = piv('NPR')             # effectif RNP de l'ISQ, réparti par âge
        self.bx = pd.read_excel(x, 'tx_naissance').set_index('year')['tx_naissance']

        # taux par âge (lignes) et durée (colonnes), estimés sur l'IMDB (v2)
        self.npr_exit_rates = pd.read_excel(x, 'npr_exits', index_col=0).values.astype('float64')
        self.npr_apply_rates = pd.read_excel(x, 'npr_apply_pr', index_col=0).values.astype('float64')
        self.npr_dur_prop = pd.read_excel(x, 'npr_proportion', index_col=0).values.astype('float64')
        self.immig_pop0 = pd.read_excel(x, 'immig_pop', index_col=0).values.astype('float64')

        # profil par âge des nouveaux RNP : effectif par âge x part de durée 0
        base = pd.read_excel(x, 'base_2026')
        entry = base['npr'].values * self.npr_dur_prop[:, 0]
        self.npr_entry_profile = entry / entry.sum()

        ip = np.zeros((101, NDUR))
        ip[:self.immig_pop0.shape[0], :] = self.immig_pop0
        self.start_pr = ip.T                         # (durée, âge) : RP récents selon les années au Québec
        self.set_start(2026, base['population'].values, base['npr'].values)
        self.set_npr_policy()
        self.set_imm_policy()

    # ------------------------------------------------------------------ paramètres
    def set_start(self, year, pop, npr):
        """Population et RNP par âge (0-100) au 1er juillet de l'année de départ."""
        self.start_year = year
        self.start_pop = np.asarray(pop, dtype='float64')
        npr = np.asarray(npr, dtype='float64')
        self.start_npr = (npr[:, None] * self.npr_dur_prop).T   # (durée, âge)
        return self

    def set_npr_policy(self, entry_cap=90e3, renewal_rate=0.825, npr_target=None, exit_multiplier=1.0):
        """entry_cap : entrées annuelles ; 1 - renewal_rate : taux de départ du Québec ;
        npr_target : dict {année: effectif} qui remplace entry_cap quand il est fourni ;
        exit_multiplier : multiplie les risques de départ calés (utilisé seulement avec calibrate())."""
        self.npr_entry_cap = entry_cap
        self.npr_renewal_rate = renewal_rate
        self.npr_target = npr_target
        self.exit_multiplier = exit_multiplier
        return self

    def calibrate(self, npr_by_age, exit_rate, accept_rate):
        """Cale une fois pour toutes les facteurs des risques IMDB (âge x durée) pour qu'appliqués à
        l'effectif npr_by_age (réparti par durée selon npr_proportion) ils donnent les taux agrégés
        observés. Ensuite, les taux agrégés varient avec la composition du stock."""
        stock = np.asarray(npr_by_age, dtype='float64')[:, None] * self.npr_dur_prop
        self.exit_factor = _calibrate_factor(stock, self.npr_exit_rates, exit_rate)
        self.apply_factor = _calibrate_factor(stock, self.npr_apply_rates, accept_rate)
        self.fixed_factors = True
        return self

    def effective_rates(self, npr_dur_age):
        """Taux agrégés de départ et de passage à la RP pour un stock (durée, âge) donné."""
        s = npr_dur_age.T
        tot = s.sum()
        ex = (s * np.minimum(self.npr_exit_rates * self.exit_factor * self.exit_multiplier, 1)).sum() / tot
        ap = (s * np.minimum(self.npr_apply_rates * self.apply_factor, 1)).sum() / tot
        return ex, ap

    def set_imm_policy(self, entry_cap=45e3, accept_rate=0.061, pr_from_abroad=None):
        """entry_cap : admissions RP totales ; accept_rate : part de l'effectif RNP qui passe à la RP
        (plafonnée par entry_cap ; le reste des admissions vient de l'étranger).
        pr_from_abroad : si fourni, admissions venues de l'étranger fixées à ce niveau ; tout le reste de
        entry_cap est comblé par des transitions depuis le stock de RNP (sélection « de l'intérieur »),
        réparties selon le profil âge x durée des risques de passage à la RP."""
        self.pr_entry_cap = entry_cap
        self.accept_rate = accept_rate
        self.pr_from_abroad = pr_from_abroad
        return self

    # ------------------------------------------------------------------ projection
    def proj(self, end_year=2070):
        years = list(range(self.start_year, end_year + 1))
        pop = pd.DataFrame(index=years, columns=AGES, dtype='float64')
        npr_tot = pd.DataFrame(index=years, columns=AGES, dtype='float64')
        flows = pd.DataFrame(index=years[1:], dtype='float64',
                             columns=['naissances', 'deces', 'emigration', 'interprov', 'rnp_entrees',
                                      'rnp_departs', 'rnp_vers_rp', 'rp_etranger', 'rp_total', 'solde_rnp',
                                      'taux_depart', 'taux_rp'])
        self.npr_sim = {self.start_year: self.start_npr.copy()}
        self.pr_sim = {self.start_year: self.start_pr.copy()}
        share = getattr(self, 'start_pr_share_npr', 0.5)       # part du stock initial arrivée comme RNP
        self.pr_npr_sim = {self.start_year: self.start_pr * share}
        self.pr_ab_sim = {self.start_year: self.start_pr * (1 - share)}
        pop.loc[self.start_year] = self.start_pop
        npr_tot.loc[self.start_year] = self.start_npr.sum(axis=0)

        for t in years[1:]:
            p0 = pop.loc[t - 1].values
            n0 = self.npr_sim[t - 1]                       # (durée, âge) au 1er juillet t-1

            # vieillissement d'un an (100+ reste 100+)
            p = np.r_[0.0, p0[:-1]]
            p[-1] += p0[-1]
            births = p0[15:46].sum() * self.bx.loc[t]
            p[0] = births

            # RNP vieillis d'un an (âge a-1 -> a) ; sorties calculées sur le stock de t-1
            if getattr(self, 'fixed_factors', False):
                exit_f = self.exit_factor * _by_year(self.exit_multiplier, t)
                apply_f = self.apply_factor
            else:
                exit_f = _calibrate_factor(n0.T, self.npr_exit_rates, 1 - _by_year(self.npr_renewal_rate, t))
                apply_f = _calibrate_factor(n0.T, self.npr_apply_rates, _by_year(self.accept_rate, t))
            exits = (n0.T * np.minimum(self.npr_exit_rates * exit_f, 1)).T       # (durée, âge t-1)
            applies = (n0.T * np.minimum(self.npr_apply_rates * apply_f, 1)).T
            pr_cap = _by_year(self.pr_entry_cap, t)
            abroad_fixed = _by_year(getattr(self, 'pr_from_abroad', None), t)
            if abroad_fixed is not None:
                target = max(pr_cap - abroad_fixed, 0.0) / n0.sum()
                f = _calibrate_factor(n0.T, self.npr_apply_rates, target)
                accepted = (n0.T * np.minimum(self.npr_apply_rates * f, 1)).T
            else:
                accepted = applies * min(1.0, pr_cap / applies.sum()) if applies.sum() > 0 else applies
            remaining = np.clip(n0 - exits - accepted, 0, None)

            n1 = np.zeros((NDUR, 101))
            n1[1:, 1:] = remaining[:-1, :-1]                # durée d -> d+1, âge a -> a+1
            n1[-1, 1:] += remaining[-1, :-1]                # 10 ans et plus reste à 10
            n1[1:, -1] += remaining[:-1, -1]                # 100+ reste 100+
            n1[-1, -1] += remaining[-1, -1]

            tgt = self.npr_target
            target_on = tgt is not None and (not isinstance(tgt, dict) or t >= min(tgt))
            if target_on:   # cible d'effectif (un dict peut ne commencer qu'après une période d'entrées fixées)
                entries_tot = max(_by_year(tgt, t) - n1.sum(), 0.0)
            else:
                entries_tot = _by_year(self.npr_entry_cap, t)
            entries = entries_tot * self.npr_entry_profile
            n1[0, :] += entries

            # composantes sur la population résidente (hors RNP pour les migrations)
            npr_aged = np.r_[0.0, n0.sum(axis=0)[:-1]]
            npr_aged[-1] += n0.sum(axis=0)[-1]
            non_npr = np.clip(p - npr_aged, 0, None)
            deaths = p * self.mx.loc[t].values
            deaths[0] = births * self.mx.loc[t].values[0]
            emig = non_npr * self.ex.loc[t].values
            interp = non_npr * self.interpx.loc[t].values

            # RNP qui partent : on les retire de la population, par âge vieilli
            departs_aged = np.r_[0.0, exits.sum(axis=0)[:-1]]
            departs_aged[-1] += exits.sum(axis=0)[-1]

            pr_abroad = max(pr_cap - accepted.sum(), 0.0)

            # RP récents selon les années au Québec : vieillissement, décès et émigration, ajout des
            # RNP devenus RP (qui gardent leurs années au Québec) et des admissions de l'étranger (durée 0)
            q0 = self.pr_sim[t - 1]
            q1 = np.zeros((NDUR, 101))
            q1[1:, 1:] = q0[:-1, :-1] + accepted[:-1, :-1]
            q1[-1, 1:] += q0[-1, :-1] + accepted[-1, :-1]
            q1[1:, -1] += q0[:-1, -1] + accepted[:-1, -1]
            q1[-1, -1] += q0[-1, -1] + accepted[-1, -1]
            q1 *= np.clip(1 - self.mx.loc[t].values - self.ex.loc[t].values, 0, 1)
            q1[0, :] += pr_abroad * self.share_ix.loc[t].values
            self.pr_sim[t] = q1
            surv = np.clip(1 - self.mx.loc[t].values - self.ex.loc[t].values, 0, 1)
            for store, add in ((self.pr_npr_sim, accepted), (self.pr_ab_sim, None)):
                z0 = store[t - 1]
                z1 = np.zeros((NDUR, 101))
                src = z0 + (add if add is not None else 0)
                z1[1:, 1:] = src[:-1, :-1]; z1[-1, 1:] += src[-1, :-1]
                z1[1:, -1] += src[:-1, -1]; z1[-1, -1] += src[-1, -1]
                z1 *= surv
                store[t] = z1
            self.pr_ab_sim[t][0, :] += pr_abroad * self.share_ix.loc[t].values

            p = p - deaths - emig + interp + entries - departs_aged + pr_abroad * self.share_ix.loc[t].values
            pop.loc[t] = p
            self.npr_sim[t] = n1
            npr_tot.loc[t] = n1.sum(axis=0)
            flows.loc[t] = [births, deaths.sum(), emig.sum(), interp.sum(), entries_tot, exits.sum(),
                            accepted.sum(), pr_abroad, accepted.sum() + pr_abroad, n1.sum() - n0.sum(),
                            exits.sum() / n0.sum(), accepted.sum() / n0.sum()]

        self.pop_sim = pop
        self.npr_sim_age = npr_tot
        self.flows = flows
        return self

    # ------------------------------------------------------------------ intégration économique
    def effective_workers(self, t, horizon=10, gap=0.30, ages=(15, 64)):
        """Population en âge de travailler pondérée par la productivité relative des immigrants (RNP et RP
        récents) selon leurs années au Québec : poids 1 - gap à l'arrivée, rattrapage linéaire, 1 après
        `horizon` années. Les natifs et les immigrants établis ont un poids de 1."""
        d = np.arange(NDUR)
        w = 1 - gap * np.clip(1 - d / horizon, 0, None) if horizon > 0 else np.ones(NDUR)
        a0, a1 = ages
        imm = self.npr_sim[t] + self.pr_sim[t]                    # (durée, âge)
        imm_w = (w[:, None] * imm).sum(axis=0)
        total = self.pop_sim.loc[t].values
        eff = total - imm.sum(axis=0) + imm_w
        return eff[a0:a1 + 1].sum()

    @staticmethod
    def profile(e0=1.0, he=1, r0=1.0, hr=1, e_end=1.0, r_end=1.0):
        """Productivité relative selon les années au Québec (0 à 10+) = emploi relatif x revenu relatif par
        travailleur, chacun linéaire de sa valeur à l'arrivée (e0, r0) à sa valeur finale (e_end, r_end) en
        he et hr années."""
        d = np.arange(NDUR)
        e = e_end - (e_end - e0) * np.clip(1 - d / max(he, 1e-9), 0, None)
        r = r_end - (r_end - r0) * np.clip(1 - d / max(hr, 1e-9), 0, None)
        return e * r

    def effective_workers_by_status(self, t, w_npr, w_rp, ages=(15, 64)):
        """Population en âge de travailler pondérée : RNP et RP issus de RNP selon w_npr, RP arrivés de
        l'étranger selon w_rp, autres résidents = 1."""
        a0, a1 = ages
        npr_like = self.npr_sim[t] + self.pr_npr_sim[t]
        rp_like = self.pr_ab_sim[t]
        imm = npr_like + rp_like
        eff = self.pop_sim.loc[t].values - imm.sum(axis=0) + (w_npr[:, None] * npr_like).sum(axis=0) \
            + (w_rp[:, None] * rp_like).sum(axis=0)
        return eff[a0:a1 + 1].sum()

    # ------------------------------------------------------------------ résumés
    def summary(self, years=None):
        years = years or list(self.pop_sim.index)
        p = self.pop_sim.loc[years]
        out = pd.DataFrame({
            'population': p.sum(axis=1),
            'rnp': self.npr_sim_age.loc[years].sum(axis=1),
            '0-19': p.loc[:, 0:19].sum(axis=1),
            '20-64': p.loc[:, 20:64].sum(axis=1),
            '65+': p.loc[:, 65:].sum(axis=1),
        })
        out['part_rnp'] = out['rnp'] / out['population']
        out['part_20_64'] = out['20-64'] / out['population']
        return out
