import numpy as np 
import pandas as pd 
from scipy.optimize import minimize
import time

class immsim:
    def __init__(self):
        # Chargement des données principales depuis un fichier Excel
        self.data = pd.read_excel('./data/tx_isq.xlsx', sheet_name='tx_by_age')
       
        # Taux de mortalité par année et âge
        self.mx = self.data[['year','age','mx']]
        self.mx.set_index(['year','age'],inplace=True)
        self.mx = self.mx.unstack()
        self.mx.columns = self.mx.columns.droplevel()
        self.mx = self.mx.astype('float64')
        
        # Taux de migration interprovinciale nette
        self.interpx = self.data[['year','age','tx_net_interp_migration']]
        self.interpx.set_index(['year','age'],inplace=True)
        self.interpx = self.interpx.unstack()
        self.interpx.columns = self.interpx.columns.droplevel()
        self.interpx = self.interpx.astype('float64')
        
        # Taux d'émigration nette
        self.ex = self.data[['year','age','tx_net_emmigrant']]
        self.ex.set_index(['year','age'],inplace=True)
        self.ex = self.ex.unstack()
        self.ex.columns = self.ex.columns.droplevel()
        self.ex = self.ex.astype('float64')
        
        # Immigration totale par année (moyenne)
        self.ix = self.data.groupby('year').mean()['tot_immig']
        self.ix = self.ix.astype('float64')
        
        # Part de l'immigration par âge
        self.share_ix = self.data[['year','age','prop_immig']]
        self.share_ix.set_index(['year','age'],inplace=True)
        self.share_ix = self.share_ix.unstack()
        self.share_ix.columns = self.share_ix.columns.droplevel()
        self.share_ix = self.share_ix.astype('float64')
        
        # Population NPR (Non-Permanent Residents) par année et âge
        self.npr_temp = self.data[['year','age','NPR']]
        self.npr_temp.set_index(['year','age'],inplace=True)
        self.npr_temp = self.npr_temp.unstack()
        self.npr_temp.columns = self.npr_temp.columns.droplevel()
        self.npr_temp = self.npr_temp.astype('float64')
        self.years_since=list(range(0,11))

        years = self.npr_temp.index.tolist()
        ages = self.npr_temp.columns.tolist()
        years_since = list(range(0, 11))
        multi_index = pd.MultiIndex.from_product([years, years_since], names=['year', 'years_since_entry'])
        self.npr = pd.DataFrame(index=multi_index, columns=ages, dtype='float64')

        # Initialisation : on répartit la population NPR initiale sur chaque years_since_entry en fonction des proportions estimées sur IMDB
        npr_initial_prop = pd.read_excel('./data/tx_isq.xlsx', sheet_name='npr_proportion', index_col=0)
        for year in years: 
            for yse in years_since:
                self.npr.loc[(year, yse), :] = self.npr_temp.loc[year, :] * npr_initial_prop.loc[:, yse]
     


        # Part de la population NPR par âge (pour chaque année et par years_since_entry)
        self.share_npr = self.npr.div(self.npr.sum(axis=1), axis=0)     #axis=1: faire la somme des valeurs des ages (colonnes)

        # VAS: Initialisation de la population immigrante (NPRs devenus PRs)
        immigrants_2023 = pd.read_excel('./data/tx_isq.xlsx', sheet_name='immig_pop', index_col=0).transpose()
        imm_years_since = list(range(11))
        imm_years = list(range(2024, 2071))
        imm_ages = list(range(0, 101))
        multi_index_imm = pd.MultiIndex.from_product([imm_years, imm_years_since], names=['year', 'imm_duration'])
        self.imm_sim = pd.DataFrame(index=multi_index_imm, columns=imm_ages, dtype='float64')
        self.imm_sim = self.imm_sim.fillna(0)
        #self.imm_sim.loc[(2024,slice(None)),:] = immigrants_2023.values
        self.imm_sim.loc[(2024,slice(None)),0:90] = immigrants_2023.values
        self.imm_sim.loc[(2024,slice(None)),91:] = 0

        # Taux de natalité par année
        self.bx = pd.read_excel('./data/tx_isq.xlsx', sheet_name='tx_naissance')
        self.bx = self.bx[['year','tx_naissance']]
        self.bx.set_index('year',inplace=True)
        
        # Population totale par année et âge
        self.pop = self.data[['year','age','population']]
        self.pop.set_index(['year','age'],inplace=True)
        self.pop = self.pop.unstack()
        self.pop.columns = self.pop.columns.droplevel()
        self.pop = self.pop.astype('float64')

        #Vas 1. Lire le fichier des taux NPR
        self.npr_exit_rates = pd.read_excel('./data/tx_isq.xlsx', sheet_name='npr_exits', index_col=0)
        self.npr_apply_pr_rates = pd.read_excel('./data/tx_isq.xlsx', sheet_name='npr_apply_pr', index_col=0)
        self.npr_exit_rates = self.npr_exit_rates.astype('float64')
        self.npr_apply_pr_rates = self.npr_apply_pr_rates.astype('float64')

        # Initialisation des politiques d'immigration et NPR
        self.set_npr_policy()
        self.set_imm_policy()
        return 

    # Définir la politique NPR (cap d'entrée et taux de renouvellement)
    def set_npr_policy(self, entry_cap = 150e3, renewal_rate = 0.88):
        self.npr_entry_cap = entry_cap
        self.npr_renewal_rate = renewal_rate
        return 

    # Définir la politique d'immigration permanente (cap d'entrée et taux d'acceptation)
    def set_imm_policy(self, entry_cap = 50e3, accept_rate = 0.06):
        self.pr_entry_cap = entry_cap
        self.accept_rate = accept_rate
        return  

    # Fonctions d'accès aux différents taux et populations
    def get_mx(self,year,age):
        return self.mx.loc[year,age]
    def get_interpx(self,year,age):
        return self.interpx.loc[year,age]
    def get_ex(self,year,age):
        return self.ex.loc[year,age]
    def get_ix(self,year):
        return self.ix[year]    
    def get_share_ix(self,year,age):
        return self.share_ix.loc[year,age]
    def get_npr(self,year,age):
        return self.npr.loc[year,age]
    def get_share_npr(self,year,age):
        return self.share_npr.loc[(year,0),age]
    #def get_share_npr_mean(self,year,age):
    #    self.share_npr_mean = self.share_npr.groupby(level='year').mean()
        return self.share_npr_mean.loc[year,age]
    # Taux de natalité pour une année donnée
    def get_bx(self,year):
        return self.bx.loc[year,'tx_naissance']
    # Population totale pour une année et un âge donnés
    def get_pop(self,year,age):
        return self.pop.loc[year,age] 

    # Calcul du nombre d'entrées NPR pour une année et un âge donnés
    def get_npr_entry(self,t,a):
        return self.npr_entry_cap * self.get_share_npr(t,   a) 

    # Calcul du nombre de sorties NPR (non-renouvellements)
    def get_npr_exits(self, t, a, factor=1):
        if t > 2025:
            npr_matrix = self.npr_sim.loc[t - 1].T
        else:
            npr_matrix = self.npr.loc[t - 1].T
        raw_exits = npr_matrix.values * (self.npr_exit_rates * factor).clip(upper=1).values
        return pd.Series(raw_exits[a-1, :], index=self.npr_exit_rates.columns)
        #return self.exit_matrix.loc[a]

    # Calcul du nombre de NPR qui demandent la résidence permanente
    def get_npr_apply_pr(self,t, a, factor=1):
        if t > 2025:
            npr_matrix = self.npr_sim.loc[t - 1].T
        else:
            npr_matrix = self.npr.loc[t - 1].T
        raw_applies = npr_matrix.values * (self.npr_apply_pr_rates * factor).clip(upper=1).values
        return pd.Series(raw_applies[a-1, :], index=self.npr_apply_pr_rates.columns)
        #return self.apply_matrix.loc[a]
    
    # VAS: calcul du nombre de PRs qui quittent
    def get_imm_exits(self, t, a):
        stock = self.imm_sim.loc[(t - 1, slice(None)), a].sum() # Sum of all PRs of age a in previous year, across all durations
        return stock * self.get_mx(t, a)  # mortality rate

    # Fonction principale de projection démographique
    def proj(self,year):
        years = list(range(year,2071))
        years_for_pd =list(range(year-1,2071))
        n_years = len(years)
        ages = list(range(0,101))
        n_ages = len(ages)
        years_since = list(range(0,11))
        # Initialisation des DataFrames de simulation
        self.pop_sim = pd.DataFrame(index=years_for_pd,columns=ages)
        self.deaths_sim = pd.DataFrame(index=years_for_pd,columns=ages)
        self.emmig_sim = pd.DataFrame(index=years_for_pd,columns=ages)
        self.interp_sim = pd.DataFrame(index=years_for_pd,columns=ages)
        multi_index = pd.MultiIndex.from_product([years_for_pd,years_since], names=['year', 'years_since_entry'])
        self.npr_sim = pd.DataFrame(index=multi_index,columns=ages)
        self.minimize_time = 0
        self.max_minimize_time = 0
        for t in years:
            # Initialisation de la population de départ
            if t==year:
                start_pop = self.pop.loc[year-1,:]
            else:
                start_pop = self.pop_sim.loc[t-1,:]
            # Initialisation de la population NPR de départ
            if t==year:
                npr_pop = self.npr.loc[year-1,:] * (614577/self.npr.loc[year-1,:].sum().sum())
            else :
                npr_pop = self.npr_sim.loc[t-1,:]

            # Naissances (somme population 15-45 ans * taux de natalité)
            self.pop_sim.loc[t,0] = start_pop.loc[[x for x in range(15,46)]].sum()*self.get_bx(t)

            #Vas: 2. Prepare NPR stock matrix
            npr_matrix = self.npr_sim.loc[t - 1].T if t > 2025 else self.npr.loc[t - 1].T
            #Vas: 3. Element-wise multiplication
            #raw_exits = npr_matrix.values * self.npr_exit_rates.values
            #raw_applies = npr_matrix.values * self.npr_apply_pr_rates.values
            total_nprs = npr_matrix.values.sum()
            
            #effective_exit_rate = raw_exits.sum() / total_nprs if total_nprs > 0 else 0
            #effective_apply_rate = raw_applies.sum() / total_nprs if total_nprs > 0 else 0

            #exit_factor = (1 - self.npr_renewal_rate) / effective_exit_rate if effective_exit_rate > 0 else 0
            #apply_factor = self.accept_rate / effective_apply_rate if effective_apply_rate > 0 else 0
            def f(exit_factor):
                exit_factor = np.exp(exit_factor[0])
                npr_exit_rates = (self.npr_exit_rates * exit_factor).clip(upper=1)
                raw_exits = npr_matrix.values * npr_exit_rates.values
                effective_exit_rate = raw_exits.sum() / total_nprs if total_nprs > 0 else 0
                return np.abs(effective_exit_rate-(1-self.npr_renewal_rate))
            
            start_time = time.time()
            res = minimize(f, x0=np.log(1), method='nelder-mead', tol=1e-6)
            end_time = time.time()
            self.minimize_time += end_time - start_time
            if end_time - start_time > self.max_minimize_time:
                self.max_minimize_time = end_time - start_time
            exit_factor = np.exp(res.x[0])

            def g(apply_factor):
                apply_factor = np.exp(apply_factor[0])
                npr_apply_rates = (self.npr_apply_pr_rates * apply_factor).clip(upper=1)
                raw_applies = npr_matrix.values * npr_apply_rates.values
                effective_apply_rate = raw_applies.sum() / total_nprs if total_nprs > 0 else 0
                return np.abs(effective_apply_rate-self.accept_rate)

            start_time = time.time()
            res = minimize(g, x0=np.log(1), method='nelder-mead', tol=1e-6)
            end_time = time.time()
            self.minimize_time += end_time - start_time
            if end_time - start_time > self.max_minimize_time:
                self.max_minimize_time = end_time - start_time
                
            apply_factor = np.exp(res.x[0])

            raw_exits = npr_matrix.values * self.npr_exit_rates.values
            raw_applies = npr_matrix.values * self.npr_apply_pr_rates.values

            self.exit_matrix = pd.DataFrame(raw_exits * exit_factor, index=range(101), columns=range(11))
            self.apply_matrix = pd.DataFrame(raw_applies * apply_factor, index=range(101), columns=range(11))

            # Vieillissement de la population (transfert à l'âge suivant)
            for a in range(1,n_ages):
                self.pop_sim.loc[t,a] = start_pop[a-1]
            # Application des décès, migrations et émigrations
            for a in range(n_ages):
                deaths = (self.pop_sim.loc[t,a]-self.npr_sim.loc[(t,slice(None)),a].sum(axis=0))*self.get_mx(t,a)
                interp = (self.pop_sim.loc[t,a]-self.npr_sim.loc[(t,slice(None)),a].sum(axis=0))*self.get_interpx(t,a)
                emigr  = (self.pop_sim.loc[t,a]-self.npr_sim.loc[(t,slice(None)),a].sum(axis=0))*self.get_ex(t,a)
                self.pop_sim.loc[t,a] = self.pop_sim.loc[t,a] - deaths + interp - emigr  
                self.deaths_sim.loc[t,a] = deaths 
                self.emmig_sim.loc[t,a] = emigr
                self.interp_sim.loc[t,a] = interp
            # Transitions NPR
            npr_exits = np.zeros((n_ages, len(years_since)))
            npr_entries = np.zeros(n_ages)
            npr_apply_pr = np.zeros((n_ages, len(years_since))) 
            for a in range(n_ages): #add factors
                if a>0:
                    npr_exits[a,:] = self.get_npr_exits(t,a,exit_factor).values
                    npr_apply_pr[a,:] = self.get_npr_apply_pr(t,a,apply_factor).values
                npr_entries[a] = self.get_npr_entry(t,a) 
            # Gestion du cap sur la résidence permanente
            tot_apply_pr = npr_apply_pr.sum().sum()
            if tot_apply_pr > self.pr_entry_cap:
                npr_accepted_pr = npr_apply_pr * self.pr_entry_cap/tot_apply_pr
                npr_rejected_pr = tot_apply_pr - npr_accepted_pr
            else :
                npr_accepted_pr = npr_apply_pr
                npr_rejected_pr = np.zeros(n_ages)

            #VAS: Mise à jour de la population immigrante 
            if t == 2026 :
                self.npr_accepted_pr = npr_accepted_pr
            if t == 2027 :
                self.npr_accepted_pr_2027 = npr_accepted_pr
            if t == 2030 :
                self.npr_accepted_pr_2030 = npr_accepted_pr
            for a in ages:
                for d in range(10):
                    self.imm_sim.loc[(t, d+1), a] = self.imm_sim.loc[(t-1, d), a]  #Shift existing immigrants by 1 year (duration 0→1, ..., 9→10)
                self.imm_sim.loc[(t, 10), a] += self.imm_sim.loc[(t-1, 10), a]   
                self.imm_sim.loc[(t, slice(1,None)), a] += npr_accepted_pr[a, 0:-1]
                self.imm_sim.loc[(t, len(years_since)-1), a] += npr_accepted_pr[a, -1]
                
                #for d in range(9, -1, -1):    #Shift existing immigrants by 1 year (duration 0→1, ..., 9→10), backwards to avoid overwriting
                    #if d < 9:
                        #self.imm_sim.loc[(t, d + 1), a] = self.imm_sim.loc[(t - 1, d), a]
                    #else:
                        #self.imm_sim.loc[(t, 10), a] += self.imm_sim.loc[(t - 1, 9), a]
                        #self.imm_sim.loc[(t, 10), a] += self.imm_sim.loc[(t - 1, 10), a]  #
                #self.imm_sim.loc[(t, slice(1,None)), a] += npr_accepted_pr[a, 0:-1]  # Add accepted PRs to
                #self.imm_sim.loc[(t,len(years_since)-1), a] += npr_accepted_pr[a, -1]  # Add accepted PRs to
                #self.pop_sim.loc[t, a] += npr_accepted_pr.iloc[a, :].sum()
            if t == 2025 :
                self.imm_sim_2025 = self.imm_sim.copy()
            if t == 2026 :
                self.imm_sim_2026 = self.imm_sim.copy()
            if t == 2027 :
                self.imm_sim_2027 = self.imm_sim.copy()
            # VAS: Ajout des PR deaths
            imm_exits = np.zeros(len(ages))
            for a in ages:
                imm_exits[a] = self.get_imm_exits(t, a)
                total_prs = self.imm_sim.loc[(t, slice(None)), a].sum()
                if total_prs > 0:
                    for d in range(len(years_since)):
                        share = self.imm_sim.loc[(t, d), a] / total_prs
                        self.imm_sim.loc[(t, d), a] -= imm_exits[a] * share
            if t == 2025 :
                self.imm_sim_2025_after_exit = self.imm_sim.copy()
            if t == 2026 :
                self.imm_sim_2026_after_exit = self.imm_sim.copy()
            if t == 2027 :
                self.imm_sim_2027_after_exit = self.imm_sim.copy()
            # Mise à jour de la population NPR
            for a in range(n_ages):
                if a>0:
                    self.npr_sim.loc[(t,slice(1,None)),a] = npr_pop.iloc[:-1,a-1].values 
                    self.npr_sim.loc[(t,len(years_since)-1),a] += npr_pop.iloc[-1,a-1]
                    self.npr_sim.loc[(t,slice(1, None)),a] -= npr_exits[a,:-1]
                    self.npr_sim.loc[(t,len(years_since)-1),a] -= npr_exits[a,-1]
                    self.npr_sim.loc[(t,slice(1, None)),a] -= npr_accepted_pr[a,:-1]
                    self.npr_sim.loc[(t,len(years_since)-1),a] -= npr_accepted_pr[a,-1]
                    #subtract exits and PR transitions from those already in duration 40
                    #self.npr_sim.loc[(t, 40), a] -= self.get_npr_exits(t, a).loc[40]
                    #self.npr_sim.loc[(t, 40), a] -= self.get_npr_apply_pr(t, a).loc[40]                   
                else :
                    self.npr_sim.loc[(t,slice(None)),0] = 0
                self.npr_sim.loc[(t,0),a] = npr_entries[a]
                self.pop_sim.loc[t,a] = self.pop_sim.loc[t,a] + npr_entries[a] - npr_exits[a].sum() #+ npr_accepted_pr[a]
            # Immigration permanente restante (hors NPR)
            immig_space = max(self.pr_entry_cap - npr_accepted_pr.sum().sum(),0)
            if immig_space > 0:
                for a in range(n_ages):
                    self.pop_sim.loc[t,a] += immig_space*self.get_share_ix(t,a) 
                    self.imm_sim.loc[(t, 0), a] += immig_space*self.get_share_ix(t,a)

        # Remise à jour de la population initiale pour l'année de départ
        self.pop_sim.loc[year-1,:] = self.pop.loc[year-1,:] 
        for y in years_since:
            self.npr_sim.loc[(year-1, y), :] = self.npr.loc[(year-1,y), :] * (614577/self.npr.loc[year-1, :].sum().sum())      
        return
    def export(self,path='./output/',type='csv'):
        """
        Prépare les données pour les exporter pour utilisation dans SIMFIN
        """
        # On répartit la population par sexe
        # On suppose un ratio de 51% d'hommes et 49% de femmes
        # Note: This is a simplification, actual ratios may vary by age group and year
        male_ratio = 0.51
        female_ratio = 1 - male_ratio
        # Create male and female DataFrames
        pop_male = self.pop_sim * male_ratio
        pop_female = self.pop_sim * female_ratio
        pop_male['sex'] = 'male'
        pop_female['sex'] = 'female'
        pop_by_sex = pd.concat([pop_male, pop_female])
        pop_by_sex = pop_by_sex.set_index('sex', append=True)
        pop_by_sex.index.names = ['year', 'sex']
        
        # On réorganise les données pour avoir une structure par âge et sexe
        pop_by_sex.reset_index(inplace=True)
        pop_by_sex_long = pd.melt(pop_by_sex, id_vars=['year', 'sex'], value_vars=[x for x in pop_by_sex.columns if x not in ['year', 'sex']])
        pop_by_sex_long.rename(columns={'variable': 'age', 'value': 'population'}, inplace=True)
        pop_by_sex_long = pop_by_sex_long.pivot(index=['age', 'sex'], columns='year', values='population')

        imm_sim = self.imm_sim.reset_index()
        imm_sim_long = pd.melt(imm_sim, id_vars=['year', 'imm_duration'], value_vars=[x for x in imm_sim.columns if x not in ['year', 'imm_duration']])
        imm_sim_long.rename(columns={'imm_duration' : 'year_in_qc', 'variable': 'age', 'value': 'population'}, inplace=True)
        imm_sim_long = imm_sim_long.pivot(index=['age', 'year_in_qc'], columns='year', values='population')

        npr_sim = self.npr_sim.reset_index()
        npr_sim_long = pd.melt(npr_sim, id_vars=['year', 'years_since_entry'], value_vars=[x for x in npr_sim.columns if x not in ['year', 'years_since_entry']])
        npr_sim_long.rename(columns={'years_since_entry': 'years_in_qc', 'variable': 'age', 'value': 'population'}, inplace=True)
        npr_sim_long = npr_sim_long.pivot(index=['age', 'years_in_qc'], columns='year', values='population')

        # Export les résultats de la simulation
        if type not in ['csv', 'xlsx']:
            raise ValueError("Type must be either 'csv' or 'xlsx'")
        if type == 'xlsx':
            pop_by_sex_long.to_excel(f'{path}pop_sim.xlsx')
            npr_sim_long.to_excel(f'{path}npr_sim.xlsx')
            imm_sim_long.to_excel(f'{path}imm_sim.xlsx')
        else:  # type == 'csv'
            pop_by_sex_long.to_csv(f'{path}pop_sim.{type}')
            npr_sim_long.to_csv(f'{path}npr_sim.{type}')
            imm_sim_long.to_csv(f'{path}imm_sim.{type}')
        return





