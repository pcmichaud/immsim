import numpy as np 
import pandas as pd 

class immsim:
    def __init__(self):
        self.data = pd.read_excel('data/tx_isq.xlsx', sheet_name='tx_by_age')
        
        self.mx = self.data[['year','age','mx']]
        self.mx.set_index(['year','age'],inplace=True)
        self.mx = self.mx.unstack()
        self.mx.columns = self.mx.columns.droplevel()
        self.mx = self.mx.astype('float64')
        self.interpx = self.data[['year','age','tx_net_interp_migration']]
        self.interpx.set_index(['year','age'],inplace=True)
        self.interpx = self.interpx.unstack()
        self.interpx.columns = self.interpx.columns.droplevel()
        self.interpx = self.interpx.astype('float64')
        self.ex = self.data[['year','age','tx_net_emmigrant']]
        self.ex.set_index(['year','age'],inplace=True)
        self.ex = self.ex.unstack()
        self.ex.columns = self.ex.columns.droplevel()
        self.ex = self.ex.astype('float64')
        self.ix = self.data.groupby('year').mean()['tot_immig']
        self.ix = self.ix.astype('float64')
        self.share_ix = self.data[['year','age','prop_immig']]
        self.share_ix.set_index(['year','age'],inplace=True)
        self.share_ix = self.share_ix.unstack()
        self.share_ix.columns = self.share_ix.columns.droplevel()
        self.share_ix = self.share_ix.astype('float64')
        self.npr = self.data[['year','age','NPR']]
        self.npr.set_index(['year','age'],inplace=True)
        self.npr = self.npr.unstack()
        self.npr.columns = self.npr.columns.droplevel()
        self.npr = self.npr.astype('float64')
        
        self.share_npr = self.npr.copy()
        for i in self.npr.index:
            self.share_npr.loc[i,:] = self.npr.loc[i,:]/self.npr.loc[i,:].sum()

        self.bx = pd.read_excel('data/tx_isq.xlsx', sheet_name='tx_naissance')
        self.bx = self.bx[['year','tx_naissance']]
        self.bx.set_index('year',inplace=True)
        
        self.pop = self.data[['year','age','population']]
        self.pop.set_index(['year','age'],inplace=True)
        self.pop = self.pop.unstack()
        self.pop.columns = self.pop.columns.droplevel()
        self.pop = self.pop.astype('float64')

        self.set_npr_policy()
        self.set_imm_policy()
        return 
    def set_npr_policy(self, entry_cap = 150e3, renewal_rate = 0.88):
        self.npr_entry_cap = entry_cap
        self.npr_renewal_rate = renewal_rate
        return 
    def set_imm_policy(self, entry_cap = 50e3, accept_rate = 0.06):
        self.pr_entry_cap = entry_cap
        self.accept_rate = accept_rate
        return 
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
        return self.share_npr.loc[year,age]
    def get_bx(self,year):
        return self.bx.loc[year,'tx_naissance']
    def get_pop(self,year,age):
        return self.pop.loc[year,age] 
    def get_npr_entry(self,t,a):
        return self.npr_entry_cap * self.get_share_npr(t,a) 
    def get_npr_exits(self,t,a):
        if t>2025:
            stock = self.npr_sim.loc[t-1,a-1]
        else :
            stock = self.npr.loc[t-1,a-1]
        return stock * (1 - self.npr_renewal_rate)
    def get_npr_apply_pr(self,t,a):
        if t>2025:
            stock = self.npr_sim.loc[t-1,a-1]
        else :
            stock = self.npr.loc[t-1,a-1]
        return stock * self.accept_rate
    def proj(self,year):
        years = list(range(year,2071))
        years_for_pd =list(range(year-1,2071))
        n_years = len(years)
        ages = list(range(0,101))
        n_ages = len(ages)
        self.pop_sim = pd.DataFrame(index=years_for_pd,columns=ages)
        self.npr_sim = pd.DataFrame(index=years_for_pd,columns=ages)
        for t in years:
            if t==year:
                start_pop = self.pop.loc[year-1,:]
            else:
                start_pop = self.pop_sim.loc[t-1,:]
            if t==year:
                npr_pop = self.npr.loc[year-1,:] * (614577/self.npr.loc[year-1,:].sum())
            else :
                npr_pop = self.npr_sim.loc[t-1,:]
            # births 
            self.pop_sim.loc[t,0] = start_pop.loc[[x for x in range(15,46)]].sum()*self.get_bx(t)
            
            # deal with exit of NPR and load previous pop
            for a in range(1,n_ages):
                # load from previous year, at previous age 
                self.pop_sim.loc[t,a] = start_pop[a-1]
            # now do deaths and other migration
            for a in range(n_ages):
                # deaths 
                deaths = self.pop_sim.loc[t,a]*self.get_mx(t,a)
                # interprovincial migration
                interp = self.pop_sim.loc[t,a]*self.get_interpx(t,a)
                # emigration 
                emigr  = self.pop_sim.loc[t,a]*self.get_ex(t,a)
                # update population
                self.pop_sim.loc[t,a] = self.pop_sim.loc[t,a] - deaths + interp - emigr   
            # NPR transitions 
            npr_exits = np.zeros(n_ages)
            npr_entries = np.zeros(n_ages)
            npr_apply_pr = np.zeros(n_ages)
            for a in range(n_ages):
                if a>0:
                    # exits from NPR (non-renewals)
                    npr_exits[a] = self.get_npr_exits(t,a)
                    # transition from NPR to PR
                    npr_apply_pr[a] = self.get_npr_apply_pr(t,a)
                # entry to NPR
                npr_entries[a] = self.get_npr_entry(t,a) 
            # check caps on permanent immigration with inflow of NPR, adjust if required 
            tot_apply_pr = npr_apply_pr.sum()
            # cap if above cap  
            if tot_apply_pr > self.pr_entry_cap:
                npr_accepted_pr = npr_apply_pr * self.pr_entry_cap/tot_apply_pr
                npr_rejected_pr = tot_apply_pr - npr_accepted_pr
            else :
                npr_accepted_pr = npr_apply_pr
                npr_rejected_pr = np.zeros(n_ages)
            # now update NPR pop
            for a in range(n_ages):
                if a>0:
                    self.npr_sim.loc[t,a] = npr_pop[a-1]
                else :
                    self.npr_sim.loc[t,0] = 0
                self.npr_sim.loc[t,a] -= npr_exits[a]
                self.npr_sim.loc[t,a] -= npr_accepted_pr[a]
                self.npr_sim.loc[t,a] += npr_entries[a]
                self.pop_sim.loc[t,a] = self.pop_sim.loc[t,a] + npr_entries[a] + npr_accepted_pr[a] - npr_exits[a]
            # permanent immigration (from NPR and remaining from abroad)
            immig_space = max(self.pr_entry_cap - npr_accepted_pr.sum(),0)
            if immig_space > 0:
                for a in range(n_ages):
                    self.pop_sim.loc[t,a] += immig_space*self.get_share_ix(t,a) 
        
        self.pop_sim.loc[year-1,:] = self.pop.loc[year-1,:]
        self.npr_sim.loc[year-1,:] = self.npr.loc[year-1,:] * (614577/self.npr.loc[year-1,:].sum())
        return 
                

                


