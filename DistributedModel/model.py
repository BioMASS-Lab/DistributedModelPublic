# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:12:01 2020

@author: cmptrsn2
"""

import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd

# conversions
m2PerAcre = 4046.86
mPerMi = 1609.34
ftPerM = 3.28
minPerH = 60
kmPerMi = 1.60934

#------------------------------------------------------------------------------
# define classes for storing farm data and model results

class farmData():
    def __init__(self, n_true, fields, field_crops, n, n_c, I_c, n_e, a, D_f, T_e, T_fs, T_se):
        self.n_true = n_true
        self.fields = fields
        self.field_crops = field_crops
        self.n = n
        self.n_c = n_c
        self.I_c = I_c
        self.n_e = n_e
        self.a = a
        self.D_f = D_f
        self.T_e = T_e
        self.T_fs = T_fs
        self.T_se = T_se
              
class grainData():
    def __init__(self, y, r_l, theta_w, theta_s, m, s_f, r_d, r_s, c_e, p, p_d, p_s):
        self.y = y
        self.r_l = r_l
        self.theta_w = theta_w
        self.theta_s = theta_s
        self.m = m
        self.s_f = s_f
        self.r_d = r_d
        self.r_s = r_s
        self.c_e = c_e
        self.p = p
        self.p_d = p_d
        self.p_s = p_s        
        
class machineData():
    def __init__(self, v_f, v_r, w_h, e_c, r_m, c_o, c_gc, c_l, c_t, l_t, n_t, t_lf, t_h, t_u):
        self.v_f = v_f
        self.v_r = v_r
        self.w_h = w_h
        self.e_c = e_c
        self.r_m = r_m
        self.c_o = c_o
        self.c_gc = c_gc
        self.c_l = c_l
        self.c_t = c_t
        self.l_t = l_t
        self.n_t = n_t
        self.t_lf = t_lf
        self.t_h = t_h
        self.t_u = t_u
        
class storageData():
    def __init__(self, p_pr, p_e, e_b, e_f, c_s, t_ls, l_s):
        self.p_pr = p_pr
        self.p_e = p_e
        self.e_b = e_b
        self.e_f = e_f
        self.c_s = c_s
        self.t_ls = t_ls
        self.l_s = l_s
        
class results():       
    def __init__(self, profit, X, T, S, S_e, E, M_s, N_s, N_se, N_e):
        self.profit = profit
        self.X = X
        self.T = T
        self.S = S
        self.S_e = S_e
        self.E = E
        self.M_s = M_s
        self.N_s = N_s
        self.N_se = N_se
        self.N_e = N_e
        
class costsByField():
    def __init__(self, C_h_f, C_f_se_f, C_gc_f, C_m_f, C_s_f, C_e_f):
        self.C_h_f = C_h_f
        self.C_f_se_f = C_f_se_f
        self.C_gc_f = C_gc_f
        self.C_m_f = C_m_f
        self.C_s_f = C_s_f
        self.C_e_f = C_e_f

class harvestCosts():
    def __init__(self, R, C_h, C_gc, C_m, C_lf, C_f_se, C_ls, C_se, C_u, C_t, C_s, C_e):
        self.R = R
        self.C_h = C_h
        self.C_gc = C_gc
        self.C_m = C_m
        self.C_lf = C_lf 
        self.C_f_se = C_f_se
        self.C_ls = C_ls 
        self.C_se = C_se
        self.C_u = C_u
        self.C_t = C_t
        self.C_s = C_s
        self.C_e = C_e 

#------------------------------------------------------------------------------
# put model into callable function

def runModel(fp, gp, mp, sp):      
    ### Create model
    dm = gb.Model("DistributedModel")
    
    ### Create decision variables
    
    # indicators
    X = dm.addMVar(shape=((fp.n+1),(fp.n+1),(fp.n+1)), vtype=GRB.BINARY, name ="X") # routes
    
    # time variables
    T = dm.addMVar(shape=fp.n, vtype=GRB.CONTINUOUS, name ="T") # time spent harvesting fields
    
    # allocation fractions
    S = dm.addMVar(shape=(fp.n, mp.n_t), vtype=GRB.CONTINUOUS, name ="S") # grain fraction sent to storage
    S_e = dm.addMVar(shape=(fp.n_c, fp.n_e, mp.n_t), vtype=GRB.CONTINUOUS, name ="Se") # grain fraction sent to endpoint from storage
    E = dm.addMVar(shape=(fp.n, fp.n_e, mp.n_t), vtype=GRB.CONTINUOUS, name ="E") # grain fraction sent to endpoint
    
    # total mass of each crop sent to storage
    M_s = dm.addMVar(shape=fp.n_c, vtype=GRB.CONTINUOUS, name ="Ms")
    
    # number of truck loads
    N_s = dm.addMVar(shape=(fp.n, mp.n_t), vtype=GRB.INTEGER, name ="Ns") # number of trips to storage
    N_se = dm.addMVar(shape=(fp.n_c, fp.n_e, mp.n_t), vtype=GRB.INTEGER, name ="Nse") # number of trips from storage to endpoint
    N_e = dm.addMVar(shape=(fp.n, fp.n_e, mp.n_t), vtype=GRB.INTEGER, name ="Ne") # number of trips to endpoint
      
    ### Set constraints
    
    ## Combine routing constraints
    
    # combine must start at home location
    dm.addConstr(sum(X[0,:,0]) == 1)
    
    # combine must end at home location
    dm.addConstr(sum(X[:,0,fp.n]) == 1)
    
    # combine cannot return to the same parcel
    diags = np.arange(fp.n+1)
    for d in range(fp.n+1):   
        dm.addConstr(X[diags,diags,d] == np.zeros(fp.n+1))
    
    # only one route per trip
    for d in range(fp.n+1):
        dm.addConstr(X[:,:,d].sum() == 1)  
         
    # every field must be harvested once
    for j in range(fp.n+1):
        dm.addConstr(X[:,j,:].sum() == 1)
    
    # combine must leave where it left off
    for d in range(fp.n):
        for j in range(fp.n+1):
            dm.addConstr(sum(X[:,j,d]) == sum(X[j,:,d+1]))
        
    ## In-field harvesting constraints
    
    # Field capacity
    for i in range(fp.n):
        dm.addConstr(T[i] >= sum([fp.I_c[i,c] * (fp.a[i] * m2PerAcre) / ((mp.v_f[c] * mPerMi) * (mp.w_h[c] / ftPerM) * mp.e_c[c]) for c in range(fp.n_c)])) # hours
    
    # Throughput capacity
    for i in range(fp.n):
        dm.addConstr(T[i] >= sum([fp.I_c[i,c] * gp.m[i] / mp.r_m[c] for c in range(fp.n_c)])) # hours
    
    ## Allocation constraints
    
    # all grain must be allocated to storage and the market endpoints
    for i in range(fp.n):
        dm.addConstr(sum([S[i,k] for k in range(mp.n_t)]) + sum([E[i,j,k] for j in range(fp.n_e) for k in range(mp.n_t)]) == gp.m[i])
          
    ## Storage constraints
    
    # mass of crop c sent to storage
    for c in range(fp.n_c):
        dm.addConstr(M_s[c] == sum([S[i,k] * fp.I_c[i,c] * gp.s_f[c] for i in range(fp.n) for k in range(mp.n_t)]))
    
    # total mass must not exceed storage capacity
    dm.addConstr(sum([M_s[c] for c in range(fp.n_c)]) <= sp.l_s)  
    
    # all stored grain must be sold
    for c in range(fp.n_c):
        dm.addConstr(sum([S_e[c,j,k] for j in range(fp.n_e) for k in range(mp.n_t)]) == M_s[c])
              
    # no storage
    if sp.l_s == 0:
        for k in range(mp.n_t):
            dm.addConstr(S[:,k] == np.zeros(fp.n))
            dm.addConstr(N_s[:,k] == np.zeros(fp.n))
        for c in range(fp.n_c):
            for j in range(fp.n_e):
                dm.addConstr(N_se[c,j,:] == np.zeros(mp.n_t))
    
    ## Hauling constraints:
        
    # storage truck loads
    for i in range(fp.n):
        for k in range(mp.n_t):
            dm.addConstr(S[i,k] <= N_s[i,k] * mp.l_t[k])
    
    # endpoint truck loads from fields
    for i in range(fp.n):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                dm.addConstr(E[i,j,k] <= N_e[i,j,k] * mp.l_t[k])
    
    # endpoint truck loads from storage
    for c in range(fp.n_c):
       for j in range(fp.n_e):
           for k in range(mp.n_t):
               dm.addConstr(S_e[c,j,k] <= N_se[c,j,k] * mp.l_t[k])
               
    ### Calculate objective function
    
    ## Revenue (R)
    R = 0
      
    # grain sold directly 
    for i in range(fp.n):
        for c in range(fp.n_c):
            for k in range(mp.n_t):
                for j in range(fp.n_e):
                    R +=  E[i,j,k] * fp.I_c[i,c] * gp.s_f[c] * gp.p_d[c,j]
                
    # grain sold from storage
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                R += S_e[c,j,k] * gp.p_s[c,j]
    
    ## Harvesting cost (C_h)
    C_h = 0
    for i in range(fp.n):
        C_h += mp.c_o * T[i]
    
    ## Grain cart operating cost (C_gc)
    C_gc = 0
    for i in range(fp.n):
        for c in range(fp.n_c):
            C_gc += fp.I_c[i,c] * fp.a[i] * mp.c_gc[c]
    
    ## Machinery transportation cost (C_m)
    C_m = 0
    for i in range(fp.n+1):
        for j in range(fp.n+1):
            for d in range(fp.n+1):
                C_m += X[i,j,d] * fp.D_f[i,j] * mp.c_o / (mp.v_r * kmPerMi)
    C_m += 2 * fp.n * mp.t_h * mp.c_o
             
    ## Grain transportation cost (C_g)
    
    # On-farm grain loading cost (C_lf)
    C_lf = 0
    for i in range(fp.n):
        for k in range(mp.n_t):
            C_lf += mp.c_l * mp.t_lf * N_s[i,k]
            for j in range(fp.n_e):
                C_lf += mp.c_l * mp.t_lf * N_e[i,j,k]
                
    # Grain hauling cost from fields to storage and market endpoints after harvest (C_f_se)
    C_f_se = 0
    for i in range(fp.n):
        for k in range(mp.n_t):
            C_f_se += mp.c_t[k] * N_s[i,k] * fp.T_fs[i]
            for j in range(fp.n_e):
                C_f_se += mp.c_t[k] * N_e[i,j,k] * fp.T_e[i,j]
                
    # Grain loading cost at storage before hauling (C_ls)
    C_ls = 0
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                C_ls += mp.c_l * sp.t_ls * N_se[c,j,k]
    
    # Grain hauling cost from storage to market endpoints (C_se)
    C_se = 0
    for c in range(fp.n_c):
        for j in range(fp.n_e):
           for k in range(mp.n_t):
               C_se += mp.c_t[k] * fp.T_se[j] * N_se[c,j,k]
    
    # Grain unloading cost at storage and market endpoints (C_u)
    C_u = 0
    for i in range(fp.n):
        for k in range(mp.n_t):        
            C_u += mp.c_l * mp.t_u * N_s[i,k]
            for j in range(fp.n_e):
                C_u += mp.c_l * mp.t_u * N_e[i,j,k]
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                C_u += mp.c_l * mp.t_u * N_se[c,j,k]
      
    C_t = C_lf + C_f_se + C_ls + C_se + C_u
    
    ## On-farm storage cost (C_s)
    C_s = 0
    for i in range(fp.n):
        for c in range(fp.n_c):
            for k in range(mp.n_t):
                C_s += (sp.p_pr * sp.e_b + sp.p_e * sp.e_f) * S[i,k] * fp.I_c[i,c] * (gp.theta_w[c] - gp.theta_s[c]) 
                C_s += sp.c_s * S[i,k] * fp.I_c[i,c] * gp.s_f[c]
        
    ## Establishment cost (C_e)
    C_e = 0
    for i in range(fp.n):
        for c in range(fp.n_c):
            C_e += fp.I_c[i,c] * fp.a[i] * gp.c_e[c]
        
    ## Set objective (P)
    dm.setObjective(R - C_h - C_gc - C_m - C_t - C_s - C_e, GRB.MAXIMIZE)
    
    ## Solve optimization problem
    dm.optimize()

    return results(dm.objVal, X.X, T.X, S.X, S_e.X, E.X, M_s.X, N_s.X, N_se.X, N_e.X)

#------------------------------------------------------------------------------
# define function for importing model data and parameters

def getData(year, farm):    
    # set of crops
    crops = ['Corn','Soybeans']
    n_c = len(crops)

    # crop grown each year by field
    cropsGrown = pd.read_csv('RawData/%s/CropsGrown.csv' %farm, index_col = ['Field'])
    allCrops = list(cropsGrown[str(year)]) # list for crops by field in given year
    n_true = len(allCrops) # total number of fields
    
    # identify fields with corn or soybean
    field_inds = [i for i,c in enumerate(allCrops) if c in crops] # field indices
    fields = [i+1 for i,c in enumerate(allCrops) if c in crops] # field numbers
    field_crops = [allCrops[i] for i in field_inds] # crop in each field with corn or soybeans
    n = len(fields) # number of fields in corn or soybeans
    
    # indicators for crops grown in each field
    I_c = np.zeros((n,n_c))
    for i in range(n):
        for c in range(n_c):
            if allCrops[field_inds[i]] == crops[c]:
                I_c[i,c] = 1
                
    # grain yields
    grainYields = pd.read_csv('RawData/%s/GrainYields.csv' %farm, index_col = ['Field'])
    
    # market locations 
    endpoints = list(pd.read_csv('RawData/%s/MarketLocations.csv' %farm, usecols=['Locations'])['Locations'])
    n_e = len(endpoints) # number of endpoints
    
    # field areas
    areas = pd.read_csv('RawData/%s/FieldAreas.txt' %farm) # ac
    a = np.zeros(n)
    for i in range(n):
        f = fields[i]
        ind_f = [k for k,val in enumerate(areas['DESCRIPTION']) if str(f) in val.split()][0] # find index of field f
        a[i] = areas['Area'][ind_f]
    
    # initialize time arrays between fields
    field_dists = pd.read_csv('RawData/%s/FieldDistances.txt' %farm) # mi
    D_f = np.zeros((n+1,n+1)) # km
    for i in range(n+1):
        ind_f1 = [k for k,val in enumerate(field_dists['Name']) if str(i) in val.split('-')[0].split()] # find indices of rows with field f1
        row_f1 = list(field_dists['Name'][ind_f1]) # isolate rows with field f1
        dist_f1 = list(field_dists['Total_Length'][ind_f1]) # get dists to and from field f1
        for j in range(i+1,n+1): # loop over all other fields
            ind_12 = [k for k,val in enumerate(row_f1) if str(j) in val.split('-')[1].split()][0] # index for distance between fields f1 and f2
            d_12 = dist_f1[ind_12] # dist between fields f1 & f2 (mi)
            D_f[i,j] = d_12*kmPerMi
            D_f[j,i] = d_12*kmPerMi
            
    # initialize time arrays from fields to elevators        
    endpoint_times = pd.read_csv('RawData/%s/EndpointTimes.txt' %farm) # min
    T_e = np.zeros((n,n_e)) # h
    for i in range(n_e):
        endpt = endpoints[i]
        ind_e = [k for k,val in enumerate(endpoint_times['Name']) if all(e in val.split() for e in endpt.split())] # indices with endpt
        row_e = list(endpoint_times['Name'][ind_e])
        time_e = list(endpoint_times['Total_TravelTime'][ind_e])
        for j in range(n):
            f = fields[j]
            ind_f = [k for k,val in enumerate(row_e) if str(f) in val.split()][0] # find rows with field j   
            te_j = time_e[ind_f]/minPerH # time between field j and elevator i (h)
            T_e[j,i] = te_j
    
    # initialize time arrays from fields to storage
    T_fs = np.zeros(n) # h
    if farm == 'GregsFarm':
        ind_s = [k for k,val in enumerate(endpoint_times['Name']) if 'Storage' in val.split()]
        row_s = list(endpoint_times['Name'][ind_s])
        time_s = list(endpoint_times['Total_TravelTime'][ind_s])
        for i in range(n):
            f = fields[i]
            ind_fs = [k for k,val in enumerate(row_s) if str(f) in val.split()][0]
            T_fs[i] = time_s[ind_fs]/minPerH
        
    # initialize time arrays from storage to elevators
    T_se = np.zeros(n_e)
    if farm == 'GregsFarm':
        for i in range(n_e):
            endpt = endpoints[i]
            ind_se = [i for i,val in enumerate(endpoint_times['Name']) if 'Storage' in val.split() and all(e in val.split() for e in endpt.split())][0]
            T_se[i] = endpoint_times['Total_TravelTime'][ind_se]/minPerH
     
    ## make variables for parameters independent of crop
    
    # grain data
    grainDf = pd.read_csv('RawData/%s/GrainData.csv' %farm, index_col = ['Parameter'])
    theta_w = grainDf.loc['Harvest moisture content',:] # harvest moisture content
    theta_s = grainDf.loc['Market moisture content',:] # market moisture content
    r_l = grainDf.loc['Grain loss rate',:] # grain loss rate
    r_s = grainDf.loc['Shrink rate',:] # shrink rate
    r_d = grainDf.loc['Discount rate',:] # discount rate
    c_e = grainDf.loc['Establishment cost',:] # establishment cost
    
    # price data
    directPrices = pd.read_csv('RawData/%s/DirectPriceData.csv' %farm, index_col = ['Endpoint'])
    storagePrices = pd.read_csv('RawData/%s/StoragePriceData.csv' %farm, index_col = ['Endpoint'])
    
    # machinery data
    machineDf = pd.read_csv('RawData/%s/MachineryData.csv' %farm, index_col = ['Parameter'])       
    v_f = machineDf.loc['Field speed',:] # speed 
    v_r = machineDf.loc['Road speed',:][0] # speed 
    w_h = machineDf.loc['Width',:] # width
    e_c = machineDf.loc['Efficiency',:] # efficiency
    r_m = machineDf.loc['Throughput capacity',:] # max throughput rate
    c_o = machineDf.loc['Operating cost','Corn'] # $/h
    t_h = machineDf.loc['Header adjustment time','Corn']/minPerH # h
    t_lf = machineDf.loc['Loading time','Corn']/minPerH # h
    t_u = machineDf.loc['Unloading time','Corn']/minPerH # h
    c_l = machineDf.loc['Labor cost','Corn'] # $/h
    c_gc = machineDf.loc['Grain cart operating cost',:] # grain cart operating cost
    
    # truck data 
    truckDf = pd.read_csv('RawData/%s/TruckData.csv' %farm, index_col = ['Parameter'])
    c_t = np.array(truckDf.loc['Operating cost',:]) # $/h
    l_t = np.array(truckDf.loc['Capacity',:]) # bu
    n_t = len(l_t) # number of trucks
    
    # storage data
    storageDf = pd.read_csv('RawData/%s/StorageData.csv' %farm, index_col = ['Parameter'])
    p_pr = float(storageDf.loc['Propane price',:]) # $/gal
    p_e = float(storageDf.loc['Electricity price',:]) # $/kWh
    e_b = float(storageDf.loc['Energy use of gas-fired burner',:]) # gal/bu/pt
    e_f = float(storageDf.loc['Energy use of drying fan',:]) # kWh/bu/pt
    c_s = float(storageDf.loc['Stored grain management cost',:]) # $/bu
    t_ls = float(storageDf.loc['Storage loading time',:])/minPerH # h
    l_s = float(storageDf.loc['Storage capacity',:]) # bu
    
    ## make arrays for parameters that depend on the crop
    
    # grain data
    y = np.zeros(n) # grain yield
    m = np.zeros(n) # total harvested wet mass    
    for i in range(n):
        f = field_inds[i]
        y[i] = grainYields.loc[f+1,allCrops[f]]
        for c in range(n_c):
            m[i] += I_c[i,c] * a[i] * y[i] * (1 - r_l[c]/100) * (1 - theta_s[c]/100)/(1 - theta_w[c]/100) # bu
            
    # grain price if sold directly
    p = np.zeros((n_c,n_e)) 
    for c in range(n_c):
        for j in range(n_e):
            p[c,j] = directPrices.loc[endpoints[j],crops[c]]
                      
    # price if grain sold directly after discount
    p_d = np.zeros((n_c,n_e))
    for c in range(n_c):
        for j in range(n_e):
            p_d[c,j] = p[c,j] - r_d[c] * (theta_w[c] - theta_s[c]) / 100
     
    # grain price if held in storage
    p_s = np.zeros((n_c,n_e)) # price 
    for c in range(n_c):
        for j in range(n_e):
            p_s[c,j] = storagePrices.loc[endpoints[j],crops[c]]
            
    # shrink factor
    s_f = (1 - r_s * (theta_w - theta_s) / 100)
              
    # store variables in classes
    farmPars = farmData(n_true, fields, field_crops, n, n_c, I_c, n_e, a, D_f, T_e, T_fs, T_se)
    grainPars = grainData(y, r_l, theta_w, theta_s, m, s_f, r_d, r_s, c_e, p, p_d, p_s)
    machinePars = machineData(v_f, v_r, w_h, e_c, r_m, c_o, c_gc, c_l, c_t, l_t, n_t, t_lf, t_h, t_u)
    storagePars = storageData(p_pr, p_e, e_b, e_f, c_s, t_ls, l_s)

    return farmPars, grainPars, machinePars, storagePars
        
#------------------------------------------------------------------------------
# define functions for analyzing model results    
 
# returns several components of total cost       
def getCosts(r, fp, gp, mp, sp):
    ## Revenue (R)
    R = 0
    
    # grain sold directly 
    for i in range(fp.n):
        for c in range(fp.n_c):
            for k in range(mp.n_t):
                for j in range(fp.n_e):
                    R +=  r.E[i,j,k] * fp.I_c[i,c] * gp.s_f[c] * gp.p_d[c,j]
                
    # grain sold from storage
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                R += r.S_e[c,j,k] * gp.p_s[c,j]
    
    ## Harvesting cost (Ch)
    C_h_f = r.T * mp.c_o
    C_h = sum(C_h_f)
    
    ## Grain cart operating cost (Cgc)
    C_gc_f = np.zeros(fp.n)
    for i in range(fp.n):
        for c in range(fp.n_c):
            C_gc_f[i] += fp.I_c[i,c] * fp.a[i] * mp.c_gc[c]
    C_gc = sum(C_gc_f)
    
    ## Machinery transportation cost (Cm)
    C_m_f = np.zeros(fp.n)
    for j in range(1,fp.n+1):
        for i in range(fp.n+1):
            for d in range(fp.n+1):
                C_m_f[j-1] += r.X[i,j,d] * fp.D_f[i,j] * mp.c_o / (mp.v_r * kmPerMi)
        C_m_f[j-1] += 2 * mp.t_h * mp.c_o
    C_m = sum(C_m_f)
          
    ## Grain transportation cost (Cg)
    
    # On-farm grain loading cost
    C_lf = 0
    for i in range(fp.n):
        for k in range(mp.n_t):
            C_lf += mp.c_l * mp.t_lf * (r.N_s[i,k] + r.N_e[i,:,k].sum())
        
    # Grain hauling cost from fields to storage and market endpoints after harvest
    C_f_se_f = np.zeros(fp.n)
    for i in range(fp.n):
        for k in range(mp.n_t):
            C_f_se_f[i] += mp.c_t[k] * (r.N_s[i,k] * fp.T_fs[i] + r.N_e[i,:,k] @ fp.T_e[i,:])
    C_f_se = sum(C_f_se_f)
    
    # Grain loading cost at storage before hauling 
    C_ls = 0
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                C_ls += mp.c_l * sp.t_ls * r.N_se[c,j,k]
    
    # Grain hauling cost from storage to market endpoints
    C_se = 0
    for c in range(fp.n_c):
        for j in range(fp.n_e):
           for k in range(mp.n_t):
               C_se += mp.c_t[k] * fp.T_se[j] * r.N_se[c,j,k]
    
    # Grain unloading cost at storage and the market endpoints  
    C_u = 0
    for i in range(fp.n):
        for k in range(mp.n_t):        
            C_u += mp.c_l * mp.t_u * (r.N_s[i,k] + r.N_e[i,:,k].sum())
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            for k in range(mp.n_t):
                C_u += mp.c_l * mp.t_u * r.N_se[c,j,k]
    C_t = C_lf + C_f_se + C_ls + C_se + C_u
    
    ## On-farm storage cost (Cs)
    C_s_f = np.zeros(fp.n)
    for i in range(fp.n):
        for c in range(fp.n_c):
            for k in range(mp.n_t):
                C_s_f[i] += (sp.p_pr * sp.e_b + sp.p_e * sp.e_f) * r.S[i,k] * fp.I_c[i,c] * (gp.theta_w[c] - gp.theta_s[c])
                C_s_f[i] += sp.c_s * r.S[i,k] * fp.I_c[i,c] * gp.s_f[c]
    C_s = sum(C_s_f)
    
    ## Establishment cost (Ce)
    C_e_f = np.zeros(fp.n)
    for i in range(fp.n):
        for c in range(fp.n_c):
            C_e_f[i] += fp.I_c[i,c] * fp.a[i] * gp.c_e[c]
    C_e = sum(C_e_f)
    
    return harvestCosts(R, C_h, C_gc, C_m, C_lf, C_f_se, C_ls, C_se, C_u, C_t, C_s, C_e), costsByField(C_h_f, C_f_se_f, C_gc_f, C_m_f, C_s_f, C_e_f)

# function that reconstructs routes
def getRoutes(rDict, years, farm):    
    pos = {}
    for year in years:
        fp, gp, mp, sp = getData(year, farm)
        pos[year] = [0]
        for d in range(fp.n+1):
            for i in range(fp.n+1):
                for j in range(fp.n+1):
                    if round(rDict[year].X[i,j,d]) == 1:
                            pos[year].append(j)
    return pos

# function that get field locations
def getFieldLocations(farm):  
    fp, gp, mp, sp = getData(2018, farm) # set baseline as 2018
    field_locations = pd.read_csv('RawData/%s/FieldLocations.txt' %farm)
    home_location = pd.read_csv('RawData/%s/HomeLocation.txt' %farm)
    lons = np.zeros(fp.n+1)
    lats = np.zeros(fp.n+1)
    
    # get home locations
    lons[0] = home_location['Lon']
    lats[0] = home_location['Lat']
    
    # get field locations
    for i in range(len(field_locations)):
        for j in range(fp.n_true):
            if str(j+1) in field_locations['NAME'][i].split(' '):
                lons[j+1] = field_locations['Lon'][i]
                lats[j+1] = field_locations['Lat'][i]
    return lons, lats

# function that saves start and end positions to dataframe
def fieldRouteDf(positions, year, farm, lats, lons):
    fp, gp, mp, sp = getData(year, farm)
    field_route = pd.DataFrame(np.zeros((fp.n,7)),columns=['trip','start_field','end_field','start_lon','start_lat','end_lon','end_lat'])   
    field_route.loc[0,'start_field'] = positions[year][0]
    field_route.loc[0,'end_field'] = fp.fields[positions[year][1]-1] 
    field_route.loc[0,'start_lon'] = lons[0]
    field_route.loc[0,'start_lat'] = lats[0]
    field_route.loc[0,'end_lon'] = lons[fp.fields[positions[year][1]-1]]
    field_route.loc[0,'end_lat'] = lats[fp.fields[positions[year][1]-1]]
    for i in range(1,fp.n):
        f1 = fp.fields[positions[year][i]-1]
        f2 = fp.fields[positions[year][i+1]-1]
        field_route.loc[i,'start_field'] = f1
        field_route.loc[i,'end_field'] = f2
        field_route.loc[i,'start_lon'] = lons[f1]
        field_route.loc[i,'start_lat'] = lats[f1]
        field_route.loc[i,'end_lon'] = lons[f2]
        field_route.loc[i,'end_lat'] = lats[f2]        
    field_route.loc[0,'trip'] = 1 # fp.n = number of trips & fields 
    # adjust field trip numbers for farms in the same loading points
    for i in range(1,fp.n):
        if field_route.loc[i,'start_lon'] == field_route.loc[i-1,'start_lon']:
            field_route.loc[i,'trip'] = field_route.loc[i-1,'trip']
        else:
            field_route.loc[i,'trip'] = field_route.loc[i-1,'trip']+1
    field_route.to_csv('Results/%s/OptimalFieldRoute_%s_%d.csv' %(farm, farm, year), index = False)

    
# function that saves harvest times and number of elevator trips in dataframes
def fieldResultDf(rDict, year, farm):
    fp, gp, mp, sp = getData(year, farm)
    field_stat = pd.DataFrame()
    field_stat['Field'] = fp.fields # field number
    field_stat['Elevator Trips'] = [rDict[year].N_e[i,:,:].sum() + rDict[year].N_s[i,:].sum() for i in range(fp.n)]
    field_stat['Harvest Time (h)'] = [rDict[year].T[i] for i in range(fp.n)]
    field_stat.to_csv('Results/%s/ResultsbyField_%s_%d.csv' %(farm, farm, year), index = False)
 
## functions that updated derived variables

# update wet grain mass
def updateGrainMass(fp, gp):
    for i in range(fp.n):
        gp.m[i] = 0
        for c in range(fp.n_c):
            gp.m[i] += fp.I_c[i,c] * fp.a[i] * gp.y[i] * (1 - gp.r_l[c]/100) * (1 - gp.theta_s[c]/100)/(1 - gp.theta_w[c]/100) # bu
    return gp
                           
# update price if grain sold directly after discount
def updateDiscountPrice(fp, gp):
    for c in range(fp.n_c):
        for j in range(fp.n_e):
            gp.p_d[c,j] = gp.p[c,j] - gp.r_d[c] * (gp.theta_w[c] - gp.theta_s[c]) / 100
    return gp

# update shrink factor
def updateShrink(gp):
    gp.s_f = (1 - gp.r_s * (gp.theta_w - gp.theta_s) / 100)
    return gp

             


