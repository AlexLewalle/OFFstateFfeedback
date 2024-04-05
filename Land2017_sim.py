
# %% Imports
from gpytGPE.utils.design import lhd
from os.path import expanduser
import seaborn as sns
from GPErks.train.emulator import GPEmulator
from GPErks.gp.experiment import GPExperiment
from torchmetrics import MeanSquaredError, R2Score
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from GPErks.utils.random import set_seed
from GPErks.log.logger import get_logger
from GPErks.perks.gsa import SobolGSA
from GPErks.gp.data.dataset import Dataset
import torch
from sklearn.decomposition import PCA
from Coppini import Coppini_Cai
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, ode
# from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import scipy.optimize
from scipy.optimize import curve_fit, least_squares
import math
import pandas as pd
import pickle
import ALpython as AL

import time

# import lhsmdu        <- old latin hypercube module
# lhsmdu.setRandomSeed(1)

home = expanduser('~')


from scipy.optimize import minimize
import copy

# %% GPE imports
GPElog = get_logger()
GPEseed = 8
set_seed(GPEseed)
# from GPErks.train.early_stop import GLEarlyStoppingCriterion

# %% Utility functions


def PoMint(PoM1):
    if PoM1 == '+':
        return +1
    elif PoM1 == '-':
        return -1


def PlotLenBasis(LenBasisPCA):
    # Min = 0
    # Max = 0
    fig, ax = plt.subplots(nrows=len(LenBasisPCA.components_))
    for j in range(len(ax)):
        ax[j].plot(t, LenBasisPCA.components_[j])
        Min, Max = ax[j].get_ylim()
        if Min > 0:
            Min = 0
        if Max < 0:
            Max = 0
        ax[j].set_ylim((Min, Max))
        ax[j].plot([t[0], t[-1]], [0, 0], 'k--')
        ax[j].set_ylabel(f'PC {j}')


t = np.linspace(0., 0.5, 100)

# %% Directory locations

CohortSaveDir = f'{home}/Dropbox/Python/BHFsim/'
CohortFileBase = 'LandCohort_saved_'

# %% Get Lengthening PCA components


# import GetKYdata_0 as KY
# LenBasisPCA = KY.DoPCA()
# LenComponents = LenBasisPCA.components_
# t = np.array(list(DF.columns)); t -= t[0]
# with open(CohortSaveDir + 'LenComponents.dat', 'wb') as fpickle:
#     pickle.dump( LenBasisPCA , fpickle)


# %%


AllParams = ('a',
             'b',
             'k',
             'eta_l',
             'eta_s',
             'k_trpn_on',
             'k_trpn_off',
             'ntrpn',
             'pCa50ref',
             'ku',
             'nTm',
             'trpn50',
             'kuw',
             'kws',
             'rw',
             'rs',
             'gs',
             'es', # added 19/11/23
             'gw',
             'phi',
             'Aeff',
             'beta0',
             'beta1',
             'Tref',
             'k2',
             'k1',
             'koffon')


ifkforce = True


# t = np.arange(0., 1., 0.001)
pCa_list = [4.5, 5.0, 5.8, 6.0, 6.2, 6.4, 6.6]
fracdSL_list = [-0.02, -0.01, 0.01, 0.02]
# LenPcl = {
#     'active': [    {'dLambda': dLambda1, 'pCa': pCa1}
#                for pCa1 in
#                 for dLambda1 in [0.02, 0.01] ],
#     'passive': [   {'dLambda': dLambda1}
#                 for dLambda1 in [0.02, 0.01] ]   }


def HillFn(x, ymax, n, ca50):
    return ymax * x**n/(x**n + ca50**n)


# %% Model definition
class Land2017:

    # Passive tension parameters
    a = 2.1e3 # Pa
    b = 9.1  # dimensionless
    k = 7  # dimensionless
    eta_l = 0.2  # s
    eta_s = 20e-3  # s

    # Active tension parameters
    k_trpn_on =  0.1e3  # /s
    k_trpn_off = 0.1e3 #0.03e3  # /s
    ntrpn = 2  # dimensionless
    pCa50ref = -np.log10(2.5e-6)  # M
    ku = 1000  # /s
    nTm = 2.2  # dimensionless
    # dimensionless (CaTRPN in Eq.9 is dimensionless as it represents a proportion)
    trpn50 = 0.35
    kuw = 0.026e3  # /s
    kws = 0.004e3 * 1# /s
    rw = 0.5
    rs = 0.25
    gs = 0.0085e3  # /s (assume "/ms" was omitted in paper)
    es = 1. # added 19/11/23 to mend asymmetry between pos and neg QS ( should be in [0,1])
    gw = 0.615e3  # /s (assume "/ms" was omitted in paper)
    phi = 2.23
    Aeff = 25
    beta0 = 2.3
    beta1 = -2.4e-6  # AL modified 25/8/22
    Tref = 40.5e3  # Pa

    k2 = 20 # 200 # s-1    (<-- k2 in Campbell2020; rate constant from unattached "ON" state to "OFF" state.)
    k1 = 2   # s-1      (<-- k1 in Campbell2020; rate constant from "OFF" state to unattacherd "ON" state.)
    ra = 1.  # Residual rate inserted for mavacamten simulation (6/4/23)
    rb = 1.  # Residual rate inserted for mavacamten simulation (6/4/23)
    koffon = None  # Defined during initialisation, depending on WhichDep
    Dep_k1ork2 = None

    koffon_ref = {'force': 1.74e-3,    # Pa-1    # value missing in Campbell2020 !! This is the value from Campbell2018 for rat.
                  'totalforce': 1.74e-3,
                  'passiveforce': 1.74e-3,
                  'Lambda': 10,
                  'bound': 20,
                  'C': 30}

    def pCai(self, t): return 4.8

    ksu_fac = 1.0

    def kwu(self):
        return self.kuw * (1/self.rw - 1) - self.kws 	# eq. 23

    def ksu(self):
        return self.kws*self.rw*(1/self.rs - 1) * self.ksu_fac  # eq. 24

    def kb(self):
        return self.ku*self.trpn50**self.nTm / (1-self.rs-(1-self.rs)*self.rw)

    SL0 = 1.8
    dLambda_ext = 0.1       # To be specified by the experiment
    Lambda_ext = 1.1          # To be specified by the experiment

    def dLambdadt_fun(self, t):
        return 0    # This gets specified by particular experiments

    # Aw = Aeff * rs/((1-rs)*rw + rs) 		# eq. 26
    # As = Aw
    def Aw(self):
        return self.Aeff * self.rs/((1-self.rs)*self.rw + self.rs) 		# eq. 26

    def As(self):
        return self.Aw()

    def __init__(self, PSet1=None, WhichDep='force', Dep_k1ork2='k1'):  

        self.WhichDep = WhichDep
        self.Dep_k1ork2 = Dep_k1ork2
        self.koffon = self.koffon_ref[WhichDep]

        if type(PSet1) == type(None):
            PSet1 = pd.Series({par1: 1. for par1 in AllParams})

        # Initialise all parameters to reference value by default.
        self.PSet = pd.Series({par1: 1. for par1 in AllParams})

        if type(PSet1) == pd.core.series.Series:
            for par1 in PSet1.index:
                assert par1 in self.PSet.index, f'Unknown parameter specified: {par1}'
                self.PSet[par1] = PSet1[par1]

        for param in PSet1.index:
            if param == 'koffon':
                setattr(self, param, self.koffon * PSet1[param])
            else:
                setattr(self, param, getattr(Land2017, param) * PSet1[param])

        self.ExpResults = {}    # Initialise experimental results
        self.Features = {}

    def pCa50(self, Lambda):
        Ca50ref = 10**-self.pCa50ref
        # Ca50 = Ca50ref * (1 + self.beta1*(min(Lambda, 1.2)-1))          # wrong???
        Ca50 = Ca50ref + self.beta1*(np.minimum(Lambda, 1.2)-1)
        if np.size(Ca50) > 1:
            if any(np.array(Ca50)<0):
                for j in range(len(Ca50)):
                    if Ca50[j] <=0:
                        Ca50[j] = np.nan
        return -np.log10(Ca50)
        # if np.size(Ca50)==1: 
        #     if Ca50 < 0:
        #         return np.nan
        #     # print((Ca50, Ca50ref, self.beta1, Lambda))
        #     else:
        #         return -np.log10(Ca50)
        # else:
        #     return np.nan
        # if np.size(Ca50)

    def h(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda_ext
        def hh(Lambda):
            return 1 + self.beta0*(Lambda + np.minimum(Lambda, 0.87) - 1.87)
        return np.maximum(0, hh(np.minimum(Lambda, 1.2)))

    def Ta(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref/self.rs * (S*(Zs+1) + W*Zw)

    def Ta_S(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref/self.rs * (S*(Zs+1))

    def Ta_W(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref/self.rs * (W*Zw)

    def F1(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        C = Lambda-1
        return self.a*(np.exp(self.b*C)-1)

    def F2(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.a*self.k*(Lambda-1 - Cd)

    def Ttotal(self, Y):
        return self.Ta(Y) + self.F1(Y) + self.F2(Y)
    
    def Tp(self, Y):
        return self.F1(Y) + self.F2(Y)
        

    def l_bounds(self, WhichParams):
        if WhichParams == 'AllParams':
            WhichParams = AllParams
        return [self.ParBounds[param1][0] for param1 in WhichParams]

    def u_bounds(self, WhichParams):
        if WhichParams == 'AllParams':
            WhichParams = AllParams
        return [self.ParBounds[param1][1] for param1 in WhichParams]

    def Ta_ss(self, pCai=None):
        # if pCai is None:
        #     pCai = self.pCai(0)
        # CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        U_ss = self.U_ss(pCai)
        return self.h(self.Lambda_ext) * self.Tref/self.rs * self.kws*self.kuw/self.ksu()/(self.kwu()+self.kws) * U_ss
    
    def U_ss(self, pCai=None):
        if pCai is None:
            pCai = self.pCai(0)
        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        U_ss = 1.0 - UE_ss - B_ss - BE_ss - W_ss - S_ss
        return U_ss
        
    def Kub(self, pCai=None):
        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        Kub = self.ku/self.kb() * CaTRPN_ss**(+self.nTm)
        return Kub
    
    def Tp_ss(self, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda_ext
        return self.a * (np.exp(self.b*(Lambda-1)) - 1)

    def Get_ss(self, pCai=None):
        assert self.Dep_k1ork2 is not None
        
        if pCai is None:
            pCai = self.pCai(0)
        if isinstance(pCai, int):
            pCai = float(pCai)
        if isinstance(pCai, np.ndarray) or isinstance(pCai, list):
            return np.array([self.Get_ss(pCai1) for pCai1 in pCai]).transpose()

        Lambda_ss = self.Lambda_ext
        Cd_ss = Lambda_ss - 1
        Zw_ss = 0
        Zs_ss = 0

        CaTRPN_ss = (self.k_trpn_off/self.k_trpn_on * (10**-pCai /
                     10**-self.pCa50(self.Lambda_ext))**-self.ntrpn + 1)**-1
        KE = self.k1 / self.k2
        Kub = self.ku/self.kb() * CaTRPN_ss**(+self.nTm)

        Q = self.kws*self.kuw/self.ksu()/(self.kwu()+self.kws) + \
            self.kuw/(self.kwu()+self.kws)
            

        if self.WhichDep in ['force', 'bound', 'totalforce']:
            if self.WhichDep in ['force', 'totalforce']:
                """
                'force' and 'totalforce' share the same mu :   Ta = mu * U 
                'totalforce' just has an additional kfFp term.
                """
                mu = self.koffon * self.h(self.Lambda_ext) * self.Tref / \
                    self.rs * self.kws*self.kuw/self.ksu()/(self.kwu()+self.kws)
            elif self.WhichDep == 'bound':
                """
                This mu differs from that of 'force' and 'totalforce'.
                This mu gives contains only the coefficients of W and S relative to U.
                """
                mu = self.koffon*self.kuw / \
                    (self.kwu()+self.kws)*(1+self.kws/self.ksu())

            """Solve quadratic for U_ss:
            """
            if self.WhichDep in ['totalforce', 'passiveforce']:
                kfFp = self.koffon*self.Tp_ss()   # Passive force contribution to feedback
            elif self.WhichDep in ['force', 'bound']:
                kfFp = 0   # Passive force term not existent for these scenarios
            
            # ORIGINAL  BEFORE ra, rb ====
            # aa = mu*(1+1/Kub+Q)
            # bb = (1+1/Kub)*(1+kfFp + 1/KE)  + (1+kfFp)*Q - mu
            # cc = -(1+kfFp)

            # if self.Dep_k1ork2 == 'k1':            
            #     aa = self.r2 * mu*(1+1/Kub+Q)
            #     bb = 1/KE*(1+1/Kub) - self.r2*mu + (1+1/Kub+Q)*self.r2*(self.r1+kfFp)
            #     cc = -self.r2*(self.r1+kfFp)
            # elif self.Dep_k1ork2 == 'k2':            
            aa = self.ra * mu*(1+1/Kub+Q)
            bb = 1/KE*(1+1/Kub) - self.ra*mu + (1+1/Kub+Q)*self.ra*(self.rb+kfFp)
            cc = -self.ra*(self.rb+kfFp)

            # When this value is small, Taylor-expand the quadratic to avoid numerical error (subtraction of almost-identical large numbers)
            SmallUCriterion = -4*aa*cc/bb**2  
            if SmallUCriterion > 1e-3:
                U_ss =  (-bb + np.sqrt(bb**2 - 4*aa*cc))/2/aa
            else:
                # U_ss =  (1+kfFp)/bb * (1 - mu*(1+1/Kub+Q)*(1+kfFp)/bb**2)  # ORIGINAL  BEFORE ra, rb ====
                # if self.Dep_k1ork2 == 'k1':
                #     U_ss =  self.r2*(self.r1+kfFp)/bb * (1 - mu*self.r2**2*(1+1/Kub+Q)*(self.r1+kfFp)/bb**2)
                # if self.Dep_k1ork2 == 'k2':
                U_ss =  self.ra*(self.rb+kfFp)/bb * (1 - mu*self.ra**2*(1+1/Kub+Q)*(self.rb+kfFp)/bb**2)
                    
            # if self.Dep_k1ork2 == 'k1':
            #     UE_ss = 1/KE / (self.r1 + mu*U_ss + kfFp)/self.r2 * U_ss
            # elif self.Dep_k1ork2 == 'k2':
            UE_ss = 1/KE / (self.rb + mu*U_ss + kfFp)/self.ra * U_ss
            BE_ss = 1/Kub * UE_ss

        if self.WhichDep == 'Lambda':
        # !! Not updated for r1, r2
            U_ss = ((1+1/Kub)*(1+1/KE/(1+self.koffon*self.Lambda_ext)) + Q) ** -1
            UE_ss = 1/KE / (1+self.koffon*self.Lambda_ext) * U_ss
            BE_ss = 1/Kub * UE_ss
            
        if self.WhichDep == 'C':
        # !! Not updated for r1, r2
            U_ss = ((1+1/Kub)*(1+1/KE/(1+self.koffon*(self.Lambda_ext-1))) + Q) ** -1
            UE_ss = 1/KE / (1+self.koffon*(self.Lambda_ext-1)) * U_ss
            BE_ss = 1/Kub * UE_ss
            
        if self.WhichDep == 'passiveforce':
        # !! Not updated for r1, r2
            Tp = self.a*(np.exp(self.b*(self.Lambda_ext-1))-1)
            U_ss = ((1+1/Kub)*(1+1/KE/(1+self.koffon*self.Tp_ss(self.Lambda_ext))) + Q) ** -1
            UE_ss = 1/KE / (1+self.koffon*Tp) * U_ss
            BE_ss = 1/Kub * UE_ss

        B_ss = 1/Kub * U_ss
        W_ss = self.kuw/(self.kwu()+self.kws) * U_ss
        S_ss = self.kws/self.ksu() * self.kuw/(self.kwu()+self.kws) * U_ss

                  
        # assert np.abs(1-(U_ss+UE_ss+B_ss+BE_ss+S_ss+W_ss)) != np.nan and np.abs(1-(U_ss+UE_ss+B_ss+BE_ss+S_ss+W_ss)) < 0.0001   , \
        #     'state probabilities don'' t add up to 1!!'
        return np.array([CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss], dtype=float)

    # %% ODE system

    def gwu(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return self.gw * abs(Zw)      # eq. 15

    def gsu(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        if Zs < -self.es:
            return -self.gs * (Zs + self.es)
        elif Zs >0:
            return self.gs * Zs
        else:
            return 0
        
        # WRONG: NEED gsu=0 at Zs=0 since Zs=0 in steady state; otherwise S decays even in steady state!!
        # if Zs+1 < 0:        # eq. 17
        #     return self.gs*(-Zs-1)        
        # elif Zs > self.es-1:
        #     return self.gs*(Zs + 1-self.es)
        # else:
        #     return 0
        


    #    CHANGED 30/1/23!!!! ==================
    # def cw(self, Y):
    #     CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
    #     return self.phi * self.kuw * self.U(Y)/W

    # def cs(self, Y):
    #     CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
    #     return self.phi * self.kws * W/S
    
    def cw(self, Y=None):
        return self.phi* self.kuw *((1-self.rs)*(1-self.rw))/((1-self.rs)*self.rw)
    def cs(self, Y=None):
        return self.phi * self.kws *((1-self.rs)*self.rw)/self.rs 
    
    # ===========================================

    def U(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.T
        return 1-B-S-W - BE-UE

    def k1_fd(self, Y):
        if self.Dep_k1ork2 == 'k1':
            CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
            assert self.WhichDep in ['totalforce', 'force', 'passiveforce', 'Lambda', 'bound', 'W', 'S', 'C']
            if self.WhichDep == 'force':
                return self.k1*(self.rb+self.koffon*max((self.Ta(Y), 0.)))
            if self.WhichDep == 'totalforce':
                return self.k1*(self.rb+self.koffon*max((self.Ttotal(Y), 0.)))
            if self.WhichDep == 'passiveforce':
                return self.k1*(self.rb+self.koffon*max((self.Tp(Y), 0.)))
            elif self.WhichDep == 'Lambda':
                return self.k1*(self.rb+self.koffon*Lambda)
            elif self.WhichDep == 'bound':
                return self.k1*(self.rb+self.koffon*(W+S))
            elif self.WhichDep == 'W':
                return self.k1*(self.rb+self.koffon*(W))
            elif self.WhichDep == 'S':
                return self.k1*(self.rb+self.koffon*(S))
            elif self.WhichDep == 'C':
                return self.k1*(self.rb+self.koffon*(Lambda-1))
        elif self.Dep_k1ork2 == 'k2':
            return self.k1*self.ra
    
    def k2_fd(self, Y):
        if self.Dep_k1ork2 == 'k2':
            CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
            assert self.WhichDep in ['totalforce', 'force', 'passiveforce', 'Lambda', 'bound', 'W', 'S', 'C']
            if self.WhichDep == 'force':
                return self.k2/(self.rb+self.koffon*max((self.Ta(Y), 0.)))
            if self.WhichDep == 'totalforce':
                return self.k2/(self.rb+self.koffon*max((self.Ttotal(Y), 0.)))
            if self.WhichDep == 'passiveforce':
                return self.k2/(self.rb+self.koffon*max((self.Tp(Y), 0.)))
            elif self.WhichDep == 'Lambda':
                return self.k2/(self.rb+self.koffon*Lambda)
            elif self.WhichDep == 'bound':
                return self.k2/(self.rb+self.koffon*(W+S))
            elif self.WhichDep == 'W':
                return self.k2/(self.rb+self.koffon*(W))
            elif self.WhichDep == 'S':
                return self.k2/(self.rb+self.koffon*(S))
            elif self.WhichDep == 'C':
                return self.k2/(self.rb+self.koffon*(Lambda-1))
        elif self.Dep_k1ork2 == 'k1':
            return self.k2/self.ra

    def dYdt(self, Y, t):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y

        dZwdt = self.Aw()*self.dLambdadt_fun(t) - self.cw(Y)*Zw
        dZsdt = self.As()*self.dLambdadt_fun(t) - self.cs(Y)*Zs
        # eq. 9        # kb = self.ku * self.trpn50**self.nTm/ (1 - self.rs - (1-self.rs)*self.rw)     # eq. 25
        dCaTRPNdt = self.k_trpn_on * \
            (10**-self.pCai(t)/10**-self.pCa50(Lambda))**self.ntrpn * \
            (1-CaTRPN) - self.k_trpn_off*CaTRPN
        dBdt = self.kb()*CaTRPN**(-self.nTm/2)*self.U(Y)  \
            - self.ku*CaTRPN**(self.nTm/2)*B  \
            - self.k2_fd(Y)*B   \
            + self.k1_fd(Y) * BE  # eq.10 in Land2017, amended to include myosin off state dynamics
        dWdt = self.kuw*self.U(Y) - self.kwu()*W - \
            self.kws*W - self.gwu(Y)*W     # eq. 12
        dSdt = self.kws*W - self.ksu()*S - self.gsu(Y)*S        # eq. 13

        # New "myosin off" states
        dBEdt = self.kb()*CaTRPN**(-self.nTm/2)*UE  \
            - self.ku*CaTRPN**(self.nTm/2)*BE  \
            + self.k2_fd(Y)*B \
            - self.k1_fd(Y) * BE
        dUEdt = -self.kb()*CaTRPN**(-self.nTm/2)*UE  \
            + self.ku*CaTRPN**(self.nTm/2)*BE  \
            + self.k2_fd(Y)*self.U(Y) \
            - self.k1_fd(Y) * UE

        # Allow this function to be defined within particular experiments
        dLambdadt = self.dLambdadt_fun(t)
        if Lambda-1-Cd > 0:     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    def dYdt_pas(self, Y, t):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y

        dZwdt = 0
        dZsdt = 0
        dCaTRPNdt = 0
        # kb = self.ku * self.trpn50**self.nTm/ (1 - self.rs - (1-self.rs)*self.rw)     # eq. 25
        dBdt = 0
        dWdt = 0
        dSdt = 0
        dBEdt = 0
        dUEdt = 0

        # Allow this function to be defined within particular experiments
        dLambdadt = self.dLambdadt_fun(t)
        if Lambda-1-Cd > 0:     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    # %% Experiments

    def Doktr(self, Lambda=1.1, tmax=10., nt=100, WhichReset=None, ifPlot=False, ifPlotStates=False, ifSave=False, ifKeepStates=False):
        assert WhichReset is not None, 'Specify ktr reset'
        Lambda_save = self.Lambda_ext

        t = np.linspace(0.0, tmax,nt)
        F0_pas = self.F1(self.Get_ss()) + self.F2(self.Get_ss())

        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss()
        U_ss = 1 - B_ss - BE_ss - UE_ss - W_ss - S_ss

        
        #### Reset 1
        """
        All in B
        """
        if WhichReset == 1:
            epsilon = 1.e-10
            B_0 = 1 - 2*epsilon 
            BE_0 = 0  
            UE_0 = 0
            BU_0 = 0.
            S_0 = epsilon
            W_0 = epsilon
            U_0 = 1 - B_0 - BE_0 - UE_0 - W_0 - S_0
            Zs_0 = 0.
            Zw_0 = 0.
            self.Lambda_ext = Lambda
            Lambda_0 = Lambda
            Cd_0 = 0 #Cd_ss    # leave unchanged
            CaTRPN_0 = CaTRPN_ss  # leave unchanged
        

        #### Reset 2
        """
        U, B, UE, BE in steady state
        """
        if WhichReset == 2:
            epsilon = 0# 1e-10
            CaTRPN_0 = CaTRPN_ss  # leave unchanged since Ca is constant
            Kub = self.ku/self.kb()*CaTRPN_0**self.nTm
            KE = self.k1/self.k2
            U_0 = 1/(1+1/KE)/(1+1/Kub)
            B_0 = U_0 / Kub        
            UE_0 = U_0 / KE
            BE_0 = U_0 /KE/Kub
            S_0 = 0.   + epsilon/2
            W_0 = 0.   + epsilon/2
            U_0 -= epsilon
            Zs_0 = 0.
            Zw_0 = 0.
            self.Lambda_ext = Lambda
            Lambda_0 = Lambda
            Cd_0 = 0.# Cd_ss    # leave unchanged
        
            
        #### Reset 3
        """
        W and S dumped into U
        """
        if WhichReset == 3:
            epsilon = 1e-10
            CaTRPN_0 = CaTRPN_ss  # leave unchanged
            Kub = self.ku/self.kb()*CaTRPN_0**self.nTm
            KE = self.k1/self.k2
            U_0 = U_ss + S_ss + W_ss  - epsilon
            B_0 = B_ss      
            UE_0 = UE_ss
            BE_0 = BE_ss
            S_0 = 0.   + epsilon/2
            W_0 = 0.   + epsilon/2
            Zs_0 = 0.
            Zw_0 = 0.
            self.Lambda_ext = Lambda
            Lambda_0 = Lambda
            Cd_0 = Cd_ss    # leave unchanged
        
        
        #### Run experiment
        # print(f'TEST: Total state populations: {B_0+U_0+BE_0+UE_0+S_0+W_0}=1?' )
        # print(f'U_0={U_0:.2f}, B_0={B_0:.2f}, UE_0={UE_0:.2f}, BE_0={BE_0:.2f} --->  U_ss={U_ss:.2f}, B_ss={B_ss:.2f}, UE_ss={UE_ss:.2f}, BE_ss={BE_ss:.2f}')
        Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0,
              Cd_0, BE_0, UE_0]  # update initial state vector

        self.dLambdadt_fun = self.FunZero
        Ysol = odeint(self.dYdt, Y0, t)

        F = self.Ttotal(Ysol)
        F_S = self.Ta_S(Ysol)
        F_W = self.Ta_W(Ysol)

        if ifPlot:
            fig1, ax1 = plt.subplots(nrows=3)

            ax1[0].plot(t, F)
            ax1[0].set_ylabel('F_total')
            ax1[1].plot(t, F_S)
            ax1[1].set_ylabel('F_S')
            ax1[2].plot(t, F_W)
            ax1[2].set_ylabel('F_W')
            fig1.suptitle(f'Land2017  ktr   --> Lambda={Lambda}')
            fig1.tight_layout()
        if ifPlotStates:
            fig2, ax2 = plt.subplots(nrows=7, figsize=(7, 10))
            ax2[0].plot(t, Ysol[:, 0])
            ax2[0].set_ylabel('CaTrpn')
            ax2[1].plot(t, Ysol[:, 1])
            ax2[1].set_ylabel('B')
            ax2[2].plot(t, Ysol[:, 2])
            ax2[2].set_ylabel('S')
            ax2[3].plot(t, Ysol[:, 3])
            ax2[3].set_ylabel('W')
            ax2[4].plot(t, self.U(Ysol))
            ax2[4].set_ylabel('U')
            ax2[5].plot(t, Ysol[:, 8])
            ax2[5].set_ylabel('BE')
            ax2[6].plot(t, Ysol[:, 9])
            ax2[6].set_ylabel('UE')
            # AL.y_axis_equalise_range(ax2)
            fig2.suptitle(f'Land2017  ktr states    --> Lambda={Lambda}');  print('Done!')
            fig2.tight_layout()

        self.Lambda_ext = Lambda_save
        self.ExpResults[f'ktr_{Lambda:.2f}'] = {'t': t,
                                                'F': F,
                                                'Fa': F_S+F_W}
        
        if ifKeepStates:
            self.ExpResults[f'ktr_{Lambda:.2f}']['B'] = Ysol[:,1]
            self.ExpResults[f'ktr_{Lambda:.2f}']['U'] = self.U(Ysol)
            self.ExpResults[f'ktr_{Lambda:.2f}']['W'] = Ysol[:,3]
            self.ExpResults[f'ktr_{Lambda:.2f}']['S'] = Ysol[:,2]
            self.ExpResults[f'ktr_{Lambda:.2f}']['BE'] = Ysol[:,8]
            self.ExpResults[f'ktr_{Lambda:.2f}']['UE'] = Ysol[:,9]

    def Doktr_Ken(self, Lambda=1.1, ifPlot=False, ifKeepData=False,
                  T_reset=0.02, Nsteps=20, T_recovery=0.5):
        
        self.Lambda_ext = Lambda    # Start with the final Lambda before slackening.
        
        #### Slackening
        
        if ifPlot:
            fig = plt.figure(figsize=(6,6))
            nplots = 4
            my_addsubplot = lambda j: fig.add_subplot(nplots, 1, j)
            ax = {'F': my_addsubplot(1), r'$\lambda$': my_addsubplot(2), 'available': my_addsubplot(3), 'bound': my_addsubplot(4)}
        
        [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0] = self.Get_ss()
        
        for jstep in range(Nsteps):
            dZ = -(S_0*(Zs_0+1) + W_0*Zw_0) / (S_0+W_0)
            Zs_0 += dZ
            Zw_0 += dZ
            Lambda_0 += dZ / self.As()
            
            Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0]
            self.dLambdadt_fun = self.FunZero
            t = np.linspace(0., T_reset/Nsteps, 10)
            Ysol0, infodict = odeint(self.dYdt, Y0, t, full_output=True)
            [CaTRPN_sol, B_sol, S_sol, W_sol, Zs_sol, Zw_sol, Lambda_sol, Cd_sol, BE_sol, UE_sol] = Ysol0.T
            U_sol = 1-B_sol-BE_sol-UE_sol-S_sol-W_sol
            F_sol = self.Ta(Ysol0)
            
            if ifPlot:
                ax['F'].plot(t-T_reset +jstep*T_reset/Nsteps, F_sol)
                ax[r'$\lambda$'].plot(t-T_reset +jstep*T_reset/Nsteps, Lambda_sol)
                ax['available'].plot(t-T_reset +jstep*T_reset/Nsteps, U_sol)
                ax['bound'].plot(t-T_reset +jstep*T_reset/Nsteps, W_sol + S_sol)
        
            [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0] = Ysol0[-1]
            
        #### Restretch
        
        Zs_0 += self.As()* (Lambda-Lambda_0)
        Zw_0 += self.As()* (Lambda-Lambda_0)
        Lambda_0 += Lambda-Lambda_0
        
        Y1 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0]
        t = np.linspace(0., T_recovery, 1000)
        Ysol1, infodict = odeint(self.dYdt, Y1, t, full_output=True)
        [CaTRPN_sol, B_sol, S_sol, W_sol, Zs_sol, Zw_sol, Lambda_sol, Cd_sol, BE_sol, UE_sol] = Ysol1.T
        U_sol = 1-B_sol-BE_sol-UE_sol-S_sol-W_sol
        F_sol = self.Ta(Ysol1)
        
        if ifPlot:
            ax['F'].plot(t, F_sol);
            ax[r'$\lambda$'].plot(t, Lambda_sol)
            ax['available'].plot(t, U_sol);
            ax['bound'].plot(t, W_sol + S_sol);
        
        if ifKeepData:
            self.ExpResults[f'ktrKen_Lambda{Lambda:.2f}_pCa{self.pCai(0):.1f}'] = \
                {'t': t, 'F':F_sol, r'$\lambda$':Lambda_sol}
        
        #### Extract ktr
        
        d2Fdt2 = np.diff(F_sol, n=2)
        if any(d2Fdt2>=0):
            imin = next(j for j,d2Fdt2_1 in enumerate(np.flip(d2Fdt2)) if d2Fdt2_1 > 0)
        else:
            imin = len(d2Fdt2)
        

        fend = self.Ta_ss()
        FitFn = lambda  t, f0, ktr : fend + (f0-fend)*np.exp(-ktr*t)    # remove fend from FitFn
        try:
            Fitval, Fitcov = curve_fit(FitFn, t[-imin:-1], F_sol[-imin:-1], 
                                       p0=(F_sol[-imin], 5))  # F_sol[-1],
            ktr = Fitval[1]
            a_ktr = Fitval[0]-fend
            if ifPlot:
                ax['F'].plot(t[-imin:-1], FitFn(t[-imin:-1], *Fitval), 'k--', label=rf'$k_\mathrm{{tr}}={ktr:.1f}~s^{{-1}}$')
                ax['F'].legend(loc='lower right')
        
        except:
            ktr = None
            a_ktr = None
            
        if ifPlot:
            for plotname in ax.keys():
                # AL.y_axis_to_zero(ax[plotname])
                ax[plotname].set_ylabel(plotname)
            # ax['F'].set_ylim((0,100e3))    
            fig.suptitle(rf'pCa {self.pCai(0)}, $\lambda={Lambda}$ ;  $T_\mathrm{{reset}}={T_reset}$ s, $N_\mathrm{{steps}}={Nsteps}$')
            fig.tight_layout()
        return ktr, a_ktr
        

    def Doktr_Ken2(self, Lambda=1.1, ifPlot=False, ifKeepData=False,
                  T_reset=0.02, T_restretch=0.001, Nsteps=20, T_recovery=0.5):
       
        self.Lambda_ext = Lambda    # Start with the final Lambda before slackening.
        
        #### Slackening
        
        if ifPlot:
            fig = plt.figure(figsize=(6,6))
            nplots = 4
            my_addsubplot = lambda j: fig.add_subplot(nplots, 1, j)
            ax = {'F': my_addsubplot(1), r'$\lambda$': my_addsubplot(2), 'available': my_addsubplot(3), 'bound': my_addsubplot(4)}
        
        [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0] = self.Get_ss()
        
        for jstep in range(Nsteps):
            dZ = -(S_0*(Zs_0+1) + W_0*Zw_0) / (S_0+W_0)
            Zs_0 += dZ
            Zw_0 += dZ
            Lambda_0 += dZ / self.As()
            
            Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0]
            self.dLambdadt_fun = self.FunZero
            t = np.linspace(0., T_reset/Nsteps, 10)
            Ysol0, infodict = odeint(self.dYdt, Y0, t, full_output=True)
            [CaTRPN_sol, B_sol, S_sol, W_sol, Zs_sol, Zw_sol, Lambda_sol, Cd_sol, BE_sol, UE_sol] = Ysol0.T
            U_sol = 1-B_sol-BE_sol-UE_sol-S_sol-W_sol
            F_sol = self.Ta(Ysol0)
            
            if ifPlot:
                ax['F'].plot(t-T_reset - T_restretch +jstep*T_reset/Nsteps, F_sol)
                ax[r'$\lambda$'].plot(t-T_reset - T_restretch +jstep*T_reset/Nsteps, Lambda_sol)
                ax['available'].plot(t-T_reset - T_restretch +jstep*T_reset/Nsteps, U_sol)
                ax['bound'].plot(t-T_reset - T_restretch +jstep*T_reset/Nsteps, W_sol + S_sol)
        
            [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0] = Ysol0[-1]
            
        #### Restretch
        t = np.linspace(0., T_restretch, 10)
        self.dLambdadt_fun = lambda t : (Lambda-Lambda_0)/T_restretch    # <--- linear restretch
        Ysol1, infodict = odeint(self.dYdt, Ysol0[-1], t, full_output=True)
        [CaTRPN_sol, B_sol, S_sol, W_sol, Zs_sol, Zw_sol, Lambda_sol, Cd_sol, BE_sol, UE_sol] = Ysol1.T
        U_sol = 1-B_sol-BE_sol-UE_sol-S_sol-W_sol
        F_sol = self.Ta(Ysol1)
        
        if ifPlot:
            ax['F'].plot(t- T_restretch , F_sol)
            ax[r'$\lambda$'].plot(t- T_restretch , Lambda_sol)
            ax['available'].plot(t- T_restretch , U_sol)
            ax['bound'].plot(t- T_restretch , W_sol + S_sol)
            
        #### Tension recovery
        self.dLambdadt_fun = self.FunZero
        t = np.linspace(0., T_recovery, 1000)
        Ysol1, infodict = odeint(self.dYdt, Ysol1[-1], t, full_output=True)
        [CaTRPN_sol, B_sol, S_sol, W_sol, Zs_sol, Zw_sol, Lambda_sol, Cd_sol, BE_sol, UE_sol] = Ysol1.T
        U_sol = 1-B_sol-BE_sol-UE_sol-S_sol-W_sol
        F_sol = self.Ta(Ysol1)
        
        if ifPlot:
            ax['F'].plot(t, F_sol);
            ax[r'$\lambda$'].plot(t, Lambda_sol)
            ax['available'].plot(t, U_sol);
            ax['bound'].plot(t, W_sol + S_sol);
        
        if ifKeepData:
            self.ExpResults[f'ktrKen_Lambda{Lambda:.2f}_pCa{self.pCai(0):.1f}'] = \
                {'t': t, 'F':F_sol, r'$\lambda$':Lambda_sol}
        
        #### Extract ktr
        
        d2Fdt2 = np.diff(F_sol, n=2)
        if any(d2Fdt2>=0):
            imin = next(j for j,d2Fdt2_1 in enumerate(np.flip(d2Fdt2)) if d2Fdt2_1 > 0)
        else:
            imin = len(d2Fdt2)
        

        fend = self.Ta_ss()
        FitFn = lambda  t, f0, ktr : fend + (f0-fend)*np.exp(-ktr*t)    # remove fend from FitFn
        try:
            Fitval, Fitcov = curve_fit(FitFn, t[-imin:-1], F_sol[-imin:-1], 
                                       p0=(F_sol[-imin], 5))  # F_sol[-1],
            ktr = Fitval[1]
            a_ktr = Fitval[0]-fend
            if ifPlot:
                ax['F'].plot(t[-imin:-1], FitFn(t[-imin:-1], *Fitval), 'k--', label=rf'$k_\mathrm{{tr}}={ktr:.1f}~s^{{-1}}$')
                ax['F'].legend(loc='lower right')
        
        except:
            ktr = None
            a_ktr = None
            
        if ifPlot:
            for plotname in ax.keys():
                # AL.y_axis_to_zero(ax[plotname])
                ax[plotname].set_ylabel(plotname)
            # ax['F'].set_ylim((0,100e3))    
            fig.suptitle(rf'pCa {self.pCai(0)}, $\lambda={Lambda}$ ;  $T_\mathrm{{reset}}={T_reset}$ s, $N_\mathrm{{steps}}={Nsteps}$')
            fig.tight_layout()
        return ktr, a_ktr
        

    def DoCaiStep(self, pCai1=7, pCai2=5, ifPlot=False, ifPlotStates=False, ifSave=False):
        pCai_original = self.pCai

        t = np.linspace(0, 10, 1000)

        self.pCai = lambda t: pCai1  # Apply initial Cai

        t = np.linspace(0, 10, 1000)
        Ysol = None
        F0 = [None]*2
        F0_S = [None]*2
        F0_W = [None]*2
        F0_pas = [None]*2
        F = [None]*2
        F_S = [None]*2
        F_W = [None]*2
        F_pas = [None]*2

        F0 = self.Ttotal(self.Get_ss())
        F0_S = self.Ta_S(self.Get_ss())
        F0_W = self.Ta_W(self.Get_ss())
        F0_pas = self.F1(self.Get_ss()) + self.F2(self.Get_ss())

        self.dLambdadt_fun = self.FunZero
        CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0 = self.Get_ss()
        Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0]

        self.pCai = lambda t: pCai2    # Apply new Cai
        Ysol = odeint(self.dYdt, Y0, t)

        F = self.Ttotal(Ysol)
        F_S = self.Ta_S(Ysol)
        F_W = self.Ta_W(Ysol)

        if ifPlot:
            fig1, ax1 = plt.subplots(nrows=3, figsize=(7, 7))
            ax1[0].plot(np.append([-t[-1]/20, 0], t), np.append([F0, F0], F))
            ax1[0].set_ylabel('F_total')
            ax1[1].plot(np.append([-t[-1]/20, 0], t),
                        np.append([F0_S, F0_S], F_S))
            ax1[1].set_ylabel('F_S')
            ax1[2].plot(np.append([-t[-1]/20, 0], t),
                        np.append([F0_W, F0_W], F_W))
            ax1[2].set_ylabel('F_W')
            fig1.suptitle(f'Stepping pCai={pCai1} to {pCai2}')
        if ifPlotStates:
            fig2, ax2 = plt.subplots(nrows=5, figsize=(7, 10))
            ax2[0].plot(t, Ysol[:, 0])
            ax2[0].set_ylabel('CaTrpn')
            ax2[1].plot(t, Ysol[:, 1])
            ax2[1].set_ylabel('B')
            ax2[2].plot(t, Ysol[:, 2])
            ax2[2].set_ylabel('S')
            ax2[3].plot(t, Ysol[:, 3])
            ax2[3].set_ylabel('W')
            ax2[4].plot(t, np.ones(len(Ysol)) -
                        (Ysol[:, 1]+Ysol[:, 2]+Ysol[:, 3]))
            ax2[4].set_ylabel('U')
            fig2.suptitle(f'Cai step - States (pCai={pCai1} to {pCai2})')

        self.pCai = pCai_original      # Reset Cai to what it was before the experiment
        self.ExpResults[f'CaiStep_{pCai1:.2f}_{pCai2:.2f}'] = {'t': t,
                                                               'F': F,
                                                               'params': {'pCai1': pCai1, 'pCai2': pCai2}}
        print(f'   -done CaiStep (pCai {pCai1}->{pCai2})')


    def DoQSP(self, QSBasesPCA, pCa_set=4.5, fracdSL_set = 0.01, ifPlot=False, ifPlotStates=False, ifPlotNormalised=False):
        tmax = QSBasesPCA.tmax
        n_pts = QSBasesPCA.n_pts
        n_components = QSBasesPCA.n_components
        
        pCai_old = self.pCai  # Remember to reset later!!
        def pCai(t): return pCa_set
        self.pCai = pCai

        QSTrace = np.array([])
        t = np.linspace(0, tmax, n_pts)
        for fracdSL1 in [fracdSL_set, -fracdSL_set]:
            dLambda = self.Lambda_ext * fracdSL1
            Yss0 = self.Get_ss(pCa_set)
            F0 = self.Ttotal(Yss0)
            F0_S = self.Ta_S(Yss0)
            F0_W = self.Ta_W(Yss0)
            Fa0 = F0_S + F0_W
            F0_pas = self.F1(Yss0) + self.F2(Yss0)
            Ysol = self.QuickStretchActiveResponse(dLambda, t)  # Ysol = None if integration not successful.                
            self.Lambda_ext -= dLambda

            """
            Ysol is a 1-by-n_pts-by-10 array containing the ODE solutions for the quick lengthening, as functions of time.
            State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE
            """
            if not Ysol is None:
                ifSuccess = True
                F_S = self.Ta_S(Ysol)
                F_W = self.Ta_W(Ysol)
                Fa = F_S + F_W
                QSTrace = np.hstack([QSTrace, (Fa/Fa0-1)/fracdSL_set])
        
        DQSTrace = np.array(np.array(QSTrace), dtype=float) - QSBasesPCA.mean_    
        Proj = np.dot(DQSTrace, QSBasesPCA.components_.T)
        
        # print(f'Proj = {Proj}')
        for jcomp in range(len(Proj)):
            self.Features[f'QSPpc{jcomp}'] = Proj[jcomp]
        self.ExpResults['QSP_residue'] = np.sum((np.dot( Proj, QSBasesPCA.components_)  + QSBasesPCA.mean_ 
                                          - QSTrace)**2 /len(QSTrace) )**0.5
        
        if ifPlot:
            fig, ax = plt.subplots(ncols=2)
            ax[0].plot(QSTrace[:n_pts], 'k--')
            ax[1].plot(QSTrace[n_pts:], 'k--')
            ax[0].plot(np.dot( Proj, QSBasesPCA.components_[:,:n_pts])  + QSBasesPCA.mean_[:n_pts])
            ax[1].plot(np.dot( Proj, QSBasesPCA.components_[:,n_pts:])  + QSBasesPCA.mean_[n_pts:])
            AL.y_axis_equalise_range(ax)

        self.pCai = pCai_old
        return DQSTrace #ifSuccess

    # def DoLengthening(self, pCa_list=[4.5], fracdSL_list = [-0.02, -0.01, 0.01, 0.02], n_pts=501, tmax=0.5, ifPlot=False, ifPlotStates=False, ifPlotNormalised=False):
    #     pCai_old = self.pCai  # Remember to reset later!!
    #     if ifPlot:
    #         fig_forces, ax_forces = plt.subplots(nrows=5)
    #     if ifPlotStates:
    #         fig_states, ax_states = plt.subplots(nrows=9, figsize=(7, 10))
    #     if ifPlotNormalised:
    #         fig_norm, ax_norm = plt.subplots()
    #         ax_norm.set_ylabel(r"$(T_a-T_a^0)\ /\ T_a^0 \delta' SL$")

    #     Result = {}
    #     for pCa in pCa_list:

    #         def pCai(t): return pCa
    #         self.pCai = pCai

    #         t = np.linspace(0, tmax, n_pts)
    #         for fracdSL1 in fracdSL_list:
                
    #             Tag1 = f'LP{pCa:.1f}{fracdSL1:+.2f}'
                
    #             dLambda = self.Lambda_ext * fracdSL1

    #             Yss0 = self.Get_ss(pCa)
    #             F0 = self.Ttotal(Yss0)
    #             F0_S = self.Ta_S(Yss0)
    #             F0_W = self.Ta_W(Yss0)
    #             Fa0 = F0_S + F0_W
    #             F0_pas = self.F1(Yss0) + self.F2(Yss0)
    #             Ysol = self.QuickStretchActiveResponse(dLambda, t)  # Ysol = None if integration not successful.                
    #             self.Lambda_ext -= dLambda


    #             """
    #             Ysol is a 1-by-1000-by-8 array containing the ODE solutions for the quick lengthening, as functions of time.
    #             State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE
    #             """
    #             if not Ysol is None:
    #                 ifSuccess = True
    #                 F = self.Ttotal(Ysol)
    #                 F_S = self.Ta_S(Ysol)
    #                 F_W = self.Ta_W(Ysol)
    #                 Fa = F_S + F_W
    #                 F_pas = self.F1(Ysol) + self.F2(Ysol)
    #                 Results = {
    #                     'Fa0': Fa0,
    #                     'F0': F0,
    #                     'F0_S': F0_S,
    #                     'F0_W': F0_W,
    #                     'F0_pas': F0_pas,
    #                     'F': F,
    #                     'F_S': F_S,
    #                     'F_W': F_W,
    #                     'Fa': Fa,
    #                     'F_pas': F_pas}
    #                 self.ExpResults[Tag1] = Results
    #                 self.QSP_t = t
    #             else:
    #                 ifSuccess = False
    #                 self.ExpResults[Tag1] = None
                    

    #             if ifPlotNormalised:
    #                 ax_norm.plot(t, (Fa/Fa0-1)/np.abs(fracdSL1))

    #             if ifPlot:
    #                 # Plot forces
    #                 normF = 1  # F0[0]
    #                 ax_forces[0].plot(np.append([-t[-1]/20, 0], t),
    #                                   np.append([F0, F0], F)/normF)
    #                 ax_forces[0].set_ylabel('F_total')
    #                 ax_forces[1].plot(np.append([-t[-1]/20, 0], t),
    #                                   np.append([F0_S+F0_W, F0_S+F0_W], F_S+F_W)/normF)
    #                 ax_forces[1].set_ylabel('F_active')
    #                 ax_forces[2].plot(np.append([-t[-1]/20, 0], t),
    #                                   np.append([F0_S, F0_S], F_S)/normF)
    #                 ax_forces[2].set_ylabel('F_S')
    #                 ax_forces[3].plot(np.append([-t[-1]/20, 0], t),
    #                                   np.append([F0_W, F0_W], F_W)/normF)
    #                 ax_forces[3].set_ylabel('F_W')
    #                 ax_forces[4].plot(np.append([-t[-1]/20, 0], t),
    #                                   np.append([F0_pas, F0_pas], F_pas)/normF)
    #                 ax_forces[4].set_ylabel('F_passive')
    #                 # fig_forces.suptitle(f'Lengthening({ActiveOrPassive}) dLambda={dLambda} : Forces')

    #             if ifPlotStates:
    #                 # Plot states
    #                 # No need to plot CaTRPN (i.e. Ysol[0][:,0]) since no dynamics.
    #                 ax_states[0].plot(t, Ysol[:, 0])
    #                 ax_states[0].set_ylabel('CaTRPN')
    #                 ax_states[1].plot(t, Ysol[:, 1])
    #                 ax_states[1].set_ylabel('B')
    #                 ax_states[2].plot(t, Ysol[:, 2])
    #                 ax_states[2].set_ylabel('S')
    #                 ax_states[3].plot(t, Ysol[:, 3])
    #                 ax_states[3].set_ylabel('W')
    #                 ax_states[4].plot(t, self.U(Ysol))
    #                 ax_states[4].set_ylabel('U')
    #                 ax_states[5].plot(t, Ysol[:, 8]+Ysol[:, 9])
    #                 ax_states[5].set_ylabel('BE+UE')
    #                 ax_states[6].plot(t, Ysol[:, 4])
    #                 ax_states[6].set_ylabel('Zs')
    #                 ax_states[7].plot(t, Ysol[:, 5])
    #                 ax_states[7].set_ylabel('Zw')
    #                 ax_states[8].plot(t, Ysol[:, 7])
    #                 ax_states[8].set_ylabel('Cd')
    #                 # fig_states.suptitle(f'Lengthening({ActiveOrPassive}) dLambda={dLambda} : States')
    #         # print(f'Done pCa = {pCa}')

    #     self.pCai = pCai_old
    #     return ifSuccess
        # if ActiveOrPassive == 'active':
        #     self.pCai = pCai_old           # reset to the old Cai

        # print(f'   -done {ActiveOrPassive[0]}LP{jPcl}{PoM}:  {LenPcl[ActiveOrPassive][jPcl]}')

    # def DoLengthening_OLD(self, jPcl=0, PoM = '+', ActiveOrPassive='active', ifPlot = False, ifPlotStates = False):
        # assert ActiveOrPassive in ['active', 'passive']
        # assert PoM in ['+', '-']

        # if PoM == '+':
        #     dLambda = LenPcl[ActiveOrPassive][jPcl]['dLambda']
        # elif PoM == '-':
        #     dLambda = - LenPcl[ActiveOrPassive][jPcl]['dLambda']

        # if ActiveOrPassive=='active':
        #     pCa = LenPcl['active'][jPcl]['pCa']
        #     pCai_old = self.pCai  # Remember to reset later!!
        #     self.pCai = lambda t: pCa
        #     Yss0 = self.Get_ss()
        #     F0 = self.Ttotal(Yss0)
        #     F0_S = self.Ta_S(Yss0)
        #     F0_W = self.Ta_W(Yss0)
        #     Fa0 = F0_S + F0_W
        #     F0_pas = self.F1(Yss0) + self.F2(Yss0)
        #     Ysol = self.QuickStretchActiveResponse(dLambda, t)

        # elif ActiveOrPassive == 'passive':
        #     Yss = self.Get_ss()
        #     F0_S = 0
        #     F0_W = 0
        #     F0_pas = self.F1(Yss) + self.F2(Yss)
        #     F0 = F0_pas
        #     Ysol = self.QuickStretchActiveResponse(dLambda, t)

        # Yss9 = self.Get_ss()
        # Fa9 = self.Ta_S(Yss9) + self.Ta_W(Yss9)
        # self.Lambda_ext -= dLambda

        # """
        # Ysol is a 1-by-1000-by-8 array containing the ODE solutions for the quick lengthening, as functions of time.
        # State variables are :  CaTRPN, B, S, W , Zs, Zw, Lambda, Cd, BE, UE
        # """
        # F = self.Ttotal(Ysol)
        # F_S = self.Ta_S(Ysol)
        # F_W = self.Ta_W(Ysol)
        # Fa = F_S + F_W
        # F_pas = self.F1(Ysol) + self.F2(Ysol)

        # if ActiveOrPassive == 'active':
        #     self.ExpResults[f'aLP{jPcl}{PoM}'] = {#'t': t,
        #                                        'Fa0': Fa0,
        #                                        'Fa9': Fa9,
        #                                        'F0': F0,
        #                                        'F0_S': F0_S,
        #                                        'F0_W': F0_W,
        #                                        'F0_pas': F0_pas,
        #                                        'F': F,
        #                                        'F_S': F_S,
        #                                        'F_W': F_W,
        #                                        'Fa': Fa,
        #                                        'F_pas': F_pas,
        #                                        'pCa': pCa,
        #                                        'dLambda': dLambda,
        #                                        'PoM': PoM}
        # elif ActiveOrPassive == 'passive':
        #     self.ExpResults[f'pLP{jPcl}{PoM}'] = {#'t': t,
        #                                         'F0': F0,
        #                                         'F': F}

        # if ifPlot:
        #     # Plot forces
        #     fig_forces, ax_forces = plt.subplots(nrows=4)
        #     normF = 1 #F0[0]
        #     ax_forces[0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0, F0], F)/normF); ax_forces[0].set_ylabel('F_total');
        #     ax_forces[1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S, F0_S], F_S)/normF); ax_forces[1].set_ylabel('F_S')
        #     ax_forces[2].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W, F0_W], F_W)/normF); ax_forces[2].set_ylabel('F_W')
        #     ax_forces[3].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas, F0_pas], F_pas)/normF); ax_forces[3].set_ylabel('F_passive')
        #     fig_forces.suptitle(f'Lengthening({ActiveOrPassive}) dLambda={dLambda} : Forces')

        # if ifPlotStates:
        #     # Plot states
        #     fig_states, ax_states = plt.subplots(nrows=9, figsize=(7,10))
        #     # No need to plot CaTRPN (i.e. Ysol[0][:,0]) since no dynamics.
        #     ax_states[0].plot(t, Ysol[:,0]); ax_states[0].set_ylabel('CaTRPN')
        #     ax_states[1].plot(t, Ysol[:,1]); ax_states[1].set_ylabel('B')
        #     ax_states[2].plot(t, Ysol[:,2]); ax_states[2].set_ylabel('S')
        #     ax_states[3].plot(t, Ysol[:,3]); ax_states[3].set_ylabel('W')
        #     ax_states[4].plot(t, self.U(Ysol)); ax_states[4].set_ylabel('U')
        #     ax_states[5].plot(t, Ysol[:,8]+Ysol[:,9]); ax_states[5].set_ylabel('BE+UE')
        #     ax_states[6].plot(t, Ysol[:,4]); ax_states[6].set_ylabel('Zs')
        #     ax_states[7].plot(t, Ysol[:,5]); ax_states[7].set_ylabel('Zw')
        #     ax_states[8].plot(t, Ysol[:,7]); ax_states[8].set_ylabel('Cd')
        #     fig_states.suptitle(f'Lengthening({ActiveOrPassive}) dLambda={dLambda} : States')

        # if ActiveOrPassive == 'active':
        #     self.pCai = pCai_old           # reset to the old Cai

        # print(f'   -done {ActiveOrPassive[0]}LP{jPcl}{PoM}:  {LenPcl[ActiveOrPassive][jPcl]}')

    def DoFpCa(self, ifPlot=False, ifSave=False, DLambda=0.0, pCai_limits=[None]*2):
        pCai_original = self.pCai
        self.Lambda_ext += DLambda     # Remember to reset below!
        if pCai_limits[0] is None:
            pCai_limits[0] = 4
        if pCai_limits[1] is None:
            pCai_limits[1] = 7
        pCai_array = np.linspace(pCai_limits[1], pCai_limits[0], 50)
        F_array = [None]*len(pCai_array)

        for ipCai, pCai1 in enumerate(pCai_array):
            self.pCai = lambda t: pCai1
            F_array[ipCai] = self.Ta(self.Get_ss())
        self.pCai = pCai_original   # Reset Cai

        # if np.diff(F_array)[-1] > max(np.diff(F_array))/10:
        #     raise Exception('FpCa saturation too incomplete!')
        # if F_array[0] > F_array[-1]/3:
        #     raise Exception('FpCa does not decay enough!')

        self.ExpResults[f'FpCa_{self.Lambda_ext:.2f}'] = {'pCai': pCai_array,
                                                          'F': F_array,
                                                          'params': {'DLambda': DLambda}}

        if ifPlot:
            fig_FpCa, ax_FpCa = plt.subplots(
                num=f'F-pCa, SL={self.SL0*self.Lambda_ext}')
            ax_FpCa.semilogx(10**-pCai_array, F_array)
            ax_FpCa.set_xlabel('Cai (M)')
            ax_FpCa.set_ylabel('F')
            ax_FpCa.set_title(f'F-pCa, SL={self.SL0*self.Lambda_ext}')

        print(f'   -done FpCa (Lambda={self.Lambda_ext:.2f})')
        self.Lambda_ext -= DLambda

        # ax_Fmax = figFpCa.add_subplot(2,3,4)
        # ax_nH = figFpCa.add_subplot(2,3,5)
        # ax_EC50 = figFpCa.add_subplot(2,3,6)
        # ax_FpCa.set_title(f'F-pCa, SL0={self.SL0}')

        # if ifPlot:
        #     bins = 10
        #     ax_Fmax.hist(Fmax_a, bins=bins, range=(0, max(Fmax_a)*1.1));
        #     ax_Fmax.set_xlabel('Fmax')
        #     ax_nH.hist(nH_a, bins=bins, range=(0, max(nH_a)*1.1)); ax_nH.set_xlabel('nH')
        #     ax_EC50.hist(EC50_a, bins=bins, range=(0, max(EC50_a)*1.1)); ax_EC50.set_xlabel('EC50')
        #     figFpCa.tight_layout()

        # Features_FpCa = {'Fmax':Fmax_a, 'nH':nH_a, 'EC50':EC50_a}
        # if ifSave:
        #     import pickle
        #     with open(f'{home}/Dropbox/Python/BHFsim/Features_FpCa.dat', 'wb') as file_features:
        #         pickle.dump([PSet, Features_FpCa], file_features)

        # return Features_FpCa

    def DoIsomTwitch(self, t=None, ifPlot=False):
        pCai_original = self.pCai
        self.pCai = lambda t: -np.log10(Coppini_Cai(t))
        self.dLambdadt_fun = self.FunZero
        Y0 = self.Get_ss()

        if t is None:
            t = np.linspace(0., 1., 100)
        Ysol1 = odeint(self.dYdt, Y0, t)

        F = self.Ta(Ysol1)

        if ifPlot:
            fig, ax = plt.subplots(nrows=2)
            ax[0].plot(t, 10**-self.pCai(t))
            ax[0].set_ylabel('Cai')
            ax[1].plot(t, F)
            ax[1].set_ylabel('F')
            fig.tight_layout()

        return F, t

    def DoDynamic(self, fmin=5, fmax=100, Numf=10, ifPlot=False):

        f_list = np.logspace(np.log10(fmin), np.log10(fmax), Numf)

        Stiffness_f = [None]*len(f_list)
        DphaseTa_f = [None]*len(f_list)

        for ifreq, freq in enumerate(f_list):
            print(f'Doing f{ifreq} = {freq}')

            Tasol, Ysol, t, Stiffness, DphaseTa = self.SinResponse(
                freq, ifPlot=ifPlot)

            Stiffness_f[ifreq] = Stiffness
            DphaseTa_f[ifreq] = DphaseTa

            # if ifPlot:        axsol[0].plot(t[-pointspercycle:]/t[-1], Sin_fun(t[-pointspercycle:], SinFit[0],SinFit[1],SinFit[2]), 'k--')

            # if ifPlot:
            #     axsol[0].plot(t/t[-1], Tasol)
            #     axsol[1].plot(t/t[-1], Ysol[:,6])

        if ifPlot:
            fig, ax = plt.subplots(
                ncols=3, nrows=1, num='Dynamic experiments', figsize=(15, 7))
            Stiffness_f = np.array(Stiffness_f)
            DphaseTa_f = np.array(DphaseTa_f)
            ax[0].semilogx(f_list, Stiffness_f, '-')
            ax[1].semilogx(f_list, DphaseTa_f, '-')
            ax[2].plot(Stiffness_f * np.cos(DphaseTa_f),
                       Stiffness_f * np.sin(DphaseTa_f))
            ax[2].set_aspect('equal', adjustable='box')

    def DoAllExperiments(self, ifPlot=False):

        # self.Doktr(Lambda=1.1, ifPlot=ifPlot)
        # self.Doktr(Lambda=1.2, ifPlot=ifPlot)

        self.DoFpCa(ifPlot=ifPlot, DLambda=0.0)
        # self.DoFpCa(ifPlot=ifPlot, DLambda=0.1)
        # self.DoFpCa(ifPlot=ifPlot, DLambda=-0.2)

        pCa_list = [4.5]
        # self.DoLengthening(pCa_list=pCa_list)

        # for jPcl in range(len(LenPcl['active'])):     # active stretching
        #     self.DoLengthening(jPcl, '+', 'active', ifPlot=ifPlot)
        #     self.DoLengthening(jPcl, '-', 'active', ifPlot=ifPlot)

        # for jPcl in range(len(LenPcl['passive'])):     # passive stretching
        #     self.DoLengthening(jPcl, '+', 'passive', ifPlot=ifPlot)
        #     self.DoLengthening(jPcl, '-', 'passive', ifPlot=ifPlot)

        # self.DoCaiStep(pCai1=7, pCai2=5, ifPlot=ifPlot)

    def QuickStretchActiveResponse(self, dLambda, t, ifResetLambda=True):
        """
        Starting from the steady state, increase Lambda by dLambda. The initial Lambda is
        given by Lambda_ext (specified previously) **before** the stretch is performed.
        Then, this Lambda_ext is **updated**.
        The initial state Y0 that is input into the ODE solver assumes that the step change occurs
        instantaneously, so that only Lambda_0 (and Zs_0 and Zw_0, which are dependent on Lambda_0)
        are altered.
        """
        self.dLambdadt_fun = self.FunZero  # =lambda t: 0
        CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0 = self.Get_ss()
        Zs_0 = Zs_0 + self.As()*dLambda
        Zw_0 = Zw_0 + self.Aw()*dLambda
        self.Lambda_ext = Lambda_0 + dLambda    # <----- update Lambda_ext
        Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0,
              Lambda_0+dLambda, Cd_0, BE_0, UE_0]
        Ysol1, infodict = odeint(self.dYdt, Y0, t, full_output=True)
        # print('MESSAGE: ' + infodict['message'])
        if infodict['message'] != 'Integration successful.':
            return None
        # Ysol1 = solve_ivp(self.dYdt, [t[0], t[-1]], Y0, t_eval=t)
        if ifResetLambda: 
            self.Lambda_ext = Lambda_0   
        return Ysol1

    def FunZero(self, t):
        return 0.

    def QuickStretchPassiveResponse(self, dLambda, t):
        self.dLambdadt_fun = self.FunZero  # = lambda t: 0
        CaTRPN_0 = 0
        B_0 = 1
        S_0 = 0
        W_0 = 0
        Zs_0 = 0
        Zw_0 = 0
        Lambda_0 = self.Lambda_ext
        Cd_0 = Lambda_0 - 1
        Zs_0 = 0
        Zw_0 = 0
        BE_0 = 0
        UE_0 = 0
        self.Lambda_ext = Lambda_0 + dLambda
        Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0,
              Lambda_0+dLambda, Cd_0, BE_0, UE_0]
        Ysol1 = odeint(self.dYdt_pas, Y0, t)
        # Ysol1 = solve_ivp(self.dYdt, [t[0], t[-1]], Y0, t_eval=t)
        self.Lambda_ext = Lambda_0
        return Ysol1

    def SinResponse(self, freq, numcycles=10, pointspercycle=30, dLambda_amplitude=0.0001, ifPlot=False):

        self.dLambdadt_fun = lambda t: \
            dLambda_amplitude * np.cos(2*np.pi*freq*t) * 2*np.pi*freq

        t = np.linspace(0, numcycles/freq, numcycles*pointspercycle)
        Y_ss0 = self.Get_ss()
        Ysol = odeint(self.dYdt, Y_ss0, t)
        Tasol = self.Ta(Ysol)  #self.Ttotal(Ysol)    #

        def Sin_fun(t, *a):
            return a[0]*np.sin(2*np.pi*freq*(t + a[1])) + a[2]
        SinFit, cov = curve_fit(Sin_fun,
                                t[-pointspercycle:], Tasol[-pointspercycle:],
                                p0=(
                                    (max(Tasol[-pointspercycle:]) -
                                     min(Tasol[-pointspercycle:]))/2,
                                    1/freq/4 -
                                    np.argmax(
                                        Tasol[-pointspercycle:])/pointspercycle/freq,
                                    np.mean(Tasol[-pointspercycle:])))

        Stiffness = SinFit[0]/dLambda_amplitude
        DphaseTa = SinFit[1]*2*np.pi*freq
        if DphaseTa < 0:
            DphaseTa += 2*np.pi
        elif DphaseTa > 2*np.pi:
            DphaseTa -= 2*np.pi

        if ifPlot:
            fig_sol, ax_sol = plt.subplots(nrows=2)
            ax_sol[0].plot(t, Tasol, '.-')
            ax_sol[0].plot(t, Sin_fun(t, *SinFit), 'k--')
            ax_sol[0].set_ylabel('Ta')
            ax_sol[1].plot(t, Ysol[:, 6])
            ax_sol[1].set_ylabel('Lambda')
            fig_sol.suptitle(f'f = {freq}')

        return Tasol, Ysol, t, Stiffness, DphaseTa

    # %% Extract features

    def Make_QSP_trace(self, WhichpCa=None, WhichfracdSL=None, n_pts=501, tmax=0.5, ifExclude0=True, ifPlot=False, ifSave=True):
        assert WhichpCa is not None and WhichfracdSL is not None, 'Must specify WhichpCa and WhichfracdSL!'
        if ifPlot:
            fig = plt.figure()
            gs = fig.add_gridspec(nrows=4, ncols=4)
            ax = fig.add_subplot(gs[:-1, 2:])
            # ax_mask = fig.add_subplot(gs[-1:, 2:])
            ax_raw = fig.add_subplot(gs[:,1])
        
        t = np.linspace(0, tmax, n_pts)
        Y = []
        for pCa1 in WhichpCa:
            for fracdSL1 in WhichfracdSL:
                # print(f'   pCa {pCa1}, fracdSL {fracdSL1}')
                Tag1 = f'LP{pCa1:.1f}{fracdSL1:+.2f}'
                # print(f'     {Tag1}')
                assert Tag1 in self.ExpResults.keys() and self.QSP_t[-1] == tmax  and len(self.QSP_t) == n_pts, f'Need to run experiment  {Tag1}'
                Result = self.ExpResults[Tag1]
                if Result is None:
                    y = np.array([None]*n_pts)
                else:
                    y = (Result['Fa']/Result['Fa0'] - 1) / np.abs(fracdSL1)
                Y += list(y)
                if ifPlot:
                    ax_raw.plot(y)
        if ifPlot:
            ax.plot(Y) 
            ax.legend(loc='upper left')
            fig.tight_layout()
            
        if ifSave:
            self.Features['QSP'] = Y
        return Y
        
    def GetFeat_QSPproj(self, QSPpca, ifSave=False):
        Trace = self.Make_QSP_trace(WhichpCa=QSPpca['pCa_list'], WhichfracdSL=QSPpca['fracdSL_list'], n_pts=QSPpca['n_pts'], tmax=QSPpca['tmax'], ifPlot=False, ifSave=ifSave)
        for j in range(len(QSPpca['components'])):
            if not None in Trace:
                self.Features[f'QSP_{j}'] = np.dot(Trace - QSPpca['mean'], QSPpca['components'][j])
            else:
                self.Features[f'QSP_{j}'] = None
        
    def GetFeat_ktr(self, Lambda=1.1, ifPlot=False):
        Tag = f'ktr_{Lambda:.2f}'
        assert Tag in self.ExpResults.keys(), '*** ktr experiments not yet performed?'
        from scipy.optimize import curve_fit
        self.Lambda_ext = Lambda
        
        
        F = self.ExpResults[Tag]['F']# ; F -= F[0]
        t = self.ExpResults[Tag]['t']
        try:
            Fss = self.Ta_ss() + self.Tp_ss();
            def FitFn(x, c): return (Fss-F[0])*(1-np.exp(-c*x)) + F[0]
            p0 = [next(1/t1 for jt1, t1 in enumerate(t) if (F[-1]-F[jt1])<(F[-1]-F[0])/np.e   )]#, F[0]]
            # print(p0)
            Fitval, cov = curve_fit(FitFn, t, F,
                                    p0=p0)
    
            # Return the decay constant
            self.Features[f'ktr_{Lambda:.2f}'] = Fitval[0]
            if ifPlot:
                fig, ax =plt.subplots()
                ax.plot(t, F)
                ax.plot(t, FitFn(t, *Fitval), 'k--')
        except:
            self.Features[f'ktr_{Lambda:.2f}'] = None
        

    def GetFeat_FpCa(self, Lambda=1.1):
        assert f'FpCa_{Lambda:.2f}' in self.ExpResults.keys(
        ), '*** FpCa experiments not yet performed?'
        from scipy.optimize import curve_fit

        F_array = self.ExpResults[f'FpCa_{Lambda:.2f}']['F']
        pCai_array = self.ExpResults[f'FpCa_{Lambda:.2f}']['pCai']
        jNotNan = [j for j, y in enumerate(F_array) if y == y]
        HillParams, cov = curve_fit(HillFn,
                                    [10**-pCai_array[j] for j in jNotNan],
                                    [F_array[j] for j in jNotNan],
                                    p0=[F_array[-1], 1,
                                        next(10**-pca for ipca, pca in enumerate(pCai_array) if F_array[ipca] > F_array[-1]/2)])  # 10**-7])

        self.Features[f'Fmax_{Lambda:.2f}'] = HillParams[0]
        self.Features[f'nH_{Lambda:.2f}'] = HillParams[1]
        self.Features[f'pEC50_{Lambda:.2f}'] = -np.log10(HillParams[2])

    def GetFeat_FpCa_cf(self, Lambda=1.1, ifPrint=False):
        from scipy.optimize import minimize
        
        Lambda_initial = self.Lambda_ext    # to reset later
        self.Lambda_ext = Lambda
        
        # Calculate Ta_ss at any pCai guaranteed to be at saturation level.
        # Fmax = self.Ta_ss(pCai=4)
        # pCa50fit = minimize(lambda x: (self.Ta_ss(x) - Fmax/2)**2, [5]).x[0]
        Fmax_active = self.Ta_ss(pCai=3)
        pCa50fit = minimize(lambda x: (self.Ta_ss(x) -Fmax_active/2)**2, [5.], method='Nelder-Mead').x[0]

        dx = 0.05
        def dFdx(x): return (self.Ta_ss(x+dx)-self.Ta_ss(x)) / \
            Fmax_active / dx * -np.log10(np.e)
        nH = 4 * dFdx(pCa50fit)

        if ifPrint:
            print(f'Fmax={Fmax_active}')
            print(f'pCa50fit={pCa50fit}')
            print(f'nH={nH}')

        self.Lambda_ext = Lambda_initial   # reset initial value
        return Fmax_active, pCa50fit, nH

    def GetFeat_DeltaFpCa(self, Lambda1=1.1, Lambda2=1.2):
        assert f'Fmax_{Lambda2:.2f}' in self.Features.keys(
        ) and f'Fmax_{Lambda1:.2f}' in self.Features.keys(), '*** FpCa experiments not yet performed?'

        DeltaFmax = self.Features[f'Fmax_{Lambda2:.2f}'] / \
            self.Features[f'Fmax_{Lambda1:.2f}']
        DeltanH = self.Features[f'nH_{Lambda2:.2f}'] - \
            self.Features[f'nH_{Lambda1:.2f}']
        DeltapEC50 = self.Features[f'pEC50_{Lambda2:.2f}'] - \
            self.Features[f'pEC50_{Lambda1:.2f}']

        self.Features[f'DeltaFmax_{Lambda1:.2f}_{Lambda2:.2f}'] = DeltaFmax
        self.Features[f'DeltanH_{Lambda1:.2f}_{Lambda2:.2f}'] = DeltanH
        self.Features[f'DeltapEC50_{Lambda1:.2f}_{Lambda2:.2f}'] = DeltapEC50

    def GetFeat_Lengthening(self, LenBasisPCA, ifPlot=False, n_pcaTruncate=None):
        assert not LenBasisPCA is None, 'Provide LenBasisPCA !'
        # Need this to match data time points with PCA basis.
        from scipy.interpolate import interp1d
        import re

        if n_pcaTruncate is None:
            n_pcaTruncate = len(LenBasisPCA.components_)
        Y = []
        pCa_list = []
        for Tag1 in list(self.ExpResults.keys()):
            # print(Tag1)
            if not 'LP' in Tag1:
                continue
            re_result = re.search(r'LP([^+-]*)(.*)', Tag1).groups()
            pCa1 = float(re_result[0])
            if not pCa1 in pCa_list:
                pCa_list += [pCa1]
            fracdSL1 = float(re_result[1])
            # Take experiment data and normalise them as per PCA
            Y += [[pCa1, fracdSL1, self.ExpResults[Tag1]['F0']] +
                  list(self.ExpResults[Tag1]['F'] / self.ExpResults[Tag1]['F0'] - 1)]

        DF = pd.DataFrame(data=Y, columns=[
                          'pCa', 'fracdSL', 'F0'] + list(t)).set_index(['pCa', 'fracdSL'])

        # Interpolate experiment data to match PCA basis points.
        Y_interp = interp1d(list(DF.columns[1:]), DF.to_numpy()[:, 1:], kind='cubic')
        Y_projection = (Y_interp(list(LenBasisPCA.DF.columns)))  @ LenBasisPCA.components_.T
        # + self.MetaData.loc[ii1]['Fref'].mean()    # + LenBasisPCA.mean_
        Y_approx = (Y_projection[:, :n_pcaTruncate] @ LenBasisPCA.components_[
                    :n_pcaTruncate, :] + 1.).T * DF.to_numpy()[:, 0]
        DFpca = pd.DataFrame(data=Y_projection, index=DF.index)

        def CoeffFitFn(x, a1, a2): return a1*x + a2*x**2
        Coeffs = []
        for jcomp in range(n_pcaTruncate):
            for jpCa, pCa1 in enumerate(pCa_list):
                xx = fracdSL_list
                yy = [DFpca.loc[pCa1, fracdSL1][jcomp]
                      for fracdSL1 in fracdSL_list]
                Fit = scipy.optimize.curve_fit(CoeffFitFn, xx, yy, p0=(0., 0.))
                Coeffs += [[jcomp, pCa1] + list(Fit[0])]
        Coeffs = pd.DataFrame(data=Coeffs, columns=[
                              'comp', 'pCa', 'a1', 'a2']).set_index(['comp', 'pCa'])

        if ifPlot:
            fig_data, ax_data = plt.subplots()
            ax_data.plot(
                t, (DF.to_numpy()[:, 1:].T + 1.) * DF.to_numpy()[:, 0])
            ax_data.set_prop_cycle(None)
            t_PCA = list(LenBasisPCA.DF.columns)
            ax_data.plot(t_PCA, Y_approx, ':')

            fig_proj, ax_proj = plt.subplots(
                nrows=n_pcaTruncate, ncols=len(pCa_list), figsize=(20, 10))
            for jcomp in range(n_pcaTruncate):
                for jpCa, pCa1 in enumerate(pCa_list):
                    y = [DFpca.loc[pCa1, fracdSL1].to_numpy()[jcomp]
                         for fracdSL1 in fracdSL_list]
                    ax_proj[jcomp, jpCa].plot(fracdSL_list, y, 'o')
                    ax_proj[jcomp, jpCa].plot(fracdSL_list, CoeffFitFn(np.array(fracdSL_list),
                                                                       Coeffs.loc[jcomp,
                                                                                  pCa1]['a1'],
                                                                       Coeffs.loc[jcomp, pCa1]['a2']), 'k--')
                    if jcomp == 0:
                        ax_proj[jcomp, jpCa].set_title(f'pCa {pCa1}')
                    if jpCa == 0:
                        ax_proj[jcomp, jpCa].set_ylabel(
                            f'pca component {jcomp}')
                AL.MatchAxes(ax_proj[jcomp])
            fig_proj.tight_layout()

            fig_coeffs, ax_coeffs = plt.subplots(
                nrows=n_pcaTruncate, ncols=2, figsize=(6, 12))
            for jcomp in range(n_pcaTruncate):
                ax_coeffs[jcomp, 0].plot(
                    pCa_list, [Coeffs.loc[jcomp, pCa1]['a1'] for pCa1 in pCa_list], '.-')
                ax_coeffs[jcomp, 1].plot(
                    pCa_list, [Coeffs.loc[jcomp, pCa1]['a2'] for pCa1 in pCa_list], '.-')
                ax_coeffs[jcomp, 0].invert_xaxis()
                ax_coeffs[jcomp, 1].invert_xaxis()
                ax_coeffs[jcomp, 0].set_ylabel(f'comp {jcomp}')
            ax_coeffs[0, 0].set_title('a1')
            ax_coeffs[0, 1].set_title('a2')
            ax_coeffs[-1, 0].set_xlabel('pCa')
            ax_coeffs[-1, 1].set_xlabel('pCa')
            fig_coeffs.suptitle('Land2017')
            fig_coeffs.tight_layout()

        for idx in DFpca.index:
            pCa1 = idx[0]
            fracdSL1 = idx[1]
            for jcomp in range(n_pcaTruncate):
                self.Features[f'LP{pCa1:.1f}{fracdSL1:+.2f}_{jcomp}'] = DFpca.loc[pCa1, fracdSL1][jcomp]
        return DFpca

    def GetFeat_phenVec(self, LenBasisPCA, phvec_key, EMPCA, n_pcaTruncate=None, ifNormalise=True, ifPlot=False):

        if n_pcaTruncate is None:
            n_pcaTruncate = len(LenBasisPCA.components_)

        phvec = []
        mask = []
        for key1 in phvec_key:
            if f'LP{key1["pCa"]:.1f}{key1["fracdSL"]:+.2f}_{key1["jcomp"]}' in self.Features.keys():  
                mask += [1]            
                phvec += [self.Features[f'LP{key1["pCa"]:.1f}{key1["fracdSL"]:+.2f}_{key1["jcomp"]}']]
            else:
                mask += [0]
                phvec += [0]
        
        # Project phenVec on EMPCA basis.
        self.Features['phenVec'] = [phvec @ EMPCA.eigvec[jcomp] for jcomp in range(EMPCA.nvec)]
        
        if ifPlot:
            fig, ax = plt.subplots()
            ax.plot(phvec, 'k.-', label='original')
            for jcomp in range(EMPCA.nvec):
                ax.plot(self.Features['phenVec'][jcomp] * EMPCA.eigvec[jcomp], label=f'{jcomp}')
            ax.plot( np.array(self.Features['phenVec']) @ EMPCA.eigvec, 'r:'  , label='reconstructed' ) 
            ax.legend(loc='upper right')
            
                    
        return self.Features['phenVec']

        # DFpca = self.GetFeat_Lengthening(LenBasisPCA, ifPlot=False, n_pcaTruncate=n_pcaTruncate)

        # # Generate Proj1 in the same format as in KY:LenDataDir_class.Get_phenVec
        # Proj1 = []
        # for pCa1 in pCa_list:
        #     projections = DFpca.loc[pCa1]
        #     NewRow = pd.DataFrame([[0.00, 0., 0.,0.,0.,0.,0., 0.]], columns=['fracdSL']+list(range(7))).set_index(['fracdSL'])
        #     projections %%%%%%%%%%%%%%
        #     Proj1 += [[pCa1] + [.to_numpy().tolist()] ]
        # Proj1 = pd.DataFrame(data=Proj1, columns=['pCa', 'projections'])
        # Proj1 = Proj1.set_index('pCa')

        # phvec = []
        # for jcomp in range(n_pcaTruncate):
        #     for jpCa, pCa1 in enumerate(pCa_list):
        #         for jfracdSL, fracdSL1 in enumerate(fracdSL_list):
        #             if ifNormalise:
        #                 facnorm = np.sqrt(LenBasisPCA.explained_variance_[jcomp])
        #             else:
        #                 facnorm = 1.
        #             if pCa1 in [idx for idx in Proj1.index]:
        #                 phvec += [Proj1.loc[pCa1].to_numpy().mean()[jfracdSL,jcomp] / facnorm]
        #             else:
        #                 phvec += [None]

        # if ifPlot:
        #     fig, ax_vec = plt.subplots()
        #     ax_vec.plot(phvec, '.-')
        #     fig.suptitle(f'Land2017 prediction')

        # return phvec

    # def GetFeat_Lengthening(self, jPcl, PoM):
    #     Tag = f'aLP{jPcl}{PoM}'
    #     self.Features[Tag + '_y0'] = self.ExpResults[Tag]['Fa0']
    #     y_ = self.ExpResults[Tag]['Fa'] / self.ExpResults[Tag]['Fa0']  - 1
    #     for n in range(len(LenBasisPCA.components_)):
    #         self.Features[Tag + f'_n{n}'] = np.dot(y_ , LenBasisPCA.components_[n])

    def GetFeat_CaiStep(self, pCai1=4, pCai2=4):
        Tag = f'CaiStep_{pCai1:.2f}_{pCai2:.2f}'
        F = self.ExpResults[Tag]['F']
        t = self.ExpResults[Tag]['t']
        self.Features[f'dFCaiStep_{pCai1:.2f}_{pCai2:.2f}'] = F[-1]-F[0]
        self.Features[f'tCaiStep_{pCai1:.2f}_{pCai2:.2f}'] = next(
            t[i1] for i1, F1 in enumerate(F) if (F1-F[0])/(F[-1]-F[0]) > 0.5)

    
    # %% Display

    def Plot_ktr(self, Lambda=1.0, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.ExpResults[f'ktr_{Lambda:.2f}']['t'], self.ExpResults[f'ktr_{Lambda:.2f}']['t'])
        
    def Plot_FpCa(self, WhichPlot='ExpResults', Lambda=1.1, ax=None, Emulators=None, TargetParams=None):
        assert WhichPlot in ['ExpResults', 'Features', 'emulated']

        if type(ax) == type(None):
            fig, ax = plt.subplots()

        pCai = self.ExpResults[f'FpCa_{Lambda:.2f}']['pCai']
        if WhichPlot == 'ExpResults':
            ax.semilogx(10**-pCai, self.ExpResults[f'FpCa_{Lambda:.2f}']['F'])
        elif WhichPlot == 'Features':
            Features_FpCa = self.Features
            ax.semilogx(10**-pCai, HillFn(10**-pCai, Features_FpCa[f'Fmax_{Lambda:.2f}'], Features_FpCa[f'nH_{Lambda:.2f}'], 10**-Features_FpCa[f'pEC50_{Lambda:.2f}']),
                        'k--')
        elif WhichPlot == 'emulated':
            Fmax_emul, err = Emulators[f'Fmax_{Lambda:.2f}'].predict(
                [self.PSet[TargetParams].to_numpy()])
            nH_emul, err = Emulators[f'nH_{Lambda:.2f}'].predict(
                [self.PSet[TargetParams].to_numpy()])
            pEC50_emul, err = Emulators[f'pEC50_{Lambda:.2f}'].predict(
                [self.PSet[TargetParams].to_numpy()])
            Features_FpCa = {f'Fmax_{Lambda:.2f}': Fmax_emul,
                             f'nH_{Lambda:.2f}': nH_emul, f'pEC50_{Lambda:.2f}': pEC50_emul}
            ax.semilogx(10**-pCai, HillFn(10**-pCai, Features_FpCa[f'Fmax_{Lambda:.2f}'], Features_FpCa[f'nH_{Lambda:.2f}'], 10**-Features_FpCa[f'pEC50_{Lambda:.2f}']),
                        'r:')

    # def LengtheningFromFeatures(self, aLPx_y0=1., aLPx_n=[]):
    #     y = LenBasisPCA.mean_.copy()
    #     for ja, a1 in enumerate(aLPx_n):
    #         y += a1 * LenBasisPCA.components_[ja]
    #     return (y + 1) * aLPx_y0

    def Plot_Lengthening(self, WhichPlot='ExpResults', jPcl=0, PoM='+', Emulators=None, TargetParams=AllParams, ax=None):
        assert WhichPlot in ['ExpResults', 'Features', 'emulated']
        if type(ax) == type(None):
            fig, ax = plt.subplots()
        Tag = f'aLP{jPcl}{PoM}'

        if WhichPlot == 'ExpResults':
            F = self.ExpResults[Tag]['Fa']
            F0 = self.ExpResults[Tag]['Fa0']
            # ax.plot(np.append( [-t[-1]/20, 0], t), np.append([F0, F0], F))
            ax.plot(np.append([-t[-1]/20, 0], t), np.append([F0, F0], F))

        if WhichPlot == 'Features':
            Features_Lengthening = self.Features
            F0 = Features_Lengthening[Tag+'_y0']
            aLPx_n = [Features_Lengthening[Tag+f'_n{n}']
                      for n in range(len(LenBasisPCA.components_))]
            F = self.LengtheningFromFeatures(F0, aLPx_n)
            ax.plot(np.append([-t[-1]/20, 0], t),
                    np.append([F0, F0], F), 'k--')

        if WhichPlot == 'emulated':
            Features_Lengthening = {}
            F0, err = Emulators[Tag +
                                '_y0'].predict([self.PSet[TargetParams].to_numpy()])
            Features_Lengthening[Tag+'_y0'] = F0
            aLPx_n = []
            for n in range(len(LenBasisPCA.components_)):
                aLPx_n1, err = Emulators[Tag+f'_n{n}'].predict(
                    [self.PSet[TargetParams].to_numpy()])
                aLPx_n += [aLPx_n1]
            F = self.LengtheningFromFeatures(F0, aLPx_n)
            ax.plot(np.append([-t[-1]/20, 0], t), np.append([F0, F0], F), 'r:')

    def ShowExp_FpCa(self, Lambda=1.10, Emulators={}, TargetParams=[]):
        fig, ax = plt.subplots()
        self.Plot_FpCa('ExpResults', Lambda, ax)
        self.Plot_FpCa('Features', Lambda, ax)
        self.Plot_FpCa('emulated', Lambda, ax, Emulators, TargetParams)
        
    
    def ShowExperiments(self, Emulators={}, TargetParams=[], PCAcomponents=[], PCAmean=[]):
        pass
        # # %% Show FpCa
        # WhichLambda = [float(temp[5:])
        #                for temp in self.ExpResults.keys() if 'FpCa' in temp]
        # if len(WhichLambda) > 0:
        #     fig_FpCa = plt.figure(figsize=(3*len(WhichLambda), 3))
        #     ax_FpCa = {Lambda1: fig_FpCa.add_subplot(
        #         1, len(WhichLambda), j+1) for j, Lambda1 in enumerate(WhichLambda)}

        #     for Lambda1 in WhichLambda:
        #         self.Plot_FpCa('ExpResults', Lambda1, ax_FpCa[Lambda1])
        #         self.Plot_FpCa('Features', Lambda1, ax_FpCa[Lambda1])
        #         self.Plot_FpCa('emulated', Lambda1,
        #                        ax_FpCa[Lambda1], Emulators, TargetParams)

        # # %% Show lengthening
        # WhichpCa = list(set([Pcl1['pCa'] for Pcl1 in LenPcl['active']]))
        # WhichpCa.sort(reverse=True)
        # if len(WhichpCa) > 0:
        #     fig_len = plt.figure(figsize=(3*len(WhichpCa), 3))
        #     ax_len = {}
        #     for jpCa, pCa1 in enumerate(WhichpCa):
        #         ax_len[pCa1] = fig_len.add_subplot(1, len(WhichpCa), jpCa+1)
        #         ax_len[pCa1].set_title(f'pCa {pCa1}')

        #     for jPcl, Pcl1 in enumerate(LenPcl['active']):
        #         for PoM in ['+', '-']:
        #             self.Plot_Lengthening(
        #                 'ExpResults', jPcl, PoM, TargetParams, ax=ax_len[Pcl1['pCa']])
        #             self.Plot_Lengthening(
        #                 'Features',   jPcl, PoM, TargetParams, ax=ax_len[Pcl1['pCa']])
        #             self.Plot_Lengthening(
        #                 'emulated',   jPcl, PoM, Emulators, TargetParams, ax=ax_len[Pcl1['pCa']])

        #             # Tag = f'aLP{jPcl}{PoM}'
        #             # LPcomp_n_pred = [ Emulators[f'{Tag}_n{n1}'].predict([self.PSet[TargetParams].to_numpy()])[0]  for n1 in range(nLenComponents)]
        #             # LPcomp_y0_pred = [ Emulators[f'{Tag}_y0'].predict([self.PSet[TargetParams].to_numpy()])[0]  for n1 in range(nLenComponents)]
        #             # yemul = [self.Features[Tag+f'_n{n1}'] *  PCAcomponents[f'active{PoM}'][n1]
        #             #          for n1 in range(nLenComponents)]
        #             # y0 = self.ExpResults[Tag]['F0']
        #             # yemul = (np.sum(np.array(yemul), axis=0) +  PCAmean[f'active{PoM}']) * y0 + y0

        #             # ax_len[Pcl1['pCa']].plot(Exp1['t'], yemul,'k:')


# %% Parameter sampling

def Sample(xmin=0, xmax=1, numsamples=1):
    """
    Homogeneous sampling between xmin and xmax.
    """
    return xmin + np.random.rand(numsamples) * (xmax-xmin)


def PlotS1(gsa, Feature):
    plt.style.use("seaborn")
    figS1, axS1 = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(ax=axS1[0], data=gsa.S1)
    sns.boxplot(ax=axS1[1], data=gsa.ST)
    figS1.suptitle(Feature)
    axS1[0].set_xticklabels(gsa.ylabels, rotation=45)
    axS1[0].set_title('S1')
    axS1[1].set_xticklabels(gsa.ylabels, rotation=45)
    axS1[0].set_title('Stotal')


def ShowDistances(Model0, Cohort, iModels):
    fig, ax = plt.subplots()
    dPav = []
    if len(iModels) == 0:
        iModels = list(range(len(Cohort.Models)))
    for i1 in iModels:
        dP = np.array(Cohort.Models[i1].X) - np.array(Model0.X)
        dPav += [np.sqrt(np.sum(dP**2) / len(Model0.X))]

        ax.plot(dP.T, '.-')

    dPav = iter(dPav)
    for i1 in iModels:
        print(f'Model {i1:3} :  distance = {next(dPav):.3f}')


# %% Cohort  (obsolete)

# class Land2017Cohort:
#     ParFac = 2

#     # %%% Definition
#     def __init__(self, n=0, Model_ref=Land2017, WhichDep='force', IncludeParams='AllParams', ExcludeParams=[], ifFilter=False, WhichEngine='LH',
#                  PSet=None, SetParameters={}):
        
#         if IncludeParams=='AllParams':
#             self.IncludeParams = list(AllParams)
#         else:
#             self.IncludeParams = IncludeParams
#         for param1 in ExcludeParams:
#             if param1 in self.IncludeParams:
#                 self.IncludeParams.remove(param1)
        
#         if n > 0:
#             self.n = n
#             self.ParRange = {}
#             self.ParBounds = {}
#             for param1 in AllParams:
#                 if param1 in SetParameters.keys():
#                     self.ParRange[param1] = tuple([SetParameters[param1]]*2)
#                     assert param1 not in IncludeParams, f'Set parameter {param1} cannot be included in cohort randomisation!'
#                     if param1 not in ExcludeParams:
#                         ExcludeParams += [param1]
#                     continue
#                 if param1 == 'pCa50ref':
#                     self.ParRange['pCa50ref'] = (0.8, 1.1)
#                 else:
#                     self.ParRange[param1] = (0.1, self.ParFac)
#                 if param1 == 'koffon':
#                     RefValue = Land2017.koffon_ref[WhichDep]
#                 else:
#                     RefValue = getattr(Land2017, param1)
#                 self.ParBounds[param1] = (self.ParRange[param1][0]*RefValue,
#                                           self.ParRange[param1][1]*RefValue)
#                 # Ensure the lower bound is less than the upper bound.
#                 if self.ParBounds[param1][0] > self.ParBounds[param1][1]:
#                     self.ParBounds[param1] = (
#                         self.ParBounds[param1][1], self.ParBounds[param1][0])

#             print(f'Generating PSet (n={n})')
#             self.PSet = self.MakeParamSet(
#                 numsamples=n, IncludeParams=IncludeParams, ExcludeParams=ExcludeParams, 
#                 SetParameters=SetParameters, WhichEngine=WhichEngine)

#             # Form the cohort based on the Latin Hypercube sampled parameters. Models[0] is the 'reference'.

#             if ifFilter == False:
#                 self.Models = [Land2017(self.PSet.loc[jModel])
#                                for jModel in self.PSet.index]
#             else:
#                 self.Models = [Land2017(self.PSet.loc[0])]
#                 self.Models[0].DoAllExperiments()
#                 # i1, PSet1 in enumerate(self.PSet[1:]):
#                 for jModel in range(1, len(self.PSet.index)):
#                     print(f'--------------Doing model {jModel} -------------')
#                     Model1 = Land2017(self.PSet.loc[jModel])
#                     Model1.TargetParams = self.TargetParams
#                     Model1.DoAllExperiments()
#                     if Model1.Filter(self.Models[0]):
#                         self.Models += [Model1]
#                         print(f'Added model {jModel} to cohort.')

#         else:  # i.e., if n not >0
#             n_models = len(PSet.index)
#             self.PSet = pd.DataFrame(
#                 [{par1: 1. if par1 not in SetParameters.keys() else SetParameters[par1]
#                   for par1 in AllParams} for j in range(n_models)])

#             for par1 in PSet.columns:
#                 self.PSet[par1] = PSet[par1]
#                 # self.PSet = pd.DataFrame([{par1: X1[jpar] for jpar, par1 in enumerate(AllParams)} for X1 in X])
#             self.Models = [Land2017(self.PSet.loc[jModel])
#                            for jModel in range(n_models)]

#         self.Bad = []
#         self.Emulators = {}
#         self.PCEmulators = {}
#         self.PCAcomponents = {}
#         self.PCAmean = {}

#     def MakeParamSet(self, numsamples=0, IncludeParams='AllParams', ExcludeParams=[], SetParameters={}, WhichEngine='LH'):

#         # Define first model as 'reference'.
#         PSet0 = [{p1: 1. if p1 not in SetParameters.keys() else SetParameters[p1]  for p1 in AllParams }]

#         if numsamples == 0:
#             PSet = []
#         elif numsamples > 0:
#             if IncludeParams == 'AllParams':
#                 TargetParams = list(AllParams)
#             else:
#                 TargetParams = IncludeParams

#             if 'passive' in ExcludeParams:
#                 ExcludeParams.remove('passive')
#                 ExcludeParams += ['a', 'b', 'k', 'eta_l', 'eta_s']
#             for remove1 in ExcludeParams:
#                 if remove1 in TargetParams:
#                     TargetParams.remove(remove1)

#             self.TargetParams = TargetParams

#             numtargetparams = len(TargetParams)

#             if WhichEngine == 'LH':
#                 # RandomSamples = lhsmdu.sample(numtargetparams, numsamples)
#                 RandomSamples = lhd(
#                     np.array([[0., 1.]]*numtargetparams), numsamples).T
#             elif WhichEngine == 'Sobol':
#                 import torch.quasirandom as tq
#                 SE = tq.SobolEngine(
#                     dimension=numtargetparams, scramble=True, seed=8)
#                 RandomSamples = np.array(SE.draw(numsamples)).T
#             # elif WhichEngine == 'regular':
                

#             PSet = [None]*numsamples
#             for s1 in range(numsamples):
#                 PSet[s1] = {}
#                 for p1 in AllParams:   # initialise all parameter factors to 1 by default.
#                     if p1 not in SetParameters.keys():
#                         PSet[s1][p1] = 1.
#                     else:
#                         PSet[s1][p1] = SetParameters[p1]
#                 # change target parameter factors to LH values.
#                 for ip1, p1 in enumerate(TargetParams):
#                     # Create a dictionary entry for each modified parameter p1.
#                     PSet[s1][p1] = self.ParRange[p1][0] + \
#                         (self.ParRange[p1][1]-self.ParRange[p1]
#                          [0])*RandomSamples[ip1, s1]
#         # The first model is always the reference model.
#         return pd.DataFrame(PSet0 + PSet)

#     # def GetX(self):
#     #     # X = np.array([list(dict1.values()) for dict1 in self.PSet])
#     #     X = self.PSet.to_numpy()
#     #     X = np.array([ list(X1) for X1 in X ])
#     #     return X

#     def Save(self, FileSuffix='suffix'):
#         self.Suffix = FileSuffix
#         FileName = CohortSaveDir + CohortFileBase + FileSuffix + '.dat'
#         with open(FileName, 'wb') as fpickle:  # Pickling
#             pickle.dump(self, fpickle)
#         print(f'Saved to {FileName}')

#     @classmethod
#     def Load(cls, FileSuffix=''):
#         FileName = CohortSaveDir + CohortFileBase + FileSuffix + '.dat'
#         with open(FileName, 'rb') as fpickle:   # Unpickling
#             return pickle.load(fpickle)

#     def CleanBad(self, ifHM=False):
#         if not ifHM:
#             assert 0 not in self.Bad, '***********  REFERENCE MODEL IS BAD !! ****************'
#         self.Bad = list(set(self.Bad))
#         self.Bad.sort()
#         print(f'Removing bad models: {self.Bad}')
#         for Bad1 in reversed(self.Bad):
#             self.Models.pop(Bad1)
#             self.PSet.drop(labels=[Bad1], inplace=True)
#             # self.PSet.pop(Bad1)
#         Removed = self.Bad.copy()
#         self.Bad = []
#         return Removed

#     def Filter(self, X1, Model_target):
#         pass
#         # def ifInRange(A, a0, a1):
#         #     return (A >= a0) and (A <= a1)

#         # ifOK = True
#         # for jPcl, Pcl1 in enumerate(LenPcl['active']):
#         #     Tag = f'aLP{jPcl}+'
#         #     F0_target = Model_target.ExpResults[Tag]['F0']
#         #     F0_test, err = self.Emulators[Tag+'_y0'].predict(X1)
#         #     Fend_target = Model_target.ExpResults[Tag]['F'][-1]-F0_target
#         #     Fend_test, err = self.Emulators[Tag+'_Dy'].predict(X1)

#         #     if not ifInRange(F0_test, F0_target*0.9, F0_target*1.1):
#         #         ifOK = False
#         #     # if not ifInRange(Fend_test, Fend_target*0.8, Fend_target*1.2):
#         #     #     ifOK = False

#         # return ifOK

#     # %%% Experiments

#     def DoFpCa(self, ifPlot=False, DLambda=0.0):
#         for iModel, Model1 in enumerate(self.Models):
#             print(f'Doing model {iModel}')
#             try:
#                 Model1.DoFpCa(ifPlot=False, DLambda=DLambda)
#             except:
#                 print(f'Bad model {iModel} ***')
#                 self.Bad.append(iModel)

#         if ifPlot:
#             self.PlotFpCa(DLambda=DLambda)

#     def DoQSP(self, QSBasesPCA, pCa_set, fracdSL_set):
#         for iModel, Model1 in enumerate(self.Models):
#             print(f'Doing model {iModel}')
#             Model1.DoQSP(QSBasesPCA=QSBasesPCA, pCa_set=pCa_set, fracdSL_set=fracdSL_set)
        
#     def DoStretchDestretch(self, dLambda=1.e-5, ifPlot=False):
#         for iModel, Model1 in enumerate(self.Models):
#             print(f'Doing model {iModel}   : dLambda={dLambda:.6f}')
#             Model1.DoStretchDestretch(dLambda=dLambda, ifPlot=False)

#     def DoLengthening(self, pCa_list=None, fracdSL_list=[-0.02,-0.01,0.01,0.02], n_pts=501, tmax=0.5):
#         assert not pCa_list is None, 'Specify pCa_list for Lengthening'
#         for iModel, Model1 in enumerate(self.Models):
#             print(f'Doing model {iModel}   : Lengthening for pCa = {pCa_list}')
#             ifSuccess = Model1.DoLengthening(pCa_list=pCa_list, fracdSL_list=fracdSL_list, n_pts=n_pts, tmax=tmax)
#             if not ifSuccess: self.Bad += [iModel]
#             # print(f'self.Bad = {self.Bad}')
    
#     def Doktr(self, Lambda=1.1, tmax=10.0, WhichReset=None, ifPlot=False):
#         for iModel, Model1 in enumerate(self.Models):
#             print(f' Doing ktr for model {iModel}')
#             Model1.Doktr(Lambda=Lambda, tmax=tmax, WhichReset=WhichReset, ifPlot=False)

#     def DoAllExperiments(self, ifPlot=False, DLambda=0.0):

#         import warnings
#         warnings.filterwarnings("error")
#         tic = time.time()

#         for iModel, Model1 in enumerate(self.Models):
#             print(
#                 f'Doing model {iModel}       time elapsed={round(time.time()-tic)}')
#             try:
#                 Model1.DoAllExperiments(ifPlot=ifPlot)
#             except Exception as Exc1:
#                 print(
#                     f'**************** Bad model {iModel} **************  {Exc1.args}')
#                 self.Bad.append(iModel)
#         warnings.resetwarnings()

#     def PlotFpCa(self, DLambda=0.0, ifHistograms=False):
#         if ifHistograms and (f'Fmax_{1+DLambda :.2f}' not in self.Models[0].Features.keys()):
#             print('Generating FpCa features')

#             for iModel, Model1 in enumerate(self.Models):

#                 try:
#                     Model1.GetFeat_FpCa(Model1.Lambda_ext+DLambda)
#                 except:
#                     self.Bad.append(iModel)
#                     print(f'Bad model : {iModel} ***** ')

#         fig_FpCa = plt.figure(num=f'FpCa (Lambda = {1+DLambda :.2f})')

#         if not ifHistograms:
#             ax_FpCa = fig_FpCa.add_subplot(1, 1, 1)
#             ax_Fmax = None
#             ax_pEC50 = None
#             ax_nH = None
#         else:
#             ax_FpCa = fig_FpCa.add_subplot(2, 1, 1)
#             ax_Fmax = fig_FpCa.add_subplot(2, 3, 4)
#             ax_nH = fig_FpCa.add_subplot(2, 3, 5)
#             ax_pEC50 = fig_FpCa.add_subplot(2, 3, 6)
#             nH_a = [
#                 Model1.Features[f'nH_{Model1.Lambda_ext+DLambda:.2f}'] for Model1 in self.Models]
#             pEC50_a = [
#                 Model1.Features[f'pEC50_{Model1.Lambda_ext+DLambda:.2f}'] for Model1 in self.Models]
#             Fmax_a = [
#                 Model1.Features[f'Fmax_{Model1.Lambda_ext+DLambda:.2f}'] for Model1 in self.Models]

#         for Model1 in self.Models:
#             D = Model1.ExpResults[f'FpCa_{Model1.Lambda_ext+DLambda :.2f}']

#             # D['Cai']=[D['Cai'][j] for j,f in enumerate(D['F']) if not math.isnan(f) ]
#             ax_FpCa.semilogx(10**-D['pCai'], D['F'])
#         if ifHistograms:
#             bins = 10
#             ax_Fmax.hist(Fmax_a, bins=bins, range=(0, max(Fmax_a)*1.1))
#             ax_Fmax.set_xlabel('Fmax')
#             ax_nH.hist(nH_a, bins=bins, range=(0, max(nH_a)*1.1))
#             ax_nH.set_xlabel('nH')
#             ax_pEC50.hist(pEC50_a, bins=bins, range=(0, max(pEC50_a)*1.1))
#             ax_pEC50.set_xlabel('pEC50')
#             fig_FpCa.tight_layout()
#         fig_FpCa.suptitle(f'FpCa (Lambda = {1+DLambda :.2f})')
#         fig_FpCa.tight_layout()

#     # def PlotLengthening(self, dLambda=1.e-5, pCai=4.8, ActiveOrPassive='active', ifNormalise = True):
#     def PlotLengthening(self, jModel=[], ActiveOrPassive='active', Pcl=[0], PoM='+', ifNormalise=True):
#         # Plot forces
#         fig_forces, ax_forces = plt.subplots()

#         if len(jModel) == 0:
#             jModel = list(range(len(self.Models)))

#         for Model1 in [self.Models[j] for j in jModel]:
#             for Pcl1 in Pcl:
#                 # f'{Lengthening_{dLambda:.6f}}'
#                 Tag = f'{ActiveOrPassive[0]}LP{Pcl1}{PoM}'
#                 t = Model1.ExpResults[Tag]['t']
#                 F = Model1.ExpResults[Tag]['F']
#                 # F_S = Model1.ExpResults[Tag]['F_S']
#                 # F_W = Model1.ExpResults[Tag]['F_W']
#                 F0 = Model1.ExpResults[Tag]['F0']
#                 # F0_S = Model1.ExpResults[Tag]['F0_S']
#                 # F0_W = Model1.ExpResults[Tag]['F0_W']
#                 # F_pas = Model1.ExpResults[Tag]['F_pas']
#                 # F0_pas = Model1.ExpResults[Tag]['F0_pas']

#                 if ifNormalise:
#                     normF = F0
#                 else:
#                     normF = 1.
#                 ax_forces.plot(
#                     np.append([-t[-1]/20, 0], t), np.append([F0, F0], F)/normF)

#                 # ax_forces[1,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S[0], F0_S[0]], F_S[0])/normF[0]);
#                 # ax_forces[1,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_S[1], F0_S[1]], F_S[1])/normF[1])
#                 # ax_forces[2,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W[0], F0_W[0]], F_W[0])/normF[0]);
#                 # ax_forces[2,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_W[1], F0_W[1]], F_W[1])/normF[1])
#                 # ax_forces[3,0].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas[0], F0_pas[0]], F_pas[0])/normF[0]);
#                 # ax_forces[3,1].plot(np.append( [-t[-1]/20, 0], t), np.append([F0_pas[1], F0_pas[1]], F_pas[1])/normF[1])

#                 if ifNormalise:
#                     ax_forces.set_ylabel('F_total (norm)')
#                     # ax_forces[1,0].set_ylabel('F_S (norm)')
#                     # ax_forces[2,0].set_ylabel('F_W (norm)')
#                     # ax_forces[3,0].set_ylabel('F_passive (norm)')
#                 else:
#                     ax_forces.set_ylabel('F_total')
#                     # ax_forces[1,0].set_ylabel('F_S')
#                     # ax_forces[2,0].set_ylabel('F_W')
#                     # ax_forces[3,0].set_ylabel('F_passive')

#     def PlotCaiStep(self, pCai1=7, pCai2=5):
#         fig, ax = plt.subplots(num='Cai step')
#         for Model1 in self.Models:
#             Tag = f'CaiStep_{pCai1:.2f}_{pCai2:.2f}'
#             t = Model1.ExpResults[Tag]['t']
#             F = Model1.ExpResults[Tag]['F']
#             ax.plot(np.append([-t[-1]/20, 0], t), np.append([F[0], F[0]], F))
#             ax.set_ylabel('F')
#         fig.suptitle(f'CaiStep  {pCai1:.2f} -> {pCai2:.2f}')

#     def Plotktr(self, Lambda):
#         fig, ax = plt.subplots(num=f'ktr  -> Lambda={Lambda}')
#         for Model1 in self.Models:
#             Model1.Plot_ktr(Lambda, ax)
#         fig.suptitle(f'ktr  -> Lambda={Lambda}')

#     # %%% Extract phenotypes

#     def GetFeat_ktr(self, Lambda=1.0):
#         FeaturesSet = [f'ktr_{Lambda:.2f}']
#         print(f'Getting {FeaturesSet} for cohort')
#         for jModel, Model1 in enumerate(self.Models):
#             Model1.GetFeat_ktr(Lambda=Lambda)
#         # return {Feat1: [Model1.Features[Feat1] for Model1 in self.Models] for Feat1 in FeaturesSet}

#     def GetFeat_FpCa(self, Lambda=1.00):
#         FeaturesSet = [f'Fmax_{Lambda:.2f}',
#                        f'nH_{Lambda:.2f}', f'pEC50_{Lambda:.2f}']
#         print(f'Getting {FeaturesSet} for cohort')
#         for Model1 in self.Models:
#             Model1.GetFeat_FpCa(Lambda=Lambda)
#         return {Feat1: [Model1.Features[Feat1] for Model1 in self.Models] for Feat1 in FeaturesSet}

#     def GetFeat_DeltaFpCa(self, Lambda1=1.10, Lambda2=1.20):
#         FeaturesSet = [f'DeltaFmax_{Lambda1:.2f}_{Lambda2:.2f}',
#                        f'DeltanH_{Lambda1:.2f}_{Lambda2:.2f}', f'DeltapEC50_{Lambda1:.2f}_{Lambda2:.2f}']
#         print(
#             f'Getting DeltaFpCa features (Lambda {Lambda2:.2f} vs {Lambda1:.2f}) for cohort')
#         for Model1 in self.Models:
#             Model1.GetFeat_DeltaFpCa(Lambda1=Lambda1, Lambda2=Lambda2)
#         return {Feat1: [Model1.Features[Feat1] for Model1 in self.Models] for Feat1 in FeaturesSet}

#         return {f'DeltaFmax_{Lambda1}_{Lambda2}': [Model1.Features[f'DeltaFmax_{Lambda1}_{Lambda2}'] for Model1 in self.Models],
#                 f'DeltanH_{Lambda1}_{Lambda2}': [Model1.Features[f'DeltanH_{Lambda1}_{Lambda2}'] for Model1 in self.Models],
#                 f'DeltapEC50_{Lambda1}_{Lambda2}': [Model1.Features[f'DeltapEC50_{Lambda1}_{Lambda2}'] for Model1 in self.Models]}

#     def GetFeat_CaiStep(self, pCai1=4, pCai2=4):
#         Suffix = f'_{pCai1:.2f}_{pCai2:.2f}'
#         FeaturesSet = ['dFCaiStep'+Suffix, 'tCaiStep'+Suffix]
#         print(f'Getting {FeaturesSet} for cohort')
#         for Model1 in self.Models:
#             Model1.GetFeat_CaiStep(pCai1=pCai1, pCai2=pCai2)
#         return {Feat1: [Model1.Features[Feat1] for Model1 in self.Models] for Feat1 in FeaturesSet}

#     def GetFeat_Lengthening(self, LenBasisPCA=None, n_pcaTruncate=None):
#         assert not LenBasisPCA is None, 'Provide LenBasisPCA !'
#         print(f'Getting lengthening features for cohort')
#         # for Model1 in self.Models:
#         #     for jPcl, Pcl1 in enumerate(LenPcl['active']):
#         #         for PoM in ['+', '-']:
#         #             Model1.GetFeat_Lengthening(jPcl, PoM)
#         for Model1 in self.Models:
#             DFpca = Model1.GetFeat_Lengthening(
#                 LenBasisPCA=LenBasisPCA, n_pcaTruncate=n_pcaTruncate, ifPlot=False)
#         return {Feat1: [Model1.Features[Feat1]
#                         for Model1 in self.Models] for Feat1 in self.Models[0].Features if 'LP' in Feat1}
    
#     def Make_QSP_trace(self,  WhichpCa=None, WhichfracdSL=None, n_pts=501, tmax=0.5, ifPlot=False):
#         Traces = []
#         for jModel, Model1 in enumerate(self.Models):
#             print(f'Doing model {jModel}')
#             Traces += [Model1.Make_QSP_trace(WhichpCa=WhichpCa, WhichfracdSL=WhichfracdSL, n_pts=n_pts,tmax=tmax, ifPlot=False, ifSave=True)]
#         return Traces
            
#     def Make_QSPpca(self, n_components=7, WhichpCa=None, WhichfracdSL=None, n_pts=501, tmax=0.5, ifPlot=False):
#         Traces = np.array(self.Make_QSP_trace(WhichpCa=WhichpCa, WhichfracdSL=WhichfracdSL, n_pts=n_pts, tmax=tmax, ifPlot=False))
#         QSPpca = PCA(n_components=n_components)
#         QSPpca.fit(Traces)       # Mean is automatically subtracted in "fit".

        
#         if ifPlot:
#             fig = plt.figure()
#             ax_raw = fig.add_subplot(3,2,1)
#             for Trace1 in Traces:
#                 ax_raw.plot(Trace1)
#             ax_raw.set_ylabel('data')
#             ax_mean = fig.add_subplot(3,2,3)
#             ax_mean.plot(QSPpca.mean_)
#             ax_mean.set_ylabel('mean')
#             for j in range(n_components):
#                 ax_pca = fig.add_subplot(n_components, 2, 2*j+2)
#                 ax_pca.plot(QSPpca.components_[j])
#                 ax_pca.plot(ax_pca.get_xlim(), [0,0], 'k:')
#                 ax_pca.set_ylabel(rf'$c_{{{j}}}$')
#                 ax_pca.set_yticks([])
#                 if j < n_components-1: ax_pca.set_xticks([])
#             ax_weights = fig.add_subplot(3,2,5)
#             ax_weights.semilogy(QSPpca.explained_variance_ratio_, '.')
#             ax_weights.set_xlabel('component')
#             ax_weights.set_ylabel('rel expl var')
            
#         return {'components': QSPpca.components_,
#                 'mean': QSPpca.mean_,
#                 'rel_expl_var': QSPpca.explained_variance_ratio_,
#                 'singular_values': QSPpca.singular_values_,
#                 'pCa_list': WhichpCa,
#                 'fracdSL_list': WhichfracdSL,
#                 'n_pts': n_pts,
#                 'tmax': tmax}

#     def GetFeat_QSPproj(self, QSPpca):
#         for Model1 in self.Models:
#             Model1.GetFeat_QSPproj(QSPpca)
        
#     def GetFeat_phenVec(self, LenBasisPCA, phvec_key, EMPCA, n_pcaTruncate=None, ifNormalise=True, ifPlot=False):
#         if n_pcaTruncate is None:
#             n_pcaTruncate = len(LenBasisPCA.components_)
#         print('Getting phenVec for cohort')
#         for Model1 in self.Models:
#             Model1.GetFeat_phenVec(LenBasisPCA, phvec_key, EMPCA, n_pcaTruncate=n_pcaTruncate, ifNormalise=ifNormalise)
            
#         # if ifPlot:
#         #     fig, ax = plt.subplots()
#         #     ax.plot([Model1.Features['phenVec'] for Model1 in self.Models])
            
#         return [[Model1.Features['phenVec'][jcomp] for jcomp in range(EMPCA.nvec)] for Model1 in self.Models]
        
        
#     # def MakeLenBasis(self, n_components=5):
#     #     Y = []
#     #     y0 = []
#     #     for jModel, Model1 in enumerate(self.Models):
#     #         for jPcl, Pcl1 in enumerate(LenPcl['active']):
#     #             for PoM in ['+', '-']:
#     #                 Tag = f'aLP{jPcl}{PoM}'
#     #                 y0 += [Model1.ExpResults[Tag]['F0']]
#     #                 Y += [ [jModel, Pcl1['pCa'], Pcl1['dLambda']*PoMint(PoM) ] + list(np.array(Model1.ExpResults[Tag]['F']) / Model1.ExpResults[Tag]['F0'] - 1)      ]

#     #     DF = pd.DataFrame(data=Y, columns=['jModel', 'pCa', 'dLambda'] + list(t))
#     #     DF = DF.set_index(['jModel', 'pCa', 'dLambda'])

#     #     LenBasisPCA = PCA(n_components=n_components)
#     #     LenBasisPCA.fit(DF.to_numpy())

#     #     DFpca = pd.DataFrame(data=LenBasisPCA.transform(DF.to_numpy()), index=DF.index)

#     #     LenBasisPCA.DF = DF
#     #     LenBasisPCA.DFpca = DFpca

#     #     return LenBasisPCA

#     # def GetFeat_Lengthening(self, ActiveOrPassive='active',  PoM='+', n_components=5):
#     #     self.nLengtheningComponents = n_components
#     #     print(f'Getting {ActiveOrPassive[0]}({PoM})LP_n0 to _n{n_components}')

#     #     Y = []
#     #     y0 = []

#     #     # Construct arrays
#     #     for Model1 in self.Models:
#     #         for jPcl in range(len(LenPcl[ActiveOrPassive])):
#     #             Tag = f'{ActiveOrPassive[0]}LP{jPcl}{PoM}'

#     #             y0 += [Model1.ExpResults[Tag]['F0']]
#     #             y_ = np.array(Model1.ExpResults[Tag]['F']) / Model1.ExpResults[Tag]['F0'] - 1
#     #             Y += [y_]

#     #     # Perform PCA on array
#     #     Y = np.array(Y)
#     #     # Dy = np.array(Dy)
#     #     pca = PCA(n_components=n_components)
#     #     pca.fit(Y)
#     #     self.PCAcomponents[ActiveOrPassive+PoM] = pca.components_
#     #     self.PCAmean[ActiveOrPassive+PoM] = pca.mean_
#     #     c = pca.components_
#     #     Ypca = pca.transform(Y)

#     #     # Extract PC coefficients and assign them to features in individual models
#     #     Ypca_iter = iter(Ypca)        # NB   Order matters!! : first Models, then LPs
#     #     y0_iter   = iter(y0)
#     #     for Model1 in self.Models:
#     #         for jPcl in range(len(LenPcl[ActiveOrPassive])):
#     #             Ypca1 = next(Ypca_iter)
#     #             Tag = f'{ActiveOrPassive[0]}LP{jPcl}{PoM}'
#     #             for n in range(n_components):
#     #                 Model1.Features[Tag+f'_n{n}'] = Ypca1[n]
#     #             Model1.Features[Tag+'_y0'] = next(y0_iter)

#     #     # Generate Features dictionaries for Lengthening
#     #     FeatDict_n = {}
#     #     FeatDict_y0 = {}
#     #     for jPcl in range(len(LenPcl[ActiveOrPassive])):
#     #         Tag = f'{ActiveOrPassive[0]}LP{jPcl}{PoM}'
#     #         for n in range(n_components):
#     #                 FeatDict_n[Tag+f'_n{n}'] = [Model1.Features[Tag+f'_n{n}'] for Model1 in self.Models]
#     #         FeatDict_y0[Tag+'_y0'] = [Model1.Features[Tag+'_y0'] for Model1 in self.Models ]
#     #     return {**FeatDict_n,
#     #             **FeatDict_y0}

#     def GetAllFeatures(self, n_components=5):

#         Features = {  # **self.GetFeat_ktr(Lambda=1.1),
#             # **self.GetFeat_ktr(Lambda=1.2),
#             **self.GetFeat_FpCa(Lambda=1.1),
#             **self.GetFeat_FpCa(Lambda=1.2),
#             **self.GetFeat_FpCa(Lambda=0.90),
#             **self.GetFeat_DeltaFpCa(Lambda1=1.1, Lambda2=1.2),
#             **self.GetFeat_Lengthening()}
#         # **self.GetFeat_Lengthening('active', PoM='+', n_components=n_components),
#         # **self.GetFeat_Lengthening('active',  PoM='-', n_components=n_components)} #,# **self.GetFeat_CaiStep(10**-7, 10**-5)}
#         return Features

#     def DisplayLengtheningPCA(self, iModel=0, ActiveOrPassive='active', PoM='+', iPcl=0):

#         fig = plt.figure()
#         for n1 in range(n_components):
#             ax = fig.add_subplot(n_components, 2, 2*n1+2)
#             ax.plot(self.PCAcomponents[ActiveOrPassive+PoM][n1])
#             ax.axis('off')

#         ax = fig.add_subplot(2, 2, 1)
#         Tag = f'{ActiveOrPassive[0]}LP{iPcl}{PoM}'
#         y = self.Models[iModel].ExpResults[Tag]['F']
#         t = self.Models[iModel].ExpResults[Tag]['t']
#         y0 = self.Models[iModel].ExpResults[Tag]['F0']
#         ax.plot(np.append([-t[-1]/20, 0], t),
#                 np.append([y0, y0], y))  # -y[-1])

#         yemul = [self.Models[iModel].Features[Tag+f'_n{n1}'] * self.PCAcomponents[ActiveOrPassive+PoM][n1]
#                  for n1 in range(n_components)]
#         yemul = (np.sum(np.array(yemul), axis=0) +
#                  self.PCAmean[ActiveOrPassive+PoM]) * y0 + y0

#         ax.plot(t, yemul, 'r--')

#         ax = fig.add_subplot(2, 2, 3)
#         ax.bar(list(range(n_components)), [
#                self.Models[iModel].Features[Tag+f'_n{n1}'] for n1 in range(n_components)])
#         ax.set_xticks(list(range(n_components)))
#         ax.set_xlabel('PCA components')

#         fig.suptitle(f'(Model {iModel}) {Tag}')
#         fig.tight_layout()

#     def Show_results_FpCa(self, WhichModels=None, Lambda=1.0, ):
#         fig, ax = plt.subplots()
#         if WhichModels == None:
#             WhichModels = list(range(len(self.Models)))
#         for Model1 in [self.Models[i1] for i1 in WhichModels]:
#             Y = Model1.ExpResults[f'FpCa_{Lambda:.2f}']
#             ax.semilogx(10**-Y['pCai'], Y['F'])

#     def MapFeature(self, Feature, ifActual=True, ifEmulated=False):
#         pass
#         # ncols = np.floor(np.sqrt(self.IncludeParams))
#         # nrows = np.ceil(np.sqrt(self.IncludeParams))
#         # fig = plt.figure()
#         # for jparam, param1 in enumerate(self.IncludeParams):
#         #     print(f'Doing {param1}')
#         #     ax = fig.add_subplot(nrows, ncols, jparam+1)

#     # %%% Emulators

#     def ReadFeature(self, WhichFeature):
#         return [Model1.Features[WhichFeature] for Model1 in self.Models]

#     # def MakeEmulator(self, WhichFeature, ifLoad=False):
#     #     def Get_best_model_link(link1):
#     #         """
#     #         The file 'best_model.pth' is a link to the best restart of the emulator training. 
#     #         The link is platform-dependent. This 'translates' it so that it can be read by the present system.
#     #         """
#     #         from os import readlink
#     #         Sold = readlink(link1)
#     #         # Snew = home + Sold[(Sold.find('Dropbox')-1):]
#     #         Snew = home + Sold[(Sold.find('GPE')-1):]
#     #         print(Snew)
#     #         return Snew

#     #     # split original dataset in training, validation and testing sets
#     #     Not_None = [j for j, y in enumerate(self.ReadFeature(WhichFeature)) if y is not None]
#     #     from sklearn.model_selection import train_test_split
#     #     X_, X_test, y_, y_test = train_test_split(
#     #         self.PSet[self.TargetParams].to_numpy()[Not_None],
#     #         np.array(self.ReadFeature(WhichFeature), dtype=float)[Not_None],
#     #         test_size=0.2,
#     #         random_state=8)
#     #     X_train, X_val, y_train, y_val = train_test_split(
#     #         X_,
#     #         y_,
#     #         test_size=0.2,
#     #         random_state=8)

        
#     #     print(f'Emulating   {WhichFeature}')
#     #     # dataset = Dataset(X_train=self.PSet[self.TargetParams].to_numpy(),
#     #     #                   y_train=self.ReadFeature(WhichFeature))  # ,
#     #     dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
#     #     # X_val=X_val,
#     #     # y_val=y_val,
#     #     # X_test=X_test,
#     #     # y_test=y_test)
#     #     # l_bounds=[self.ParRange[param1][0] for param1 in AllParams],
#     #     # u_bounds=[self.ParRange[param1][1] for param1 in AllParams])

#     #     device = "cpu"

#     #     # self.emulator_config_file = f'{home}/Dropbox/Python/BHFsim/saved_emulators/Cohort_{self.Suffix}_emulator_config_{WhichFeature}.ini'
#     #     self.emulator_config_file = f'{home}/GPE/saved_emulators/Cohort_{self.Suffix}_emulator_config_{WhichFeature}.ini'
#     #     if not ifLoad:
#     #         likelihood = GaussianLikelihood()
#     #         mean_function = LinearMean(input_size=dataset.input_size)
#     #         kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
#     #         metrics = [MeanSquaredError(), R2Score()]
#     #         experiment = GPExperiment(
#     #             dataset,
#     #             likelihood,
#     #             mean_function,
#     #             kernel,
#     #             n_restarts=3,
#     #             metrics=metrics,
#     #             seed=GPEseed,  # reproducible training
#     #             learn_noise=False  #
#     #         )
#     #         experiment.save_to_config_file(self.emulator_config_file)

#     #         emulator = GPEmulator(experiment, device)

#     #         from GPErks.train.snapshot import NeverSaveSnapshottingCriterion
#     #         # self.emulator_snapshot_dir = f'{home}/Dropbox/Python/BHFsim/saved_emulators/Cohort_{self.Suffix}_emulator_{WhichFeature}'
#     #         self.emulator_snapshot_dir = f'{home}/GPE/saved_emulators/Cohort_{self.Suffix}_emulator_{WhichFeature}'
#     #         train_restart_template = "restart_{restart}"
#     #         train_epoch_template = "epoch_{epoch}.pth"
#     #         emulator_snapshot_file = train_epoch_template
#     #         snpc = NeverSaveSnapshottingCriterion(
#     #             f'{self.emulator_snapshot_dir}/{train_restart_template}', emulator_snapshot_file)

#     #         optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
#     #         # esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

#     #         print(f'Training emulator for {WhichFeature}')
#     #         best_model, best_train_stats = emulator.train(
#     #             optimizer, snapshotting_criterion=snpc)
#     #         print(
#     #             f'   Emulator training completed and saved for {WhichFeature}')

#     #     else:
#     #         from GPErks.gp.experiment import load_experiment_from_config_file
#     #         # Load skeleton
#     #         experiment = load_experiment_from_config_file(
#     #             self.emulator_config_file,
#     #             dataset  # notice that we still need to provide the dataset used!
#     #         )

#     #         # best_model_file = Get_best_model_link(f'{home}/Dropbox/Python/BHFsim/saved_emulators/emulator_{WhichFeature}/best_model.pth')
#     #         best_model_file = Get_best_model_link(
#     #             f'{home}/GPE/saved_emulators/emulator_{WhichFeature}/best_model.pth')

#     #         best_model_state = torch.load(
#     #             best_model_file, map_location=torch.device(device))

#     #         emulator = GPEmulator(experiment, device)
#     #         emulator.model.load_state_dict(best_model_state)
#     #         print(f'   Emulator loaded for {WhichFeature}')

#     #     self.Emulators[WhichFeature] = emulator

#     def MakeAllEmulators(self, WhichFeatures=[]):
#         if WhichFeatures == []:
#             WhichFeatures = self.Models[0].Features.keys()
#         for Feat1 in WhichFeatures:
#             self.MakeEmulator(Feat1)

#     # def TestEmulatorPredictions(self, Model_ref):
#     #     '''
#     #     Compare the values of whichever features are available in Model_ref with the emulator predictions.
#     #     '''
#     #     FoI = Model_ref.Features.keys()
#     #     Y_true = np.array([None]*len(FoI))
#     #     Y_pred = np.array([None]*len(FoI))
#     #     Y_perr = np.array([None]*len(FoI))
#     #     for iFeat, Feat1 in enumerate(FoI):
#     #         print(f' Testing {Feat1} emulator')
#     #         Y_true[iFeat] = Model_ref.Features[Feat1]
#     #         Y_pred[iFeat], Y_perr[iFeat] = self.Emulators[Feat1].predict([Model_ref.PSet[self.TargetParams].to_numpy()])
#     #         print(f'True {Feat1} = {Y_true[iFeat]}')
#     #         print(f'   Emulated {Feat1} = {Y_pred[iFeat]} +/- {Y_perr[iFeat]}')
#     #     fig1, ax1 = plt.subplots()
#     #     ax1.errorbar(x=np.arange(len(FoI)),
#     #          y= Y_pred/Y_true,
#     #          yerr = Y_perr/Y_true,
#     #          capthick=10,
#     #          marker='o',
#     #          linestyle='')
#     #     ax1.plot(ax1.axes.get_xlim(), [1,1], 'k--')
#     #     ax1.axes.xaxis.set_ticks(np.arange(len(FoI)))
#     #     ax1.set_xticklabels(FoI, rotation = 'vertical')
#     #     ax1.set_ylabel('Features:   emulated/true')
#     #     fig1.tight_layout()

#     def TestEmulators(self, PSet1=None):

#         if type(PSet1) == int:
#             Model_test = self.Models[PSet1]
#             PSet1 = Model_test.PSet

#         # If PSet1 is 'None', generate random model.
#         elif type(PSet1) == type(None):
#             ifOK = False
#             while not ifOK:
#                 try:
#                     PSet1 = self.MakeParamSet(numsamples=1, IncludeParams=self.TargetParams, ExcludeParams=[
#                                               'passive'], WhichEngine='LH').loc[1]
#                     Model_test = Land2017(PSet1)
#                     Model_test.DoAllExperiments()

#                     ifOK = True
#                 except:
#                     print('Changing model.')

#         elif type(PSet1) == pd.core.series.Series:
#             try:
#                 Model_test = Land2017(PSet1)
#                 Model_test.DoAllExperiments()
#             except:
#                 print('Model failed in the experiments.')

#         if type(PSet1) in [type(None), pd.core.series.Series]:
#             for Expt1 in Model_test.ExpResults.keys():
#                 if 'FpCa' in Expt1:
#                     Lambda = float(Expt1[5:])
#                     Model_test.GetFeat_FpCa(Lambda)
#                 if 'aLP' in Expt1:
#                     jPcl = int(Expt1[3:-1])
#                     PoM = Expt1[-1]
#                     Model_test.GetFeat_Lengthening(jPcl, PoM)

#         # for Feat1 in self.Emulators.keys():
#         #     Model_test.Features[Feat1], err = self.Emulators[Feat1].predict([PSet1[self.TargetParams].to_numpy()])

#         Model_test.ShowExperiments(
#             self.Emulators, self.TargetParams, LenBasisPCA.components_, LenBasisPCA.mean_)

#         ''' ======================================================
#         Compare the values of whichever features are available in Model_ref with the emulator predictions.
#         '''
#         FoI = Model_test.Features.keys(
#         )  # [keys1 for keys1 in Model_test.Features.keys() if ('_y0' in keys1) or not ('LP' in keys1)]
#         Y_true = np.array([None]*len(FoI))
#         Y_pred = np.array([None]*len(FoI))
#         Y_perr = np.array([None]*len(FoI))
#         for iFeat, Feat1 in enumerate(FoI):
#             print(f' Testing {Feat1} emulator')
#             Y_true[iFeat] = Model_test.Features[Feat1]
#             Y_pred[iFeat], Y_perr[iFeat] = self.Emulators[Feat1].predict(
#                 [Model_test.PSet[self.TargetParams].to_numpy()])
#             print(f'True {Feat1} = {Y_true[iFeat]}')
#             print(f'   Emulated {Feat1} = {Y_pred[iFeat]} +/- {Y_perr[iFeat]}')
#         fig1, ax1 = plt.subplots()
#         ax1.errorbar(x=np.arange(len(FoI)),
#                      y=Y_pred/Y_true,
#                      yerr=Y_perr/Y_true,
#                      capthick=10,
#                      marker='o',
#                      linestyle='')
#         ax1.plot(ax1.axes.get_xlim(), [1, 1], 'k--')
#         ax1.axes.xaxis.set_ticks(np.arange(len(FoI)))
#         ax1.set_xticklabels(FoI, rotation='vertical')
#         ax1.set_ylabel('Features:   emulated/true')
#         fig1.tight_layout()

#     def MakeAllEmulators_pca(self):
#         nPC = len(self.Models[0].Features_pca)
#         for ipc in range(nPC):
#             print(f'Emulating PC{ipc} out of {nPC}')
#             Features_pca = [Model1.Features_pca[ipc] for Model1 in self.Models]
#             dataset = Dataset(X_train=self.GetX(),
#                               y_train=Features_pca,
#                               l_bounds=[self.ParRange[param1][0]
#                                         for param1 in AllParams],
#                               u_bounds=[self.ParRange[param1][1] for param1 in AllParams])
#             device = "cpu"

#             pcemulator_config_file = f'{home}/Dropbox/Python/BHFsim/saved_emulators/Cohort_{self.Suffix}_pcemulator_config_{ipc}.ini'
#             likelihood = GaussianLikelihood()
#             mean_function = LinearMean(input_size=dataset.input_size)
#             kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
#             metrics = [MeanSquaredError(), R2Score()]
#             experiment = GPExperiment(
#                 dataset,
#                 likelihood,
#                 mean_function,
#                 kernel,
#                 n_restarts=3,
#                 metrics=metrics,
#                 seed=GPEseed,  # reproducible training
#                 learn_noise=False  # True  #
#             )
#             experiment.save_to_config_file(pcemulator_config_file)

#             emulator = GPEmulator(experiment, device)

#             from GPErks.train.snapshot import NeverSaveSnapshottingCriterion
#             pcemulator_snapshot_dir = f'{home}/Dropbox/Python/BHFsim/saved_emulators/Cohort_{self.Suffix}_pcemulator_{ipc}'
#             train_restart_template = "restart_{restart}"
#             train_epoch_template = "epoch_{epoch}.pth"
#             emulator_snapshot_file = train_epoch_template
#             snpc = NeverSaveSnapshottingCriterion(
#                 f'{pcemulator_snapshot_dir}/{train_restart_template}', emulator_snapshot_file)

#             optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
#             # esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

#             print(f'Training emulator for PC {ipc}')
#             best_model, best_train_stats = emulator.train(
#                 optimizer, snapshotting_criterion=snpc)
#             print(f'   Emulator training completed and saved for PC{ipc}')

#             self.PCEmulators[ipc] = emulator

#     def EmulateModel(self, PSet1, ifPlot=False):
#         # for par1 in AllParams:
#         #     assert PSet1[par1] == 1. or par1 in self.TargetParams
#         Model1 = Land2017(PSet1)
#         for Feat1 in self.Emulators.keys():
#             Model1.Features[Feat1], err = self.Emulators[Feat1].predict(
#                 [Model1.PSet[self.TargetParams].to_numpy()])

#         # %% Emulate FpCa
#         WhichLambda = [float(temp[5:])
#                        for temp in Model1.Features.keys() if 'Fmax' in temp]
#         for Lambda1 in WhichLambda:
#             pCai = np.linspace(7, 4, 50)
#             Model1.ExpResults[f'FpCa_{Lambda1}'] = {'pCai': pCai,
#                                                     'F': HillFn(10**-pCai, Model1.Features[f'Fmax_{Lambda1}'], Model1.Features[f'nH_{Lambda1}'], Model1.Features[f'pEC50_{Lambda1}']),
#                                                     }
#             if ifPlot:
#                 fig, ax = plt.subplots()
#                 ax.semilogx(
#                     10**-pCai, Model1.ExpResults[f'FpCa_{Lambda1}']['F'])

#         return Model1

#     # %% GSA

#     # def DoGSA(self, WhichFeatures=[], ifPlot=True, ifSave=False):
#     #     # pass
#     #     GSA_S1 = {}
#     #     GSA = {}
#     #     if WhichFeatures == []:
#     #         WhichFeatures = list(self.Models[0].Features.keys())
#     #     elif WhichFeatures == 'FpCa':
#     #         WhichFeatures = [Feat1 for Feat1 in self.Models[0].Features.keys()
#     #                           if Feat1.find('Fmax') != -1
#     #                           or Feat1.find('nH') != -1
#     #                           or Feat1.find('pEC50') != -1]
#     #     elif WhichFeatures == 'lengthening':
#     #         WhichFeatures = [Feat1 for Feat1 in self.Models[0].Features.keys()
#     #                           if Feat1.find('LP') != -1]
#     #     elif WhichFeatures == 'ktr':
#     #         WhichFeatures = [Feat1 for Feat1 in self.Models[0].Features.keys()
#     #                           if Feat1.find('ktr') != -1]

#     #     for Feat1 in WhichFeatures:
#     #         assert Feat1 in self.Emulators.keys(
#     #         ), f'Emulator missing for {Feat1}'
#     #     if ifPlot:
#     #         if len(WhichFeatures) < 4:
#     #             numrows = 1
#     #             numcols = len(WhichFeatures)
#     #         else:
#     #             numrows = math.floor(np.sqrt(len(WhichFeatures)))
#     #             numcols = math.ceil(len(WhichFeatures)/numrows)
#     #         figS1 = plt.figure(figsize=([14,  9]))
#     #         axS1 = [figS1.add_subplot(numrows, numcols, j+1)
#     #                 for j in range(len(WhichFeatures))]

#     #         # if len(WhichFeatures)==1: axS1 = np.array([axS1])
#     #         # axS1 = np.ravel(axS1)
#     #         figCumulS1, axCumulS1 = plt.subplots()

#     #     CumulS1 = np.array([0.]*len(self.TargetParams))

#     #     tic = time.time()
#     #     X_train = self.PSet[self.TargetParams].to_numpy()
#     #     for iFeat, Feat1 in enumerate(WhichFeatures):
#     #         print(
#     #             f'Starting GSA for {Feat1} ----------------------------  time elapsed = {round(time.time()-tic)}')
#     #         FeatureData = np.array([Model1.Features[Feat1]
#     #                                 for Model1 in self.Models])
#     #         # FeatureData /= FeatureData[0]         # REMOVED AL 2022 05 10
#     #         dataset = Dataset(X_train=X_train,
#     #                           y_train=FeatureData,
#     #                           l_bounds=[self.ParRange[param1][0]
#     #                                     for param1 in self.TargetParams],  # AllParams],
#     #                           u_bounds=[self.ParRange[param1][1] for param1 in self.TargetParams])  # AllParams])
#     #         gsa1 = SobolGSA(dataset, n=128, seed=GPEseed)
#     #         gsa1.estimate_Sobol_indices_with_emulator(
#     #             self.Emulators[Feat1], n_draws=100)
#     #         print(f'   GSA complete for {Feat1} ----------------------------')
#     #         gsa1.summary()

#     #         if ifPlot == True:
#     #             sns.boxplot(ax=axS1[iFeat], data=gsa1.S1)
#     #             axS1[iFeat].set_title(Feat1)
#     #             if len(WhichFeatures) > 30:
#     #                 axS1[iFeat].set_xticks([])
#     #             else:
#     #                 axS1[iFeat].set_xticklabels(self.TargetParams, rotation=90)

#     #         GSA[Feat1] = gsa1
#     #         GSA_S1[Feat1] = gsa1.S1
#     #         CumulS1 += np.mean(gsa1.S1, axis=0)

#     #     if ifPlot:
#     #         figS1.tight_layout()
#     #         axCumulS1.barh(self.TargetParams, CumulS1, align='center')

#     #     if ifSave:
#     #         import pickle
#     #         with open('GSA_S1.dat', 'wb') as fpickle:
#     #             pickle.dump(GSA_S1, fpickle)

#     #     return (GSA, CumulS1)

#     def DoGSA_PC(self, ifPlot=True, ifSave=True):
#         GSA_S1 = {}
#         GSA = {}

#         nPC = len(self.Models[0].Features_pca)

#         if ifPlot:
#             numrows = math.floor(np.sqrt(nPC))
#             numcols = math.ceil(nPC/numrows)
#             figS1, axS1 = plt.subplots(
#                 numrows, numcols, figsize=([14,  9]), num='GSA')
#             if nPC == 1:
#                 axS1 = np.array([axS1])
#             axS1 = np.ravel(axS1)
#             figCumulS1, axCumulS1 = plt.subplots(
#                 num='Summary - Cumulative sensitivities')

#         CumulS1 = np.array([0.]*len(AllParams))

#         X_train = self.GetX()
#         for ipc in range(nPC):
#             print(
#                 f'Starting GSA for PC{ipc}  out of {nPC} ----------------------------')
#             FeatureData = np.array([Model1.Features_pca[ipc]
#                                    for Model1 in self.Models])
#             FeatureData /= FeatureData[0]
#             dataset = Dataset(X_train=X_train,
#                               y_train=FeatureData,
#                               l_bounds=[self.ParRange[param1][0]
#                                         for param1 in AllParams],
#                               u_bounds=[self.ParRange[param1][1] for param1 in AllParams])
#             gsa1 = SobolGSA(dataset, n=128, seed=GPEseed)
#             gsa1.estimate_Sobol_indices_with_emulator(
#                 self.PCEmulators[ipc], n_draws=100)
#             print(f'   GSA complete for PC{ipc} ----------------------------')
#             gsa1.summary()

#             if ifPlot == True:
#                 sns.boxplot(ax=axS1[ipc], data=gsa1.S1)
#                 axS1[ipc].set_title(f'PC{ipc}')
#                 axS1[ipc].set_xticklabels(AllParams, rotation=90)

#             GSA[ipc] = gsa1
#             GSA_S1[ipc] = gsa1.S1
#             CumulS1 += np.mean(gsa1.S1, axis=0)

#         if ifPlot:
#             figS1.tight_layout()
#             axCumulS1.barh(AllParams, CumulS1, align='center')

#         if ifSave:
#             import pickle
#             with open('pcGSA_S1.dat', 'wb') as fpickle:
#                 pickle.dump(GSA_S1, fpickle)

#         return (GSA, CumulS1)

#     def PCtoNormTD(self, aLPx_n=[], PoM='+'):
#         """
#         Input: PC coefficients 
#         Returns: time-dependent lengthening transient, normalised by initial resting value.
#         """
#         assert len(self.PCAcomponents['active'+PoM]) == len(aLPx_n),  \
#             f"Number  of pc vectors ({len(self.PCAcomponents['active'+PoM])})  must match number of coefficients ({len(aLPx_n)})."
#         y = self.PCAmean['active'+PoM].copy()
#         for ja, a1 in enumerate(aLPx_n):
#             y += a1 * self.PCAcomponents['active'+PoM][ja]
#         return (y + 1)

#     # %% PCA
#     def DoPCA(self, WhichFeatures=[], MaxComponents=10, ifPlot=True):
#         if WhichFeatures == []:
#             WhichFeatures = list(self.Models[0].Features.keys())

#         from sklearn.decomposition import PCA
#         pca = PCA()
#         Y = np.array([[Model1.Features[feat1]
#                      for feat1 in WhichFeatures] for Model1 in self.Models])
#         # Y /= np.mean(np.abs(Y), axis=0)
#         Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
#         # Y /= Y[0]      # <-------- Normalise all features against the values of the reference model!!
#         # Y1 = np.array([y1/np.std(y1) for y1 in Y])
#         # Y = (Y.T / np.std(Y.T)).T
#         # Y1 = np.array(
#         #     [[ y1[ifeat]/self.Models[0].Features[feat1] for ifeat, feat1 in enumerate(WhichFeatures)] for y1 in Y])
#         pca.fit(Y)

#         if ifPlot:
#             fig_pca, ax_pca = plt.subplots()
#             ax_pca.bar(np.arange(len(pca.explained_variance_ratio_)),
#                        pca.explained_variance_ratio_)
#             ax_pca.set_xlim((-1, 26))
#             ax_pca.set_ylabel('explained variance ratio')
#             ax_pca.set_xlabel('principal components')
#             fig_pca.suptitle(f'PCA')

#         for i1, c1 in enumerate(pca.components_[0:MaxComponents]):
#             jPCmain = list(reversed(np.argsort(np.abs(c1))))[0:MaxComponents]
#             print(f'PC{i1}')

#             for j in jPCmain:
#                 print(f'     {WhichFeatures[j]:20s}    :  {c1[j]:.2f}')

#         for i1 in range(len(self.Models)):
#             self.Models[i1].Features_pca = pca.components_[
#                 0:MaxComponents] @ Y[i1]

#         return pca, Y

#     # %% Next wave
#     def NextWave(self, numsamples, Ithresh=1):
#         pass
#         # TargModel = self.Models[0]
#         # TargFeat = TargModel.Features

#         # def Impl(Feat, errFeat, X1):

#         #     ytest, errtest = self.Emulators[Feat].predict([X1])
#         #     ytarg = TargFeat[Feat]
#         #     Implausibility_measure = (
#         #         ytest-ytarg)**2/(errtest**2 + (errFeat)**2)
#         #     return Implausibility_measure > Ithresh

#         # def Impl_LP(jPcl, PoM, X1):

#         #     Forceerror = 1000  # Pa      <-- assumed force noise
#         #     Tag = f'aLP{jPcl}{PoM}'

#         #     n_components = len(self.PCAcomponents['active'+PoM])
#         #     a = []
#         #     for n in range(n_components):
#         #         a1, aerr = self.Emulators[f'{Tag}_n{n}'].predict([X1])
#         #         a += [a1]
#         #     ynorm = self.PCtoNormTD(a, PoM)

#         #     # = mean dispcrepancy in Pa
#         #     LPerr = np.mean(
#         #         ynorm * TargModel.ExpResults[Tag]['F0'] - TargModel.ExpResults[Tag]['F'])

#         #     Implausibility_measure = LPerr**2/((Forceerror)**2)

#         #     if Implausibility_measure > Ithresh:
#         #         return True

#         # import torch.quasirandom as tq
#         # numtargetparams = len(self.TargetParams)

#         # SE = tq.SobolEngine(dimension=numtargetparams, scramble=True, seed=8)
#         # RandomSamples = np.array(SE.draw(int(numsamples)))

#         # XX = np.ones(RandomSamples.shape)
#         # for j1, p1 in enumerate(self.TargetParams):
#         #     XX[:, j1] = self.ParRange[p1][0] + \
#         #         (self.ParRange[p1][1]-self.ParRange[p1][0]) * \
#         #         RandomSamples[:, j1]

#         # NIMPL = []
#         # tic = time.time()
#         # for j1, X1 in enumerate(XX):
#         #     if j1 % 1000 == 0:
#         #         print(
#         #             f'Done {j1} ({round(time.time()-tic)}),  good so far: {len(NIMPL)}')

#         #     ifIMPL = False
#         #     for Lambda in [1.1, 1.2]:
#         #         if Impl(f'Fmax_{Lambda}', 100, X1):
#         #             ifIMPL = True
#         #         if Impl(f'nH_{Lambda}', 0.2, X1):
#         #             ifIMPL = True
#         #         if Impl(f'pEC50_{Lambda}', 0.5, X1):
#         #             ifIMPL = True

#         #     if not ifIMPL:
#         #         for jPcl, Pcl1 in enumerate(LenPcl['active']):
#         #             for PoM in ['+', '-']:
#         #                 if Impl_LP(jPcl, PoM, X1):
#         #                     ifIMPL = True

#         #     if not ifIMPL:
#         #         NIMPL += [list(X1)]

#         # return NIMPL

# def MakeQSPBasis(TraceList, MaskList=None, tmax=0.5, n_pts=100, n_components=7, ifPlot=False):
#     from scipy.interpolate import interp1d   
#     import empca
#     t = np.linspace(0., tmax, n_pts)
    
#     # ifFullRecord = np.array(MaskList).all()
#     EMPCA = empca.empca(np.array(TraceList), np.array(MaskList), nvec=n_components)
        
#     # for Y1 in TraceList:
#     #     QSP_fPCA = PCA(n_components=n_components)
#     #     QSP_fPCA = 

#     return EMPCA        




#%% CohortDF

    #%%% Definition

class CohortDF:

    
    def MakePSetLH(self, numsamples=0, IncludeParams='AllParams', ExcludeParams=[], SetParameters={},  ParFac=2, ParRange={}, WhichEngine='LH', WhichDep='force'):
        # Define list of parameters to be included in the LH sampling: IncludeParams
        
        if type(IncludeParams) == str:
            IncludeParams = [IncludeParams]
        if type(ExcludeParams) == str:
            ExcludeParams = [ExcludeParams]        
        if 'AllParams' in IncludeParams:
            IncludeParams.remove('AllParams')
            IncludeParams += list(AllParams)
        if 'passive' in IncludeParams:
            IncludeParams.remove('passive')
            IncludeParams += ['a', 'b', 'k', 'eta_l', 'eta_s']
        if 'passive' in ExcludeParams:
            ExcludeParams.remove('passive')
            ExcludeParams += ['a', 'b', 'k', 'eta_l', 'eta_s']
        for param1 in ExcludeParams + list(SetParameters.keys()):
            if param1 in IncludeParams:
                IncludeParams.remove(param1)
        self.IncludeParams = IncludeParams
                
        # Define parameter bounds
        self.ParRange = ParRange
        self.ParBounds = {}
        
        for param1 in IncludeParams:
            
            """ If ParRange is specified for param1, then use that range; otherwise use ParFac as default setting."""
            if param1 in ParRange.keys(): 
                self.ParRange[param1] = ParRange[param1]
            else:
                if param1 == 'pCa50ref':
                    self.ParRange['pCa50ref'] = (0.8, 1.1)
                else:
                    self.ParRange[param1] = (0.1, ParFac)
            
            if param1 == 'koffon':
                RefValue = Land2017.koffon_ref[WhichDep]
            else:
                RefValue = getattr(Land2017, param1)
            self.ParBounds[param1] = (self.ParRange[param1][0]*RefValue,
                                      self.ParRange[param1][1]*RefValue)
            # Ensure the lower bound is less than the upper bound.
            if self.ParBounds[param1][0] > self.ParBounds[param1][1]:
                self.ParBounds[param1] = (
                    self.ParBounds[param1][1], self.ParBounds[param1][0])
            
        if numsamples == 0:
            PSet0 = [{p1: 1. if p1 not in SetParameters.keys() else SetParameters[p1]  for p1 in AllParams }]
            PSet = [PSet0]
        elif numsamples > 0:
            # if IncludeParams == 'AllParams':
            #     TargetParams = list(AllParams)
            # else:
            #     TargetParams = IncludeParams

            # if 'passive' in ExcludeParams:
            #     ExcludeParams.remove('passive')
            #     ExcludeParams += ['a', 'b', 'k', 'eta_l', 'eta_s']
            # for remove1 in ExcludeParams:
            #     if remove1 in TargetParams:
            #         TargetParams.remove(remove1)

            # self.TargetParams = TargetParams

            numparamsLH = len(self.IncludeParams)

            if WhichEngine == 'LH':
                # RandomSamples = lhsmdu.sample(numtargetparams, numsamples)
                RandomSamples = lhd(
                    np.array([[0., 1.]]*numparamsLH), numsamples).T
            elif WhichEngine == 'Sobol':
                import torch.quasirandom as tq
                SE = tq.SobolEngine(
                    dimension=numparamsLH, scramble=True, seed=8)
                RandomSamples = np.array(SE.draw(numsamples)).T
            # elif WhichEngine == 'regular':
                

            PSet = [None]*numsamples
            for s1 in range(numsamples):
                PSet[s1] = {}
                for p1 in AllParams:                        # initialise all parameter factors to 1 by default, except for the set ones
                    if p1 in SetParameters.keys():
                        PSet[s1][p1] = SetParameters[p1]
                    else:
                        PSet[s1][p1] = 1.
                for ip1, p1 in enumerate(self.IncludeParams):    # change target parameter factors to LH values if not already set
                    PSet[s1][p1] = self.ParRange[p1][0] + \
                        (self.ParRange[p1][1]-self.ParRange[p1][0])*RandomSamples[ip1, s1]
        # return pd.DataFrame(PSet0 + PSet)
        
        self.PSetDF = pd.DataFrame(PSet)        

    
    
    def __init__(self, PSetDF=None, numsamples=0, IncludeParams='AllParams', ExcludeParams=[], SetParameters={},  ParFac=2, ParRange={}, WhichEngine='LH', WhichDep='force'):
        
        if PSetDF is not None: 
            self.PSetDF = PSetDF
        else:
            self.MakePSetLH(numsamples=numsamples, IncludeParams=IncludeParams, ExcludeParams=ExcludeParams, SetParameters=SetParameters, ParFac=ParFac, ParRange=ParRange)
        
            if IncludeParams=='AllParams':
                self.IncludeParams = list(AllParams)
            else:
                self.IncludeParams = IncludeParams
            
            
            
            
        # for param1 in ExcludeParams:
        #     if param1 in self.IncludeParams:
        #         self.IncludeParams.remove(param1)
                
        # self.ParRange = {}
        # self.ParBounds = {}
        # for param1 in AllParams:
        #     if param1 in SetParameters.keys():
        #         self.ParRange[param1] = tuple([SetParameters[param1]]*2)
        #         assert param1 not in IncludeParams, f'Set parameter {param1} cannot be included in cohort randomisation!'
        #         if param1 not in ExcludeParams:
        #             ExcludeParams += [param1]
        #         continue
        #     if param1 == 'pCa50ref':
        #         self.ParRange['pCa50ref'] = (0.8, 1.1)
        #     else:
        #         self.ParRange[param1] = (0.1, self.ParFac)
        #     if param1 == 'koffon':
        #         RefValue = Land2017.koffon_ref[WhichDep]
        #     else:
        #         RefValue = getattr(Land2017, param1)
        #     self.ParBounds[param1] = (self.ParRange[param1][0]*RefValue,
        #                               self.ParRange[param1][1]*RefValue)
        #     # Ensure the lower bound is less than the upper bound.
        #     if self.ParBounds[param1][0] > self.ParBounds[param1][1]:
        #         self.ParBounds[param1] = (
        #             self.ParBounds[param1][1], self.ParBounds[param1][0])
        
        
        self.SimResults = pd.DataFrame()    # Create empty DF for simulation results
        self.Emulators = {}
    
    def Save(self, FileSuffix='suffix'):
        self.Suffix = FileSuffix
        FileName = CohortSaveDir + CohortFileBase + FileSuffix + '.pkl'
        with open(FileName, 'wb') as fpickle:  # Pickling
            pickle.dump(self, fpickle)
        print(f'Saved to {FileName}')

    @classmethod
    def Load(cls, FileSuffix=''):
        FileName = CohortSaveDir + CohortFileBase + FileSuffix + '.pkl'
        with open(FileName, 'rb') as fpickle:   # Unpickling
            return pickle.load(fpickle)
    
    #%%% Generate simulations
    
    def DoFpCa(self, Lambda=1.0):
        Results_list = [[None]*3]*len(self.PSetDF)
        for counter, idx1 in enumerate(self.PSetDF.to_numpy()):
            if counter % 100 == 0: print(f'Doing FpCa for model {counter}')
            PSet1 = pd.Series({param1: idx1[jparam1] for jparam1, param1 in enumerate(self.PSetDF.columns)})
            Model1 = Land2017(PSet1)
            Results_list[counter] = Model1.GetFeat_FpCa_cf(Lambda)
        self.SimResults[[(Lambda, 'Fmax'), (Lambda, 'pCa50'), (Lambda, 'nH')]] = Results_list
        
    def DoQSP(self, QSBasesPCA, Lambda_set=1.0, pCa_set=4.5, fracdSL_set=0.01):
        n_components = QSBasesPCA.n_components
        Results_list = [[None]*(n_components + 1)] * len(self.PSetDF)
        for counter, idx1 in enumerate(self.PSetDF.to_numpy()):
            if counter % 100 == 0: print(f'Doing QSP for model {counter}')
            PSet1 = pd.Series({param1: idx1[jparam1] for jparam1, param1 in enumerate(self.PSetDF.columns)})
            Model1 = Land2017(PSet1)            
            Model1.Lambda_ext = Lambda_set
            try:
                Model1.DoQSP(QSBasesPCA, pCa_set=pCa_set, fracdSL_set=fracdSL_set)
            
                Results_list[counter] = [Model1.Features[f'QSPpc{jcomp}'] for jcomp in range(n_components)]  \
                    + [Model1.ExpResults['QSP_residue']]
                print(f' **************  Model {counter} done successfully !! *****************************')
            except:
                print(f'Problem with model {counter} !!')
                continue 
                
        self.SimResults[[(Lambda_set, f'QSPpc{jcomp}') for jcomp in range(n_components)] + [(Lambda_set, 'QSPresid')]] = Results_list

    def DoExperiments(self, ExpDic, QSBasesPCA=None):
        for Lambda in ExpDic.keys():
            if 'FpCa' in ExpDic[Lambda]:
                self.DoFpCa(Lambda)
            if 'QSP' in ExpDic[Lambda]:
                assert QSBasesPCA is not None, 'Must specify QSBasesPCA !!'
                self.DoQSP(QSBasesPCA, Lambda_set=Lambda, pCa_set=4.5, fracdSL_set=0.01)
        self.SimResults.columns = pd.MultiIndex.from_tuples(self.SimResults.columns, names=['Lambda', 'Feature'])
    
#%% Emulators


def ReduceCohort(Cohort, N):
    import copy
    import random
    Cohort1 = CohortDF() #copy.deepcopy(Cohort)
    Lidx = random.sample(list(Cohort.PSetDF.index), N)
    Cohort1.PSetDF = Cohort.PSetDF.loc[Lidx]
    Cohort1.SimResults = Cohort.SimResults.loc[Lidx]
    return Cohort1

def MakeEmulator(Cohort, WhichFeature, ifLoad=False):
    def Get_best_model_link(link1):
        """
        The file 'best_model.pth' is a link to the best restart of the emulator training. 
        The link is platform-dependent. This 'translates' it so that it can be read by the present system.
        """
        from os import readlink
        Sold = readlink(link1)
        # Snew = home + Sold[(Sold.find('Dropbox')-1):]
        Snew = home + Sold[(Sold.find('GPE')-1):]
        print(Snew)
        return Snew

    # split original dataset in training, validation and testing sets
    Not_None = [j for j, y in enumerate(Cohort.SimResults[WhichFeature]) if y is not None]
    from sklearn.model_selection import train_test_split
    X_, X_test, y_, y_test = train_test_split(
        Cohort.PSetDF[Cohort.IncludeParams].to_numpy()[Not_None],
        np.array(Cohort.SimResults[WhichFeature], dtype=float)[Not_None],
        test_size=0.2,
        random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(
        X_,
        y_,
        test_size=0.2,
        random_state=8)

    
    print(f'Emulating   {WhichFeature}')
    # dataset = Dataset(X_train=self.PSet[self.TargetParams].to_numpy(),
    #                   y_train=self.ReadFeature(WhichFeature))  # ,
    dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
    # X_val=X_val,
    # y_val=y_val,
    # X_test=X_test,
    # y_test=y_test)
    # l_bounds=[self.ParRange[param1][0] for param1 in AllParams],
    # u_bounds=[self.ParRange[param1][1] for param1 in AllParams])

    device = "cpu"

    # self.emulator_config_file = f'{home}/Dropbox/Python/BHFsim/saved_emulators/Cohort_{self.Suffix}_emulator_config_{WhichFeature}.ini'
    emulator_config_file = f'{home}/GPE/saved_emulators/Cohort_{Cohort.Suffix}_emulator_config_{WhichFeature}.ini'
    if not ifLoad:
        likelihood = GaussianLikelihood()
        mean_function = LinearMean(input_size=dataset.input_size)
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dataset.input_size))
        metrics = [MeanSquaredError(), R2Score()]
        experiment = GPExperiment(
            dataset,
            likelihood,
            mean_function,
            kernel,
            n_restarts=3,
            metrics=metrics,
            seed=GPEseed,  # reproducible training
            learn_noise=False  #
        )
        experiment.save_to_config_file(emulator_config_file)

        emulator = GPEmulator(experiment, device)

        from GPErks.train.snapshot import NeverSaveSnapshottingCriterion
        # self.emulator_snapshot_dir = f'{home}/Dropbox/Python/BHFsim/saved_emulators/Cohort_{self.Suffix}_emulator_{WhichFeature}'
        emulator_snapshot_dir = f'{home}/GPE/saved_emulators/Cohort_{Cohort.Suffix}_emulator_{WhichFeature}'
        train_restart_template = "restart_{restart}"
        train_epoch_template = "epoch_{epoch}.pth"
        emulator_snapshot_file = train_epoch_template
        snpc = NeverSaveSnapshottingCriterion(
            f'{emulator_snapshot_dir}/{train_restart_template}', emulator_snapshot_file)

        optimizer = torch.optim.Adam(experiment.model.parameters(), lr=0.1)
        # esc = GLEarlyStoppingCriterion(max_epochs=1000, alpha=0.1, patience=8)

        print(f'Training emulator for {WhichFeature}')
        best_model, best_train_stats = emulator.train(
            optimizer, snapshotting_criterion=snpc)
        print(
            f'   Emulator training completed and saved for {WhichFeature}')

    else:
        from GPErks.gp.experiment import load_experiment_from_config_file
        # Load skeleton
        experiment = load_experiment_from_config_file(
            emulator_config_file,
            dataset  # notice that we still need to provide the dataset used!
        )

        # best_model_file = Get_best_model_link(f'{home}/Dropbox/Python/BHFsim/saved_emulators/emulator_{WhichFeature}/best_model.pth')
        best_model_file = Get_best_model_link(
            f'{home}/GPE/saved_emulators/emulator_{WhichFeature}/best_model.pth')

        best_model_state = torch.load(
            best_model_file, map_location=torch.device(device))

        emulator = GPEmulator(experiment, device)
        emulator.model.load_state_dict(best_model_state)
        print(f'   Emulator made for {WhichFeature}')

    # self.Emulators[WhichFeature] = emulator
    return emulator
    

        
        
        

    #%%% GSA
def DoGSA(Cohort, Emulators, WhichFeatures=[], ifPlot=True, ifSave=False):
    # pass
    GSA_S1 = {}
    GSA = {}
    for Feat1 in WhichFeatures:
        assert Feat1 in Emulators.keys(), f'Emulator missing for {Feat1}'
    if ifPlot:
        if len(WhichFeatures) < 4:
            numrows = 1
            numcols = len(WhichFeatures)
        else:
            numrows = math.floor(np.sqrt(len(WhichFeatures)))
            numcols = math.ceil(len(WhichFeatures)/numrows)
        figS1 = plt.figure(figsize=([14,  9]))
        axS1 = [figS1.add_subplot(numrows, numcols, j+1)
                for j in range(len(WhichFeatures))]

        # if len(WhichFeatures)==1: axS1 = np.array([axS1])
        # axS1 = np.ravel(axS1)
        figCumulS1, axCumulS1 = plt.subplots()

    CumulS1 = np.array([0.]*len(Cohort.IncludeParams))

    tic = time.time()
    X_train = Cohort.PSetDF[Cohort.IncludeParams].to_numpy()
    for iFeat, Feat1 in enumerate(WhichFeatures):
        print(
            f'Starting GSA for {Feat1} ----------------------------  time elapsed = {round(time.time()-tic)}')
        
        dataset = Dataset(X_train=X_train,
                          y_train=Cohort.SimResults[Feat1],
                          l_bounds=[Cohort.ParRange[param1][0]
                                    for param1 in Cohort.IncludeParams],  # AllParams],
                          u_bounds=[Cohort.ParRange[param1][1] for param1 in Cohort.IncludeParams])  # AllParams])
        gsa1 = SobolGSA(dataset, n=128, seed=GPEseed)
        gsa1.estimate_Sobol_indices_with_emulator(
            Emulators[Feat1], n_draws=100)
        print(f'   GSA complete for {Feat1} ----------------------------')
        gsa1.summary()

        if ifPlot == True:
            sns.boxplot(ax=axS1[iFeat], data=gsa1.S1)
            axS1[iFeat].set_title(Feat1)
            if len(WhichFeatures) > 30:
                axS1[iFeat].set_xticks([])
            else:
                axS1[iFeat].set_xticklabels(Cohort.IncludeParams, rotation=90)

        GSA[Feat1] = gsa1
        GSA_S1[Feat1] = gsa1.S1
        CumulS1 += np.mean(gsa1.S1, axis=0)

    if ifPlot:
        figS1.tight_layout()
        axCumulS1.barh(Cohort.IncludeParams, CumulS1, align='center')

    if ifSave:
        import pickle
        with open('GSA_S1.dat', 'wb') as fpickle:
            pickle.dump(GSA_S1, fpickle)

    return (GSA, CumulS1)

    
#%% Calibration
 
# def FitKoffkforce_NOT_WORKING(Lambda_list, Fmax_list_target, pCa50_list_target, nH_list_target, a_set=2.1):    
#     from scipy.optimize import minimize
#     import copy
#     def GetFpCaMetrics(Model1):
#         Fmax_list = []; pCa50_list = []; nH_list = []
#         for Lambda in Lambda_list:
#             Fmax1, pCa501, nH1 = Model1.GetFeat_FpCa_cf(Lambda=Lambda)
#             Fmax_list += [Fmax1]; pCa50_list += [pCa501]; nH_list += [nH1]
#         return (Fmax_list, pCa50_list, nH_list)
#     def ScaleFpCa(Model_test):
#         Fmax_list_test, pCa50_list_test, nH_list_test = GetFpCaMetrics(Model_test)
#         # Cost_Fmax  = lambda x : np.sum( (np.array(Fmax_list_test)*x - np.array(Fmax_list_target))**2 )
#         # Cost_pCa50 = lambda x : np.sum( (10**-np.array(pCa50_list_test)*x - 10**-np.array(pCa50_list_target))**2 )
#         # fac_kPa = minimize(Cost_Fmax,  [1.], method='Nelder-Mead').x[0]
#         # fac_Ca  = minimize(Cost_pCa50, [1.], method='Nelder-Mead').x[0]
        
#         fac_kPa = Fmax_list_target[0] / Fmax_list_test[0]
#         fac_Ca  = 10**-pCa50_list_target[0]/10**-pCa50_list_test[0]
        
#         Model_test.Tref *= fac_kPa
#         Model_test.a *= fac_kPa
#         Model_test.koffon /= fac_kPa
#         Model_test.pCa50ref -= np.log10(fac_Ca)
        
    
#     Model_ref = Land2017(PSet1=pd.Series({'beta0': 0., 'beta1': 0.}), WhichDep='totalforce')
#     Model_ref.Lambda_ext = 1.
#     Model_ref.k1 = 0.841
#     Model_ref.k2 = 1913
#     Model_ref.a = 2.1
#     Model_ref.pCa50ref = 6.66
#     Model_koffon = 0.001396
#     def Cost_Koffkforce(x):
#         Model_test = copy.deepcopy(Model_ref)
#         Model_test.k1 *= x[0]
#         Model_test.koffon *= x[1]
#         ScaleFpCa(Model_test)
#         Fmax_list_test, pCa50_list_test, nH_list_test = GetFpCaMetrics(Model_test)

#         # cost_Fmax =  np.sum( (np.array(Fmax_list_test) - np.array(Fmax_list_target))**2 / np.mean(Fmax_list_target)**2 )
#         # cost_pCa50 = np.sum( (10**-np.array(pCa50_list_test) - 10**-np.array(pCa50_list_target))**2 / np.mean(10**-np.array(pCa50_list_target))**2 )
        
#         cost_Fmax = ((Fmax_list_test[1]-Fmax_list_test[0])/(Fmax_list_target[1]-Fmax_list_target[0]) - 1)**2
#         cost_pCa50 =  ((pCa50_list_test[1]-pCa50_list_test[0])/(pCa50_list_target[1]-pCa50_list_target[0]) - 1 ) **2
#         return cost_Fmax + cost_pCa50
        
#     Koff_best_, koffon_best_ = minimize(Cost_Koffkforce, [1., 1.], method='Nelder-Mead').x
#     print(f'Koff_best_ = {Koff_best_},  koffon_best_ = {koffon_best_}')
    
#     Model_best = copy.deepcopy(Model_ref)
#     Model_best.Lambda_ext = 1.
#     Model_best.k2 /= Koff_best_
#     Model_best.koffon *= koffon_best_
#     ScaleFpCa(Model_best)
#     Fmax_list_best, pCa50_list_best, nH_list_best = GetFpCaMetrics(Model_best)
    
#     fig, ax = plt.subplots(1, 3)
#     for Lambda in Lambda_list:                
#         Model_best.DoFpCa(DLambda = Lambda - Model_best.Lambda_ext)
#         data = Model_best.ExpResults[f'FpCa_{Lambda:.2f}']
#         ax[0].plot(data['pCai'], np.array(data['F'])/1000)
#     ax[0].set_xlabel('pCa')
#     ax[0].invert_xaxis()
#     ax[0].set_ylabel(r'$F_\mathrm{active}$ /kPa')
    
#     ax[1].plot(Lambda_list, np.array(Fmax_list_target)/1000, 'ko:')
#     ax[1].plot(Lambda_list, np.array(Fmax_list_best)/1000)
#     ax[1].set_ylabel(r'$F_\mathrm{max}$ /kPa')
#     ax[1].set_xlabel(r'$\lambda$')
#     ax[2].plot(Lambda_list, pCa50_list_target, 'ko:')
#     ax[2].plot(Lambda_list, pCa50_list_best)
#     ax[2].set_ylabel('pCa50ref')
#     ax[2].set_xlabel(r'$\lambda$')
#     fig.tight_layout()
    
#     return Model_best #Model_best.k1/Model_best.k2, Model_best.koffon


def FitPassive(SL_list, Fpas, b=Land2017.b):
    def cost(x):
        a, SL0 = x
        return np.sum((a*(np.exp(b*(np.array(SL_list)/SL0-1)) - 1) - Fpas)**2)
    return minimize(cost, [2.1, 1.8], method='Nelder-Mead').x

def Carp2PSet(ModelParams):
    if 'k1' not in ModelParams.keys() and 'Koff' not in ModelParams.keys() and 'kforce' not in ModelParams.keys():
        ModelParams['Koff'] = 1000
        ModelParams['k1'] = 10
        ModelParams['kforce'] = 0
    
    return pd.Series(
        {'Tref': ModelParams['Tref'] * 1000 / Land2017.Tref,
        #'trpn50': ModelParams['perm50'] / Land2017.trpn50,
        #'nTm': ModelParams['nperm'] / Land2017.nTm,
        #'ntrpn': ModelParams['TRPN_n'] / Land2017.ntrpn,
        #'k_trpn_on': ModelParams['koff'] *1e3 / Land2017.k_trpn_on,
        #'k_trpn_off': ModelParams['koff'] *1e3 / Land2017.k_trpn_off,
        'rs': ModelParams['dr'] / Land2017.rs,
        'rw': ModelParams['wfrac'] / Land2017.rw,
        #'Aeff': ModelParams['TOT_A'] / Land2017.Aeff,
        #'ku': ModelParams['ktm_unblock'] *1e3 / Land2017.ku,
        'beta0': ModelParams['beta_0'] / Land2017.beta0,
        'beta1': ModelParams['beta_1'] * 1e-6 / Land2017.beta1,
        #'gs': ModelParams['gamma']*1e3 / Land2017.gs,
        #'gw': ModelParams['gamma_wu']*1e3 / Land2017.gw,
        #'phi': ModelParams['phi'] / Land2017.phi,
        'pCa50ref': -np.log10(ModelParams['ca50']*1e-6) / Land2017.pCa50ref,
        #'kws': ModelParams['mu'],
        #'kuw': ModelParams['nu'],
        'a': ModelParams['a']*1000 / Land2017.a,
        'k1': ModelParams['k1'] *1e3/ Land2017.k1,
        'koffon': ModelParams['kforce'] / Land2017.koffon_ref['totalforce']/1000, 
        'k2': ModelParams['k1']*1e3 / ModelParams['Koff']   / Land2017.k2})
        
def FitKoffkforce(Lambda_list, Fmax_list_target, pCa50_list_target, nH_list_target, a_set=Land2017.a/1000, b_set=Land2017.b, SL0_set=Land2017.SL0):
    
    Fmax_target0, Fmax_target1 = Fmax_list_target
    pCa50_target0, pCa50_target1 = pCa50_list_target 

    M0 = Land2017(WhichDep='totalforce')
    M0.beta0 = 0.; M0.beta1 = 0.; M0.a = a_set*1000; M0.Tref = Fmax_list_target[0]*1000; M0.pCa50ref = pCa50_list_target[0]
    M0.k1 = 0.148*1e3; M0.k2 = M0.k1/0.0148; M0.SL0 = SL0_set
    M0.b = b_set
    Fmax_ref0, pCa50_ref0, nH_ref0 = M0.GetFeat_FpCa_cf(Lambda=Lambda_list[0])
    
    def ScaleModel(Model):
        Fmax_prelim0, pCa50_prelim0, nH_prelim0 = Model.GetFeat_FpCa_cf(Lambda=Lambda_list[0]) #1.0)    
        scale_Tref = Fmax_target0/Fmax_prelim0
        scale_ca50 = 10**-pCa50_target0/10**-pCa50_prelim0
        Model.Tref *= scale_Tref
        Model.a *= scale_Tref
        Model.koffon /= scale_Tref
        Model.pCa50ref -= np.log10(scale_ca50)

    def errbetas(x):
        Koff_test_, kforce_test_ = x
        Model_test = copy.deepcopy(M0)
        Model_test.k1 *= Koff_test_
        Model_test.koffon *= kforce_test_
        ScaleModel(Model_test)
        
        Fmax_test0, pCa50_test0, nH_test0 = Model_test.GetFeat_FpCa_cf(Lambda=Lambda_list[0]) #1.0)
        Fmax_test1, pCa50_test1, nH_test1 = Model_test.GetFeat_FpCa_cf(Lambda=Lambda_list[1]) # 1.1)

        # print(f'Fmax_test1: {Fmax_test1:.0f} --> {-Fmax_target1:.0f}')
        # print(f'pCa50_test1: {pCa50_test1:.3f} --> {-pCa50_target1:.3f}' )
        # print(f'a = {ModelParams_test["a"]}')
        # print('')
        
        err_Fmax = ((Fmax_test1-Fmax_test0)/(Fmax_target1-Fmax_target0) - 1 ) **2
        err_ca50 = ((pCa50_test1-pCa50_test0)/(pCa50_target1-pCa50_target0) - 1 ) **2
        return  err_Fmax + err_ca50
      
    def GetFpCaMetrics(Model1):
        Fmax_list = []; pCa50_list = []; nH_list = []
        for Lambda in Lambda_list:
            Fmax1, pCa501, nH1 = Model1.GetFeat_FpCa_cf(Lambda=Lambda)
            Fmax_list += [Fmax1]; pCa50_list += [pCa501]; nH_list += [nH1]
        return (Fmax_list, pCa50_list, nH_list)

    Koff_ref_, kforce_ref_ = minimize(errbetas, [1., 1.], method='Nelder-Mead').x 

    Model_best = copy.deepcopy(M0)
    Model_best.k1 *= Koff_ref_
    Model_best.koffon *= kforce_ref_
    ScaleModel(Model_best)

    Koff_best = Model_best.k1/Model_best.k2
    print('')
    print(f'Koff     = {Koff_best}')
    print(f'kforce   = {Model_best.koffon}')
    print(f'Tref     = {Model_best.Tref}')
    print(f'pCa50ref = {Model_best.pCa50ref}')
    print(f'a        = {Model_best.a}')
    
    Fmax_list_best, pCa50_list_best, nH_list_best = GetFpCaMetrics(Model_best)
    
    fig, ax = plt.subplots(1, 3)
    for Lambda in Lambda_list:                
        Model_best.DoFpCa(DLambda = Lambda - Model_best.Lambda_ext)
        data = Model_best.ExpResults[f'FpCa_{Lambda:.2f}']
        ax[0].plot(data['pCai'], np.array(data['F']), label=rf'$\lambda={Lambda:.2f}$')
    ax[0].set_xlabel('pCa')
    ax[0].invert_xaxis()
    ax[0].set_ylabel(r'$F_\mathrm{active}$ /kPa')
    ax[0].legend(loc=0)
    
    ax[1].plot(Lambda_list, np.array(Fmax_list_target), 'kx:', label='target')
    ax[1].plot(Lambda_list, np.array(Fmax_list_best), 'o-', fillstyle='none', label='fitted')
    ax[1].set_ylabel(r'$F_\mathrm{max}$ /kPa')
    ax[1].set_xlabel(r'$\lambda$')
    ax[2].plot(Lambda_list, pCa50_list_target, 'kx:')
    ax[2].plot(Lambda_list, pCa50_list_best, 'o-', fillstyle='none')
    ax[2].set_ylabel('pCa50ref')
    ax[2].set_xlabel(r'$\lambda$')
    ax[1].legend(loc='center right')
    fig.tight_layout()

    return {'Koff': Koff_best, 'koffon': Model_best.koffon, 'Tref': Model_best.Tref, 'pCa50ref': Model_best.pCa50ref, 'a': Model_best.a}

# %% Main


if __name__ == '__main__':

    # %%% Generate Cohort
    # n = 500  # 5000     ;
    # FileSuffix = f'{n}'

    # Cohort = Land2017Cohort(n=n, ExcludeParams=['passive'], ifFilter=False, WhichEngine='Sobol')
    # Cohort.DoAllExperiments(ifPlot=False)
    # Cohort.CleanBad()
    # Cohort.Save(FileSuffix=FileSuffix)
    # Cohort = Land2017Cohort.Load(FileSuffix=FileSuffix)

    # %%% Define LenBasisPCA

    # n_components = 5

    # Choose one of these:
    # import GetKYdata_0;  GetKYdata_0.MakeLenBasis(n_components=n_components)        #  Faruk's data   (HUMAN)
    # import KYdata_20220321;   LenBasisPCA = KYdata_20220321.MakeLenBasis(n_components = n_components)      # Greg's data  (RAT)
    # LenBasisPCA = Cohort.MakeLenBasis(n_components=n_components)

    # with open(CohortSaveDir + 'LenComponents.dat', 'wb') as fpickle:
    #     pickle.dump(LenBasisPCA, fpickle)

    # with open(CohortSaveDir + 'LenComponents.dat', 'rb') as fpickle:
    #     LenBasisPCA = pickle.load(fpickle)

    # t = LenBasisPCA.DF.columns

    # PlotLenBasis(LenBasisPCA)

    # %%% Get features

    # Cohort.GetAllFeatures(n_components=n_components)

    # fig = plt.figure()
    # jModel = 100
    # plt.plot(Cohort.Models[jModel].ExpResults['aLP0+']['Fa'] / Cohort.Models[jModel].ExpResults['aLP0+']['Fa0'] - 1)
    # y = np.array([Cohort.Models[jModel].Features[f'aLP0+_n{n}'] for n in range(5)]) @ LenBasisPCA.components_
    # plt.plot(y + LenBasisPCA.mean_)

    # for j in range(4):
    #     Exp1 = Cohort.Models[0].ExpResults[f'aLP{j}+']
    #     print(Exp1['pCa'], Exp1['dLambda'])
    #     plt.plot(Exp1['t'], Exp1['F'])
    #     Exp1 = Cohort.Models[0].ExpResults[f'aLP{j}-']
    #     print(Exp1['pCa'], Exp1['dLambda'])
    #     plt.plot(Exp1['t'], Exp1['F'])

    # Test PC decomposition of lengthening traces, determined by Cohort.GetAllFeatures
    # Cohort.DisplayLengtheningPCA(iModel=0, ActiveOrPassive='active', PoM='+', iPcl = 0)

    # %%% Do PCA

    # Cohort.pca, Y = Cohort.DoPCA(MaxComponents=10, ifPlot=True)

    # Cohort.MakeAllEmulators_pca()
    # GSA, CumulS1 = Cohort.DoGSA_PC()

    # %%% Test model filter
    # jFilter = [] #Cohort.Filter()
    # print(f'{len(jFilter)} models pass the filter: {jFilter}')
    # # Cohort.Show_results_FpCa(WhichModels=jFilter, Lambda=1.1)
    # Cohort.Show_results_Lengthening(WhichModels=jFilter, Pcl=LengtheningProtocols['active'])

    # ShowDistances(Cohort.Models[0], Cohort, jFilter)

    # %%% Make Cohort emulators

    # Cohort.MakeAllEmulators([])
    # Cohort.Save(FileSuffix=f'{n}e')
    # Cohort = Land2017Cohort.Load(FileSuffix=f'{n}e')

    # Cohort.TestEmulators(0)

    # %%%% Test emulators
    # for jModel in range(4):
    #     Cohort.Models[jModel].ShowExperiments(Cohort.Emulators, Cohort.TargetParams, Cohort.PCAcomponents, Cohort.PCAmean)

    # count = 0
    # while count < 4:
    #     PSet1 = Cohort.MakeParamSet(numsamples=1, IncludeParams='AllParams', ExcludeParams='passive').loc[1]
    #     Model1 = Land2017(PSet1)
    #     try:
    #         Model1.DoAllExperiments()
    #     except:
    #         print('skip model')
    #         continue
    #     for Lambda1 in [1.1, 1.2]:
    #         Model1.GetFeat_FpCa(Lambda=Lambda1)
    #     for jPcl in range(len(LenPcl['active'])):
    #         for PoM in ['+', '-']:
    #             Model1.GetFeat_Lengthening(Cohort.PCAcomponents, Cohort.PCAmean, jPcl, PoM)
    #     Model1.ShowExperiments(Cohort.Emulators, Cohort.TargetParams, Cohort.PCAcomponents, Cohort.PCAmean)
    #     count += 1

    # Cohort.TestEmulators(Cohort.Models[0])
    # Cohort.TestEmulators_lengthening(Cohort.Models[0])

    # %%% Generate cohort for filtering

    # print('Start filtering =================================================')
    # PSet_filt = Cohort.MakeParamSet(numsamples=1000, WhichEngine='Sobol')
    # XX = np.array([list(PSet1.values()) for PSet1 in PSet_filt])

    # FilteredOK = [jX for jX in range(len(XX)) if Cohort.Filter(XX[jX,:,None].T, Cohort.Models[0])]
    # print(f'Filter :  {len(FilteredOK)} good out of {len(XX)}.')

    # Cohort2 = Land2017Cohort(X=XX[FilteredOK])
    # Cohort2.DoAllExperiments()
    # Cohort2.PlotLengthening(ActiveOrPassive='active', PoM='+', ifNormalise=False)

    # %%% Test the PCA for dimensionality reduction of stretch/destretch experiments.
    # # # # pca, Y_pca = Cohort.Test_SD_PC(ActiveOrPassive='active', n_components=10)
    # # # # pca, Y_pca = Cohort.Test_SD_PC(ActiveOrPassive='passive', n_components=10)

    # %%% Test emulators
    # WhichFeature = 'nH_1.1'   #'EC50_1.1' #
    # print(Cohort.Models[0].Features[WhichFeature])
    # print(Cohort.Emulators[WhichFeature].predict(Cohort.Models[0].X))
    # Cohort.TestEmulators(Cohort.Models[2])  # compares all the actual features with the predictions

    # %%% Do GSA
    # GSA, ParamSummary = Cohort.DoGSA(ifPlot=True, ifSave=True)

    # with open('GSA_S1.dat', 'rb') as fpickle:
    #     GSA_S1 = pickle.load(fpickle)

    # WhichFeatures = list(Cohort.Emulators.keys())
    # for i1, Par1 in enumerate(Cohort.TargetParams):
    #     S1 = [np.mean(GSA_S1[Feat1], axis=0)[i1] for Feat1 in WhichFeatures]
    #     jMainS1 = list(reversed(np.argsort(S1)))[0:5]
    #     print(Par1)
    #     for j in jMainS1:
    #         print(f'     {WhichFeatures[j]:20s}     :    {S1[j]/np.sum(S1):.2f}')
    #     print('\n')

    # %%% Display cohort results. -------------------------------
    # Cohort.PlotFpCa(DLambda=0.0, ifHistograms=True)
    # Cohort.PlotFpCa(DLambda=0.1, ifHistograms=True)
    # Cohort.PlotStretchDestretch(dLambda=0.001)
    # Cohort.PlotStretchDestretch_passive(dLambda=0.001)
    # Cohort.PlotCaiStep()

    # %%% Check if Cai step response is described by several exponentials
    # Fit_fun = lambda x, a, b, n,c : a + b* x**n/(x**n+c**n)
    # fig, ax = plt.subplots(nrows=2)
    # Fits = [None]*len(Cohort.Models)
    # for i1, Model1 in enumerate(Cohort.Models):
    #     t = Model1.ExpResults['CaiStep_7.0_5.0']['t']
    #     y = Model1.ExpResults['CaiStep_7.0_5.0']['F']
    #     Fits[i1], cov = curve_fit(Fit_fun, t, y,
    #                           p0=(y[0],
    #                               y[-1]-y[0],
    #                               next(t[i] for i in range(len(y)) if y[-1]-y[i]< (y[-1]-y[0])/2),
    #                               1
    #                               ),
    #                           maxfev=10000)
    #     print(f'done {i1}')
    #     ax[0].plot(t, y)
    # ax[0].set_prop_cycle(None)
    # for i1, Model1 in enumerate(Cohort.Models):
    #     y = Model1.ExpResults['CaiStep_7.0_5.0']['F']
    #     yfit = Fit_fun(t, *Fits[i1])
    #     ax[0].plot(t, yfit, ':')
    #     ax[1].plot(t, y-yfit)

    #%%% Test GetKoffkforce
    
    #### Set 1
    # Lambda_list = np.array([1.05, 1.15]) #np.array([1.0, 1.1])
    # Fmax_list_target = np.array([170., 210.]) 
    # pCa50_list_target = np.array([6.34, 6.56])
    # nH_list_target = [5.14, 5.14]
    # Model_best = FitKoffkforce(Lambda_list, Fmax_list_target, pCa50_list_target, nH_list_target, a_set=2.0)
    
    #### Set 2   (from AL_20230125.py)
    # Lambda_list = np.array([1.0, 1.1]) 
    # Fmax_list_target = np.array([36.841, 42.416]) 
    # pCa50_list_target = np.array([5.763, 5.7796])
    # nH_list_target = [2.980, 3.003]
    # Model_best = FitKoffkforce(Lambda_list, Fmax_list_target, pCa50_list_target, nH_list_target, a_set=0.5)
    
    #### Set 3   (from AL_20230125.py)
    Lambda_list = np.array([1.0, 1.2]) 
    Fmax_list_target = np.array([36.841, 52.624]) 
    pCa50_list_target = np.array([5.763, 5.844])
    nH_list_target = [2.980, 3.044]
    Model_best = FitKoffkforce(Lambda_list, Fmax_list_target, pCa50_list_target, nH_list_target, a_set=0.4)