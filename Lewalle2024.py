
# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd



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
             'gw',
             'phi',
             'Aeff',
             'beta0',
             'beta1',
             'Tref',
             'k2',
             'k1',
             'koffon')



def HillFn(x, ymax, n, ca50):
    return ymax * x**n/(x**n + ca50**n)


# %% Model definition

class Lewalle2024:

    # Defaults
    SL0 = 1.8               # um
    pCai = lambda self, t: 4.5
    
    # Passive tension parameters
    a = 241                 # Pa
    b = 9.1                 # dimensionless
    k = 8.86                # dimensionless
    eta_l = 0.2             # s
    eta_s = 20e-3           # s

    # Active tension parameters
    k_trpn_on =  0.1e3      # s-1
    k_trpn_off = 0.1e3      # s-1
    ntrpn = 2.58              # dimensionless
    pCa50ref = 5.25         # M
    ku = 1000               # s-1
    nTm = 2.2               # dimensionless
    trpn50 = 0.35           # dimensionless
    kuw = 4.98              # s-1
    kws = 19.10             # s-1
    rw = 0.5                # dimensionless
    rs = 0.25               # dimensionless
    gs = 42.1               # s-1 
    es = 1.0                # Controls asymmetry in gsu between pos and neg quick stretches ( Must be in [0,1])
    gw = 28.3               # s-1 
    phi = 0.1498            # dimensionless
    Aeff = 125              # dimensionless
    beta0 = 0.0             # dimensionless
    beta1 = 0.0             # dimensionless
    Tref = 23.e3            # Pa

    # OFF-state-specific parameters
    k1 = 0.877    # s-1  
    k2 = 12.6     # s-1 
        
    ra = 1.  # Residual rate inserted for mavacamten simulation (6/4/23)
    rb = 1.  # Residual rate inserted for mavacamten simulation (6/4/23)

    koffon = None       # Defined during initialisation, depending on WhichDep (='force' or 'totalforce', etc.)
    Dep_k1ork2 = None   # Set to either 'k1' or 'k2'
    koffon_ref = {'force':          None,     # Pa-1
                  'totalforce':     0.00144064,     # Pa-1
                  'passiveforce':   None,     # Pa-1
                  'Lambda': None,                     # dimensionless
                  'C': None}                          # dimensionless

      
    Lambda_ext = 1.0   # Default externally applied elongation factor

    def kwu(self):
        return self.kuw * (1/self.rw - 1) - self.kws 	                          # eq. 23 in Land2017

    def ksu(self):
        return self.kws*self.rw*(1/self.rs - 1)                     # eq. 24 in Land2017

    def kb(self):
        return self.ku*self.trpn50**self.nTm / (1-self.rs-(1-self.rs)*self.rw)    # eq. 25 in Land2017 (mind typo: reads trpn50 not CaTRPN)


    def dLambdadt_fun(self, t):
        return 0    # This gets specified by particular experiments


    def Aw(self):
        return self.Aeff * self.rs/((1-self.rs)*self.rw + self.rs) 		# eq. 26 in Land2017

    def As(self):
        return self.Aw()        # as assumed in Land2017

    def __init__(self, PSet1=None, WhichDep='totalforce', Dep_k1ork2='k1'): #, KOFF=0.017679504554596895):  
        """
        Model initialisation.
        
        Parameters
        ----------
        PSet1 : pandas series, optional
            Specifies factors by which the reference parameters are multiplied.
            e.g., PSet1 = pd.Series({'Tref': 1.1, 'pCa50ref':1.01}) increases Tref by 10%, pCa50ref by 1% etc.
        WhichDep : string, optional
            Specifies the feedback 'paradigm': 'totalforce', 'force', 'passiveforce', 'C'
        Dep_k1ork2 : string, optional
            Specifies whether feedback is applied to k1 or k2:
            'k1':  k1 -> k1 * (1 + kfeedback*X)
            'k2':  k2 -> k2 / (1 + kfeedback*X)
        """
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
                setattr(self, param, getattr(Lewalle2024, param) * PSet1[param])

        self.ExpResults = {}    # Initialise experimental results
        self.Features = {}

    def pCa50(self, Lambda):
        Ca50ref = 10**-self.pCa50ref
        Ca50 = Ca50ref + self.beta1*(np.minimum(Lambda, 1.2)-1)
        if np.size(Ca50) > 1:
            if any(np.array(Ca50)<0):
                for j in range(len(Ca50)):
                    if Ca50[j] <=0:
                        Ca50[j] = np.nan
        return -np.log10(Ca50)


    def h(self, Lambda=None):
        """        Sets force-magnitude LDA in Land2017 (beta0, beta1 != 0)        """
        if Lambda is None:
            Lambda = self.Lambda_ext
        def hh(Lambda):
            return 1 + self.beta0*(Lambda + np.minimum(Lambda, 0.87) - 1.87)
        return np.maximum(0, hh(np.minimum(Lambda, 1.2)))

    def Ta(self, Y):
        """         Total active tension        """
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref/self.rs * (S*(Zs+1) + W*Zw)

    def Ta_S(self, Y):
        """         Active tension contribution from S state        """
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref/self.rs * (S*(Zs+1))

    def Ta_W(self, Y):
        """         Active tension contribution from W state        """
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        return self.h(Lambda) * self.Tref/self.rs * (W*Zw)

    def F1(self, Y):
        """         Passive spring element - eq. 3 in Land2017        """
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        C = Lambda-1
        return self.a*(np.exp(self.b*C)-1)

    def F2(self, Y):
        """         Passive spring element - eq. 4 in Land2017        """
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.transpose()
        C = Lambda-1
        return self.a*self.k*(C - Cd)

    def Ttotal(self, Y):
        """         Total tension (active + passive)        """
        return self.Ta(Y) + self.F1(Y) + self.F2(Y)
    
    def Tp(self, Y):
        """         Total passive tension        """
        return self.F1(Y) + self.F2(Y)
        
    def Ta_ss(self, pCai=None):
        """         Active tension (steady-state)        """        
        U_ss = self.U_ss(pCai)
        return self.h(self.Lambda_ext) * self.Tref/self.rs * self.kws*self.kuw/self.ksu()/(self.kwu()+self.kws) * U_ss
    
    def U_ss(self, pCai=None):
        """         State U population (steady-state)        """
        if pCai is None:
            pCai = self.pCai(0)
        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        U_ss = 1.0 - UE_ss - B_ss - BE_ss - W_ss - S_ss
        return U_ss
        
    def Kub(self, pCai=None):
        """         Unblocked/blocked eqb constant  (eq. 10 in Land2017)        """
        CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss = self.Get_ss(pCai)
        Kub = self.ku/self.kb() * CaTRPN_ss**(+self.nTm)
        return Kub
    
    def Tp_ss(self, Lambda=None):
        """         Passive tension (steady state)        """
        if Lambda is None:
            Lambda = self.Lambda_ext
        return self.a * (np.exp(self.b*(Lambda-1)) - 1)

    def Get_ss(self, pCai=None):
        """         Calculate steady-state state values.        """
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

            """Solve quadratic for U_ss:            """
            if self.WhichDep in ['totalforce', 'passiveforce']:
                kfFp = self.koffon*self.Tp_ss()   # Passive force contribution to feedback
            elif self.WhichDep in ['force', 'bound']:
                kfFp = 0   # Passive force term not existent for these scenarios
            
         
            aa = self.ra * mu*(1+1/Kub+Q)
            bb = 1/KE*(1+1/Kub) - self.ra*mu + (1+1/Kub+Q)*self.ra*(self.rb+kfFp)
            cc = -self.ra*(self.rb+kfFp)

            # When this value is small, Taylor-expand the quadratic to avoid numerical error (subtraction of almost-identical large numbers)
            SmallUCriterion = -4*aa*cc/bb**2  
            if SmallUCriterion > 1e-3:
                U_ss =  (-bb + np.sqrt(bb**2 - 4*aa*cc))/2/aa
            else:
                U_ss =  self.ra*(self.rb+kfFp)/bb * (1 - mu*self.ra**2*(1+1/Kub+Q)*(self.rb+kfFp)/bb**2)
                    
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

        return np.array([CaTRPN_ss, B_ss, S_ss, W_ss, Zs_ss, Zw_ss, Lambda_ss, Cd_ss, BE_ss, UE_ss], dtype=float)

    # %% ODE system

    def gwu(self, Y):
        """ Distortion decay rate W->U """
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        return self.gw * abs(Zw)      # eq. 16 in Land2017

    def gsu(self, Y):
        """ Distortion decay rate S->U """
        # eq. 17 in Land2017
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y
        if Zs < -self.es:
            return -self.gs * (Zs + self.es)
        elif Zs >0:
            return self.gs * Zs
        else:
            return 0
        

    def cw(self, Y=None):
        return self.phi* self.kuw *((1-self.rs)*(1-self.rw))/((1-self.rs)*self.rw)      # eq. 27 in Land2017
    def cs(self, Y=None):
        return self.phi * self.kws *((1-self.rs)*self.rw)/self.rs                       # eq. 28 in Land2017
    
    # ===========================================

    def U(self, Y):
        CaTRPN, B, S, W, Zs, Zw, Lambda, Cd, BE, UE = Y.T
        return 1.-B-S-W - BE-UE

    def k1_fb(self, Y):
        """ Net k1, with feedback """
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
    
    def k2_fb(self, Y):
        """ Net k2, with feedback """
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
        dCaTRPNdt = self.k_trpn_on * \
            (10**-self.pCai(t)/10**-self.pCa50(Lambda))**self.ntrpn * \
            (1-CaTRPN) - self.k_trpn_off*CaTRPN
        dBdt = self.kb()*CaTRPN**(-self.nTm/2)*self.U(Y)  \
            - self.ku*CaTRPN**(self.nTm/2)*B  \
            - self.k2_fb(Y)*B   \
            + self.k1_fb(Y) * BE  # eq.10 in Land2017, amended to include myosin off state dynamics
        dWdt = self.kuw*self.U(Y) - self.kwu()*W - \
            self.kws*W - self.gwu(Y)*W     # eq. 12
        dSdt = self.kws*W - self.ksu()*S - self.gsu(Y)*S        # eq. 13

        # OFF-state specific equations 
        dBEdt = self.kb()*CaTRPN**(-self.nTm/2)*UE  \
            - self.ku*CaTRPN**(self.nTm/2)*BE  \
            + self.k2_fb(Y)*B \
            - self.k1_fb(Y) * BE
        dUEdt = -self.kb()*CaTRPN**(-self.nTm/2)*UE  \
            + self.ku*CaTRPN**(self.nTm/2)*BE  \
            + self.k2_fb(Y)*self.U(Y) \
            - self.k1_fb(Y) * UE

        # This function to be user-defined for particular experiments
        dLambdadt = self.dLambdadt_fun(t)
        if Lambda-1-Cd > 0:     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    def dYdt_pas(self, Y, t):
        """ ODEs for passive system only."""
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

        # This function to be user-defined for particular experiments
        dLambdadt = self.dLambdadt_fun(t)
        if Lambda-1-Cd > 0:     # i.e., dCd/dt>0    (from eq. 5)
            dCddt = self.k/self.eta_l * (Lambda-1-Cd)     # eq. 5
        else:
            dCddt = self.k/self.eta_s * (Lambda-1-Cd)     # eq. 5

        return (dCaTRPNdt, dBdt, dSdt, dWdt, dZsdt, dZwdt, dLambdadt, dCddt, dBEdt, dUEdt)

    # %% Experiments




    def DoFpCa(self, DLambda=0.0, pCai_limits=[6.5, 5.0], ifPlot=False):
        """ Simulate steady-state F-pCa relationship """
        pCai_original = self.pCai
        self.Lambda_ext += DLambda     # Remember to reset below!
        pCai_array = np.linspace(pCai_limits[1], pCai_limits[0], 50)
        F_array = np.array([None]*len(pCai_array))

        F_array = self.Ta_ss(pCai_array)
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
            ax_FpCa.semilogx(10**-pCai_array, F_array /1000)
            ax_FpCa.set_xlabel('Cai (M)')
            ax_FpCa.set_ylabel('F')
            ax_FpCa.set_title(f'F-pCa, SL={self.SL0*self.Lambda_ext}')

        print(f'   -done FpCa (Lambda={self.Lambda_ext:.2f})')
        self.Lambda_ext -= DLambda      # Reset to initial Lambda

    def DoDynamic(self, dLambdadt_imposed, 
                  t, 
                  DLambda_init = 0.,   # Initial imposed step offset to Lambda
                  ifPlot=False):
        """ Determine the dynamic response to an imposed length change dLambdadt_imposed.
        dLambdadt_imposed is a user-specified function of time."""
        
        self.dLambdadt_fun = dLambdadt_imposed
        
        # Determine steady-state conditions
        CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0, Cd_0, BE_0, UE_0 = self.Get_ss()
        Ta_ss_init = self.Ta_ss()
        
        # Apply step change:        
        Zs_0 += self.As()*DLambda_init
        Zw_0 += self.Aw()*DLambda_init
        self.Lambda_ext = Lambda_0 + DLambda_init
        Y0 = [CaTRPN_0, B_0, S_0, W_0, Zs_0, Zw_0, Lambda_0 + DLambda_init, Cd_0, BE_0, UE_0]
       
        Ysol = odeint(self.dYdt, Y0, t)
        Tasol = self.Ta(Ysol)          
        
        self.ExpResults['Dynamic'] = {'t': t, 'Tasol': Tasol, 'Ysol': Ysol}
        
        if ifPlot:
            fig_sol, ax_sol = plt.subplots(nrows=2)
            ax_sol[0].plot(t, Ysol[:, 6])  
            ax_sol[0].plot([0., 0.], [Lambda_0, Ysol[0,6]], 'limegreen')
            ax_sol[0].set_ylabel(r'$\lambda_\mathrm{imposed}$')
            ax_sol[1].plot(t, Tasol /1000)
            ax_sol[1].plot([0., 0.], [Ta_ss_init/1000, Tasol[0]/1000], 'limegreen')
            ax_sol[1].set_ylabel(r'$T_a$  /kPa')
            fig_sol.suptitle('Imposed dynamic length variation')


    def DoNyquist(self, fmin=0.3, fmax=100, Numf=20, ifPlot=False, ifPlot_allf=False):

        f_list = np.logspace(np.log10(fmin), np.log10(fmax), Numf)

        Stiffness_f = [None]*len(f_list)
        DphaseTa_f = [None]*len(f_list)

        for ifreq, freq in enumerate(f_list):
            print(f'Doing f{ifreq} = {freq}')

            Tasol, Ysol, t, Stiffness, DphaseTa = self.SinResponse(freq, ifPlot=ifPlot_allf)

            Stiffness_f[ifreq] = Stiffness
            DphaseTa_f[ifreq] = DphaseTa

        if ifPlot:
            fig, ax = plt.subplots(
                ncols=3, nrows=1, num='Dynamic experiments', figsize=(15, 7))
            Stiffness_f = np.array(Stiffness_f)
            DphaseTa_f = np.array(DphaseTa_f)
            ax[0].semilogx(f_list, Stiffness_f /1000, '-')
            ax[1].semilogx(f_list, DphaseTa_f, '-')
            ax[2].plot(Stiffness_f * np.cos(DphaseTa_f) /1000,
                       Stiffness_f * np.sin(DphaseTa_f) /1000)
            ax[2].set_aspect('equal', adjustable='box')
            ax[0].set_xlabel('f /Hz')
            ax[1].set_xlabel('f /Hz')
            ax[0].set_ylabel('|Z| /kPa')
            ax[1].set_ylabel('phase /rad')
            ax[2].set_xlabel('Re(Z) /kPa')
            ax[2].set_ylabel('Im(Z) /kPa')



    def SinResponse(self, freq, numcycles=10, pointspercycle=30, dLambda_amplitude=0.0001, ifPlot=False):
        from scipy.optimize import curve_fit
        self.dLambdadt_fun = lambda t: \
            dLambda_amplitude * np.cos(2*np.pi*freq*t) * 2*np.pi*freq

        t = np.linspace(0, numcycles/freq, numcycles*pointspercycle)
        self.DoDynamic(lambda t: dLambda_amplitude * np.cos(2*np.pi*freq*t) * 2*np.pi*freq,
                       t=t)
        
        Ysol = self.ExpResults['Dynamic']['Ysol']
        Tasol = self.ExpResults['Dynamic']['Tasol']        

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
            ax_sol[1].plot(t, Tasol /1000)
            ax_sol[1].plot(t, Sin_fun(t, *SinFit)/1000, 'k--')
            ax_sol[1].set_ylabel(r'$T_a$ /kPa')
            ax_sol[0].plot(t, Ysol[:, 6])
            ax_sol[0].set_ylabel(r'$\lambda_\mathrm{imposed}$')
            fig_sol.suptitle(f'f = {freq}')

        return Tasol, Ysol, t, Stiffness, DphaseTa



    
        
    # %% Extract features


    def GetFeat_FpCa(self, Lambda=1.1, ifPrint=False):
        """ Extract Fmax, pCa50, and nH for the F-pCa relationship. """
        
        from scipy.optimize import minimize
        
        Lambda_initial = self.Lambda_ext    # to reset later
        self.Lambda_ext = Lambda
        
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



#%% Main

if __name__ == '__main__':
    pass
    
    Model1 = Lewalle2024()
    

    #%%% Test FpCa LDA    

    
    fig1, ax1 = plt.subplots()    
    pCa_list = np.linspace(6.5, 4.5, 50)
    for jLambda, Lambda1 in enumerate([1.05, 1.25]):
        Model1.Lambda_ext = Lambda1
        Ftotal1 = Model1.Tp_ss(Lambda1) + Model1.Ta_ss(pCa_list)
        ax1.plot(pCa_list, Ftotal1 /1000, label = rf'SL = {Model1.SL0*Lambda1:.1f} $\mu m$')

    ax1.set_title('Steady state F-pCa relationship')
    ax1.set_xlabel('pCa')
    ax1.set_ylabel(r'$F_\mathrm{total}$ /kPa')
    ax1.invert_xaxis()
    ax1.legend(loc=0)

    # # Alternatively:             
    # Model1.Lambda_ext = 1.0
    # Model1.DoFpCa(ifPlot=True)
    
    
    #%%% Test sine response
    
    Model1 = Lewalle2024()
    Model1.Lambda_ext = 1.0
    
    freq = 10 # Hz
    def dLambdadt_imposed(t):
        amplitude = 0.0001
        return amplitude * np.cos(2*np.pi*freq*t) * 2*np.pi*freq
    numcycles = 20 
    pointspercycle = 30
    Model1.DoDynamic(dLambdadt_imposed = dLambdadt_imposed, 
                      t = np.linspace(0, numcycles/freq, numcycles*pointspercycle),
                      ifPlot=True)
    
    #%%% Test step length change
    
    Model1 = Lewalle2024()
    Model1.Lambda_ext = 1.0
    
    Model1.DoDynamic(dLambdadt_imposed = lambda t: 0,    # length is constant after step change
                      t = np.linspace(0, 10, 1000),
                      DLambda_init = 0.01,   # imposed step change magnitude
                      ifPlot=True)
    
    