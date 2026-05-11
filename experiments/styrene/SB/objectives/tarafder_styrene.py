#tarafder_styrene.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tarafder (2005) optimized the styrene reactor using the pseudo‑homogeneous PFR model (Sheel–Crowe) with Ergun for pressure drop and ideal‑gas thermodynamics/transport

In Tarafder–Rangaiah–Ray (2005) case 1 refers to bi‑objective optimization of the reactor section (maximize styrene flow rate and selectivity)
Process scope: Reactor + EB pre‑heater (HE1) only, Three reactor
configurations are compared: SB (single bed), SI (steam
injection mid‑bed), and DB (double bed with inter‑bed reheat).

NSGA‑II settings used by Tarafder :
100 generations, 80 population, real‑coded operators.
Recommended parameters: pc=0.7, pm=0.05, ηc=10, ηm=20
===========================================================================

SOURCES (shorthand used below in docstrings/comments)
-----------------------------------------------------
[Yee2003]  Yee et al., Chem. Eng. Commun. (CCE) 27, 2003 — Appendix A (Table A1 kinetics,
           Table A2 plant/bed data; model equations & notes incl. K_eq for r1).
[Tarafder2005] Tarafder, Rangaiah, Ray, Chem. Eng. Sci. 60, 2005 — SB/SI/DB configs,
           decision-variable bounds, constraints, and Pareto figures (Figs. 1–3).
[E&E1994] Elnashaie & Elshishini, "Modelling, Simulation, and Optimization of Industrial
           Fixed-Bed Catalytic Processes", 1994 — Ch. 6, Tables 6.24–6.26 (ΔH_i(T),
           Cp(T) correlations for organics/inorganics); Ergun usage context.
[Abdalla1994] Abdalla et al., Appl. Catal. A 113 (1994) — intrinsic kinetics background
           (lab/heterogeneous extraction) referenced for optional kinetics switch.

Notes on units: R = 8.314 kJ/(kmol·K); pressures in bar inside rate laws; molar flows in kmol/h.
"""



from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable
import math
import numpy as np

# -----------------------------------------------------------------------------
# 0) Global constants and species map
# -----------------------------------------------------------------------------

R = 8.314           # kJ/(kmol·K) — standard gas constant for kJ-based energy balance
BAR_TO_PA = 1e5
HOUR_TO_SEC = 3600.0


# Species set mirrors industrial EB→ST system used by [Yee2003, Appx A] / [Tarafder2005]
SPECIES = [
    'EB', 'ST', 'H2', 'BZ', 'ETH', 'TOL', 'CH4', 'CO', 'CO2', 'H2O'
]
IDX = {s: i for i, s in enumerate(SPECIES)}

# Molecular weights (kg/kmol). Same component list as [Tarafder2005, page 348, reactions 1-6].
MW = np.array([
    106.167,  # EB: Ethyl Benzene (C6H5CH2CH3)
    104.149,  # ST: styrene (C8H8):C6H5CHCH2
    2.016,    # H2
    78.113,   # BZ:C₆H₆
    28.054,   # ETH:Ethylene (C₂H₄)
    92.141,   # TOL: Toluene (C₇H₈): C6H5CH3
    16.043,   # CH4
    28.010,   # CO
    44.009,   # CO2
    18.015,   # H2O
], dtype=float)

# -----------------------------------------------------------------------------
# 1) Thermochemistry (Elnashaie & Elshishini, 1994; Tables 6.24–6.26)
# -----------------------------------------------------------------------------
# Cp correlations: Cp_j(T) [kJ/kmol-K]; T in K.
# - Organics (Tab. 6.25): Cp = a + b T + c/T^2 (6.215 PAGE 370), with column header shown as
#   We use c = -γ×10^3 with the equation above.
# [E&E1994, Table 6.26] (inorganics): Cp = a + b T + c/T^2 (6.215 PAGE 370) with c printed as c×10^-5.
_CP_ORG = {
    'EB' : dict(a=-6.89,  b=0.0620, c= -3.72e3),
    'ST' : dict(a=-11.02, b=0.0750, c= -5.69e3),
    'BZ' : dict(a=-1.71,  b=0.325, c=-11.10e3),
    'TOL': dict(a= 2.41,  b=0.392, c=-13.10e3),
    'ETH': dict(a=11.85,  b=0.120, c= -3.65e3),
    'CH4': dict(a=14.16,  b=0.076, c= -1.80e3),
}

# - Inorganics (Tab. 6.26): Cp = a + b T + c/T^2; **c is listed as c×10^-5**.
_CP_INORG = {
    'H2O': dict(a=30.57, b=0.0103,     c=0.0),       # steam: c not = - in table 6.26 which given → 0
    'H2' : dict(a=27.30, b=0.00327,    c=5.02e-5),
    'CO' : dict(a=28.43, b=0.00410,    c=4.61e-5),
    'CO2': dict(a=44.26, b=0.00879,    c=86.30e-5),
}
# -----------------------------------------------------------------------------
def Cp_species(T: float, species: str) -> float:
    """Return Cp_j(T) [kJ/kmol-K] from book correlations."""
    s = species.upper()
    if s in _CP_ORG:
        co = _CP_ORG[s]
        return float(co['a'] + co['b']*T + co['c']/(T*T)) #Cp = a + b T + c/T^2 (6.215 PAGE 370)
    if s in _CP_INORG:
        ci = _CP_INORG[s]
        return float(ci['a'] + ci['b']*T + ci['c']/(T*T)) #Cp = a + b T + c/T^2 (6.215 PAGE 370)
    raise KeyError(f"No Cp coefficients for '{species}'")


# -----------------------------------------------------------------------------

    """
    Mole-fraction weighted mixture Cp(T) [kJ/kmol-K].
    Used in energy balance as ∑F_j Cp_j(T). See [Yee2003, Appx A, energy eqn].
    """
#calculates the specific heat capacity (C_p) of a gas mixture at a given temperature.The calculation is based on a mole-fraction weighted average of the individual heat capacities of the species in the mixture.
def mixture_cp(F: np.ndarray, T: float) -> float: #This array holds the molar flow rates of each chemical species in the mixture
#T: float: This is a parameter named T, a floating-point number representing the temperature of the mixture (e.g., in Kelvin).

    Ft = float(F.sum()) #Calculate Total Molar Flow
    if Ft <= 0.0:#This is a safety check. If the total molar flow Ft is zero or negative (which is physically unlikely but could occur in a simulation's initial state), the code cannot proceed to calculate mole fractions because it would cause a division-by-zero error.
        return float(np.mean([Cp_species(T, s) for s in SPECIES])) #In this case, it calculates a simple arithmetic average of the heat capacities of all possible species defined in the global SPECIES list at the given temperature T. This provides a reasonable default value and prevents the program from crashing.
    y = F / Ft #Calculate Mole Fractions
    cps = np.array([Cp_species(T, s) for s in SPECIES], dtype=float) #calculates the specific heat capacity for each individual species at the given temperature T.
    return float(np.dot(y, cps)) #calculates the dot product of the mole fraction array y and the individual heat capacity array cps. TO FIND the denominator in that energy balance equation (Yee2003, Appx A (A3), OR 6.212 PAGE 369).

# -----------------------------------------------------------------------------

    
# -- NEW: species and mixture enthalpy (from ∫Cp dT) -------------------------
# This functions, in turn, are used in the he1_energy_balance
def integral_Cp_species(T: float, species: str) -> float: #T: float: The target temperature in Kelvin (K)  str) -> float: The calculated molar sensible enthalpy in kJ/kmol.
    """
    ∫_0^T Cp_j(θ) dθ [kJ/kmol] using Cp = a + b T + c/T^2
    => a T + 0.5 b T^2 - c/T   (for T>0). Valid for our 800–950 K range.
    """
    T = float(T) 
    if T <= 1e-9:  # This is a critical robustness check. The final term in the integrated formula is -c/T. If T were zero, this would cause a "division by zero" error and crash the simulation.
        return 0.0
    s = species.upper() #This block finds the empirical constants a, b, and c for the requested species.
    if s in _CP_ORG:
        co = _CP_ORG[s]
    elif s in _CP_INORG:
        co = _CP_INORG[s]
    else:
        raise KeyError(f"No Cp coefficients for '{species}'")
    a, b, c = co['a'], co['b'], co['c'] # unpacks the a, b, and c values from the dictionary co.
    return a*T + 0.5*b*T*T - c/max(1e-9, T) # you can return to the handwriting note to see how this comes (https://kansas-my.sharepoint.com/:i:/g/personal/m443a781_home_ku_edu/ERk-uexPYg9Eh7Kmfbv6EVMBQLXfftv_PNs_74TrjMEHpQ?e=BLDHg6 ).


# -----------------------------------------------------------------------------

def mixture_enthalpy(F: np.ndarray, T: float) -> float: #gives absolute enthalpy flow relative to 0 K
    """Mixture sensible enthalpy at T relative to 0 K [kJ/h]. Based on ∑F_j ∫Cp_j dT."""
    return sum(F[j] * integral_Cp_species(T, SPECIES[j]) for j in range(10)) #H_mix=∑ F_j ∫(0toT) C_p,jdT

def mixture_enthalpy_change(F: np.ndarray, T1: float, T2: float) -> float: # gives the ΔH between two temperatures for the same composition
    """ΔH_mix = ∫_{T1}^{T2} Cp_mix dT [kJ/h] for a fixed-composition stream.Used in HE1."""
    return mixture_enthalpy(F, T2) - mixture_enthalpy(F, T1) # ΔH_mix = ∫_{T1}^{T2} Cp_mix dT

# ΔH of reaction vector (kJ/kmol): ΔH_i(T) = a_i + b_i T  [E&E1994, eq. 6.213, Table 6.24].
# Values read from the table (units already kJ/kmol and kJ/kmol/K).
_DH_COEF = {
    1: dict(a=12075.110, b= +4.56),   # EB → ST + H2 the book is misswrite the constant for the reaction no 1. i have check with two references which confirm this misswriting. Sheel, J. P.  1969
    2: dict(a=108818.11, b= -7.96),   # EB → BZ + ETH
    3: dict(a=-53178.20, b=-13.19),   # EB + H2 → TOL + CH4
    4: dict(a= 82065.74, b= +8.84),   # 2H2O + ETH → 2CO + 4H2
    5: dict(a=211255.19, b=+16.58),   # H2O + CH4 → CO + 3H2
    6: dict(a=-45223.66, b=+10.47),   # H2O + CO → CO2 + H2
}

def delta_H_vector(T: float) -> np.ndarray:
    """ΔH_i(T) for reactions i=1..6 [kJ/kmol], sign: + = endothermic."""
    return np.array([_DH_COEF[i]['a'] + _DH_COEF[i]['b']*T for i in range(1, 7)],
                    dtype=float)

# -----------------------------------------------------------------------------
# 2) Catalyst bed geometry and operating parameters
# -----------------------------------------------------------------------------

@dataclass
class CatalystBed:
    """
    Packed-bed geometry and properties.
    Defaults from [Yee2003, Appx A, Table A2]: Bed void fraction (ε)=0.445, Catalyst bulk density (ρ_b)=2146 kg/m³, Catalyst particle diameter (d_p) =0.0047 m.
    """
    D: float               # reactor diameter [m] — decision variable [Tarafder2005]
    L_over_D: float        # L/D ratio [-] — decision variable [Tarafder2005]
    eps: float = 0.445     # [Yee2003, Table A2]
    rho_b: float = 2146.0  # [Yee2003, Table A2]
    dp: float = 0.0047     # [Yee2003, Table A2]

    @property
    def L(self) -> float:
        """Bed length L = (L/D)*D. Decision-variable coupling. [Tarafder2005]"""
        return self.L_over_D * self.D

    @property
    def area(self) -> float:
        """Cross-sectional area =π r^2 = π(D/2)^2 used in Ergun and molar→mass flux. [Yee2003]"""
        return math.pi * (0.5 * self.D)**2


@dataclass
class OperatingParams:
    """
    Operating decisions and fixed utilities following [Tarafder2005, Sec. 4–6] and [Yee2003].
    SB: Pin, SOR, F0_EB, D1, L/D1, T_EB, α, TC2
    SI: + δ (steam split), λ (injection location)
    DB: + Tmix2 (reheat setpoint), and second-bed geometry
    """
    # Decision variables (Tarafder 2005, Sec. 4–6):
    Pin: float                 # inlet pressure [bar]  — bounds in [Tarafder2005]
    SOR: float                 # steam-to-EB molar ratio [-]  — [Tarafder2005]
    F0_EB: float               # fresh EB feed [kmol/h]
    bed1: CatalystBed          # first bed geometry
    T_EB: float                # cold EB to HE1 [K]
    alpha: float               # fraction of total steam pre-mixed (saturated) [Tarafder2005]
    TC2: float                 # HE1 cold outlet temp (EB+αsteam) [K]

    # SI only:
    delta: float = None        # fraction of (1-α) superheated steam at inlet
    lam: float = None          # injection location fraction of bed length
    # DB only:
    Tmix2: float = None        # inter-bed reheat setpoint [K]
    bed2: CatalystBed = None   # second bed

    # Utilities (per Yee 2003):
    T_sat: float = 405.0       # saturated steam T used in HE1 side [K]
    T_sh: float = 1025.0       # superheated steam temperature [K]

    # EB feed impurities (kmol/h); Yee 2003 Table A2
    F0_impurities: Dict[str, float] = field(
        default_factory=lambda: {'ST': 0.67, 'BZ': 0.11, 'TOL': 0.88}
    )

    # HE1 overall U for approach checks [kJ/h/m^2/K] — used only in constraints
    U_HE1: float = 55.0

# -----------------------------------------------------------------------------
# 3) Kinetics (Yee 2003, Appendix A → Table A1), reversible r1
# -----------------------------------------------------------------------------

@dataclass
class Kinetics:
    """
    Sheel–Crowe pseudo-homogeneous rate constants consolidated by Yee (2003).
    k_i(T) = exp(A_i - E_i/(R T)), with partial pressures in bar.

    Optionally, intrinsic parameters from Elnashaie & Elshishini (1994, Table 7)
    can be enabled by source='ELNASH1994_INDUSTRIAL' (A converted from h^-1).
    """
    source: str = "YEE2003"  # or "ELNASH1994_INDUSTRIAL"

    def __post_init__(self):
        if self.source.upper() == "YEE2003":
            self.A = np.array([-0.0854, 13.2392, 0.2961, -0.0724, -2.9344, 21.2402])
            self.E = np.array([ 90981.4, 207989.2, 91515.3, 103996.7, 65723.3, 73628.4 ])

        elif self.source.upper() == "ELNASH1994_INDUSTRIAL":
            # Table 7 (industrial catalyst); k = exp(A - E/(R T)) gives per-second rates. 
            self.A = np.array([0.851, 14.000, 0.560, 0.120, -3.210, 21.240])
            self.E = np.array([90891.0, 207989.0, 91515.0, 103996.0, 65723.0, 73628.0])

        else:
            raise ValueError("Unknown kinetics source.")

        # Non-integer orders noted by Yee (Appendix A notes)
        #self.n_H2_r3 = 1   # order in H2 for r3: EB + H2 → TOL + CH4
        #self.n_ETH_r4 = 0.5    # order in ETH for r4

    def k(self, T: float) -> np.ndarray:
        return np.exp(self.A - self.E / (R * T))

    @staticmethod
    def K_eq_r1(T: float) -> float:
        # Yee’s correlation for EB <-> ST + H2 equilibrium (Appendix A note)
        return math.exp(-(122725.0 - 126.3*T - 0.002194*T*T) / (R * T))

# -----------------------------------------------------------------------------
# 4) Transport & mixture utilities
# -----------------------------------------------------------------------------
#mixture molecular weight MW=∑y_j MW_j (mixture 𝑀𝑊 is needed  for density)
def mixture_MW(F: np.ndarray) -> float:
    Ft = float(F.sum())
    if Ft <= 0.0:
        return float(MW.mean())
    y = F / Ft #Fj:molar flow rates (kmol/h)
    return float(np.dot(y, MW))

# ρ_g = P (mixture_MW)/(RT) (ρ_g is needed for density Ergun equation)
def gas_density(P_bar: float, T: float, F: np.ndarray) -> float:
    """Ideal-gas mixture density [kg/m^3]."""
    MW_bar = mixture_MW(F)     # kg/kmol
    P_Pa = P_bar * BAR_TO_PA
    R_J = 8.314                # J/mol-K
    MW_kg_per_mol = MW_bar / 1000.0 #divide the usual kg/kmol by 1000
    return (P_Pa * MW_kg_per_mol) / (R_J * T) #The model assumes ideal‑gas behaviour in the reactor (stated in the modelling assumptions for gas mixtures in the Elnashaie & Elshishini text used by Yee/Tarafder). 

def gas_viscosity(T: float) -> float:
    """
    Sutherland-type correlation, scaled so μ(900 K) ≈ 3.0e-5 Pa·s.
    Guard bands avoid non-physical values if the ODE wanders.
    """
    T0, S, mu0 = 300.0, 110.0, 1.42e-5
    Te = float(np.clip(T, 200.0, 2000.0))
    return mu0 * (Te / T0)**1.5 * (T0 + S) / (Te + S)

def ergun_dPdz(P_bar: float, T: float, F: np.ndarray, bed: CatalystBed) -> float:
    """
    Ergun equation (momentum balance) → dP/dz [bar/m].
    Uses superficial mass flux G [kg/m^2/s] and mixture density.
    """
    rho_g = max(1e-6, gas_density(P_bar, T, F)) #compute mixture density 𝜌_𝑔 (ideal gas; above), max(1e-6, rho_g) avoids divide‑by‑zero if the integrator briefly pushes
    mdot_h = float(np.dot(F, MW))  # mass flow          kg/h
    G = (mdot_h / HOUR_TO_SEC) / bed.area #mass flux G  kg/m^2/s

    eps, dp = bed.eps, bed.dp
    mu = gas_viscosity(T)


    # rearranged with rho_g to keep formula numerically stable
    #Ergun looks “different” from Yee’s printed (A4): same equation, just different unit system.
    term1 = 150.0 * (1.0 - eps)**2 * mu    * (G / rho_g) / (eps**3 * dp**2)
    term2 = 1.75  * (1.0 - eps)    * (G**2 / rho_g)      / (eps**3 * dp)
    return (term1 + term2) / BAR_TO_PA # Pa→bar
'''

# -----------------------------------------------------------------------------
# Replace the old Sutherland-fit with a corresponding-states (Chung et al.) pure‑gas
# viscosity plus Wilke mixing to obtain μ_mix(T, y). This section is self‑contained.
#
# References:
#   - Chung, Lee & Starling, Ind. Eng. Chem. Res. 27, 671–679 (1988) — generalized CS model
#   - Neufeld, Janzen & Aziz, J. Chem. Phys. 57, 1100–1102 (1972) — collision integral Ω_μ(T*)
#   - Wilke, J. Chem. Phys. 18, 517–519 (1950) — gas mixture viscosity mixing rule
# Units:
#   - Inputs: T [K], composition as molar flows F [kmol/h]
#   - Critical volume Vc is used in cm^3/mol per Chung’s original units
#   - Output: dynamic viscosity μ [Pa·s]
# -----------------------------------------------------------------------------

# Critical + shape data for species used in this flowsheet.
# Fields:
#   Tc [K], Pc [bar] (optional if Vc is given), Zc [-] (optional, used only to derive Vc if Vc_cm3 not given),
#   Vc_cm3 [cm^3/mol] (optional; if omitted, it is derived from Tc, Pc, Zc by Vc = Zc*R*Tc/Pc),
#   omega [-], dipole_D [debye], kappa_assoc [-] (Chung association factor; 0 for non‑associating fluids)
_CS_DATA = {
    # Aromatics & feed/product
    'EB' : dict(Tc=617.1, Pc=36.1, Zc=0.290, omega=0.304, dipole_D=0.58,  kappa_assoc=0.0),  # Ethylbenzene
    'ST' : dict(Tc=646.2, Pc=40.0, Zc=0.285, omega=0.300, dipole_D=0.13,  kappa_assoc=0.0),  # Styrene (ω≈0.30 placeholder; update if you have a vetted value)
    'BZ' : dict(Tc=562.1, Pc=48.9, Zc=0.271, omega=0.211, dipole_D=0.00,  kappa_assoc=0.0),  # Benzene
    'TOL': dict(Tc=591.8, Pc=41.1, Zc=0.264, omega=0.266, dipole_D=0.36,  kappa_assoc=0.0),  # Toluene
    # Lights and inorganics
    'ETH': dict(Tc=282.3, Pc=50.4, Zc=0.280, omega=0.087, dipole_D=0.00,  kappa_assoc=0.0),  # Ethylene
    'CH4': dict(Tc=190.6, Pc=46.0, Zc=0.286, omega=0.011, dipole_D=0.00,  kappa_assoc=0.0),  # Methane
    'CO' : dict(Tc=132.9, Pc=34.95, Zc=0.294, omega=0.050, dipole_D=0.12, kappa_assoc=0.0),  # Carbon monoxide
    'CO2': dict(Tc=304.2, Pc=73.8, Zc=0.274, omega=0.225, dipole_D=0.00,  kappa_assoc=0.0),  # Carbon dioxide
    'H2O': dict(Tc=647.1, Pc=220.6, Zc=0.229, omega=0.344, dipole_D=1.85, kappa_assoc=0.076),# Water (assoc. factor per Chung/Poling)
    'H2' : dict(Tc=33.2,  Pc=12.9, Zc=0.303, omega=-0.220,dipole_D=0.00,  kappa_assoc=0.0),  # Hydrogen (CS methods are less reliable here; used with care)
}

def _ensure_Vc_cm3(species: str) -> float:
    """Return Vc in cm^3/mol, deriving it from (Tc,Pc,Zc) if not explicitly provided."""
    d = _CS_DATA[species]
    if 'Vc_cm3' in d and d['Vc_cm3'] and d['Vc_cm3'] > 0.0:
        return float(d['Vc_cm3'])
    # Derive from Vc = Zc * R * Tc / Pc, using R = 0.08314 L·bar/mol/K, then convert to cm^3/mol.
    Tc = d['Tc']; Pc = d['Pc']; Zc = d['Zc']
    if any(k not in d or d[k] is None for k in ('Tc','Pc','Zc')):
        raise KeyError(f"Critical data incomplete for Vc derivation of '{species}'. Provide Vc_cm3 or (Tc,Pc,Zc).")
    R_bar = 0.08314  # L·bar/(mol·K)
    Vc_L_per_mol = Zc * R_bar * Tc / Pc
    Vc_cm3_per_mol = 1000.0 * Vc_L_per_mol
    _CS_DATA[species]['Vc_cm3'] = Vc_cm3_per_mol
    return Vc_cm3_per_mol

def _omega_mu_neufeld(Tstar: float) -> float:
    """Neufeld collision integral for viscosity Ω_μ(T*)."""
    # Ω = A*T*^-B + C*exp(-D*T*) + E*exp(-F*T*)
    A, B = 1.16145, 0.14874
    C, D = 0.52487, 0.77320
    E, F = 2.16178, 2.43787
    return A*(Tstar**(-B)) + C*math.exp(-D*Tstar) + E*math.exp(-F*Tstar)

def _pure_mu_chung(species: str, T: float) -> float:
    """
    Chung–Lee–Starling corresponding‑states gas viscosity for a pure component.
    Returns μ_i(T) in Pa·s.
    """
    if species not in _CS_DATA:
        raise KeyError(f"No corresponding-states data for species '{species}'. Add to _CS_DATA.")
    d = _CS_DATA[species]
    Tc = float(d['Tc'])
    Vc_cm3 = _ensure_Vc_cm3(species)
    omega = float(d['omega'])
    mu_D = float(d.get('dipole_D', 0.0))
    kappa = float(d.get('kappa_assoc', 0.0))

    # Reduced temperature for Chung/Neufeld (scaled): T* = 1.2593 * T/Tc
    T = float(np.clip(T, 200.0, 2000.0))
    Tstar = 1.2593 * (T / Tc)
    Omega = _omega_mu_neufeld(Tstar)

    # Molecular shape/polarity factor F_c
    # p_r = 131.3 * μ[D] / sqrt(Vc[cm^3/mol] * Tc[K])
    pr = 131.3 * mu_D / math.sqrt(Vc_cm3 * Tc) if mu_D > 0.0 else 0.0
    F_c = 1.0 - 0.2756*omega + 0.059035*(pr**4) + kappa

    # Base constant 40.785 yields μ in micropoise when:
    #   M in g/mol, T in K, Vc in cm^3/mol, Ω dimensionless
    # Convert to SI: 1 μP = 1e-7 Pa·s
    # Here MW array is kg/kmol, numerically equal to g/mol.
    M_g_per_mol = float(MW[IDX[species]])  # kg/kmol == g/mol
    mu_microP = 40.785 * F_c * math.sqrt(M_g_per_mol * T) / ( (Vc_cm3**(2.0/3.0)) * Omega )
    return mu_microP * 1e-7  # Pa·s

def _wilke_mixture_mu(y: np.ndarray, mu_i: np.ndarray, M_g_per_mol: np.ndarray) -> float:
    """
    Wilke (1950) mixing rule for gas viscosity.
    y: mole fractions (shape [n])
    mu_i: pure-component viscosities [Pa·s] (shape [n])
    M_g_per_mol: molecular weights in g/mol (shape [n])
    Returns μ_mix [Pa·s].
    """
    n = y.size
    if n == 0 or y.sum() <= 0.0:
        return float(np.mean(mu_i)) if mu_i.size else 2.8e-5
    y = y / y.sum()
    # Compute interaction parameters φ_ij
    # φ_ij = [1 + (μ_i/μ_j)^{1/2} (M_j/M_i)^{1/4}]^2 / [sqrt(8) (1 + M_i/M_j)^{1/2}]
    M = M_g_per_mol
    mu = mu_i
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            rij = math.sqrt(max(1e-300, mu[i] / mu[j]))
            mij = (M[j] / M[i])**0.25
            denom = math.sqrt(8.0) * math.sqrt(1.0 + (M[i] / M[j]))
            phi[i, j] = ((1.0 + rij * mij)**2) / denom
    # Wilke sum
    denom_vec = phi @ y
    denom_vec = np.where(denom_vec > 1e-300, denom_vec, 1e-300)
    return float(np.sum(y * mu / denom_vec))

def gas_viscosity(T: float, F: np.ndarray) -> float:
    """
    Mixture gas viscosity via corresponding‑states (Chung) for μ_i(T) and Wilke mixing.
    T [K], F [kmol/h] over SPECIES order. Returns μ_mix [Pa·s].
    """
    # Compute mole fractions over the 10‑component set used elsewhere in this module
    F = np.maximum(0.0, np.asarray(F, dtype=float))
    Ft = F.sum()
    if Ft <= 0.0:
        # fallback to steam at T as a benign default
        return _pure_mu_chung('H2O', T)
    y = F / Ft

    # Pure viscosities (corresponding‑states)
    mu_i = np.zeros(10, dtype=float)
    for j, s in enumerate(SPECIES):
        mu_i[j] = _pure_mu_chung(s, T)

    # Molecular weights for Wilke (in g/mol)
    M_g_per_mol = MW.copy()  # kg/kmol == g/mol numerically
    return _wilke_mixture_mu(y, mu_i, M_g_per_mol)

def ergun_dPdz(P_bar: float, T: float, F: np.ndarray, bed: CatalystBed) -> float:
    """
    Ergun equation (momentum balance) → dP/dz [bar/m].
    Uses superficial mass flux G [kg/m^2/s], mixture density, and μ_mix from CS+Wilke.
    """
    rho_g = max(1e-6, gas_density(P_bar, T, F))
    mdot_h = float(np.dot(F, MW))           # kg/h
    G = (mdot_h / HOUR_TO_SEC) / bed.area   # kg/m^2/s

    eps, dp = bed.eps, bed.dp
    mu = gas_viscosity(T, F)                # <-- composition‑dependent μ_mix

    term1 = 150.0 * (1.0 - eps)**2 * mu    * (G / rho_g) / (eps**3 * dp**2)
    term2 = 1.75  * (1.0 - eps)    * (G**2 / rho_g)      / (eps**3 * dp)
    return (term1 + term2) / BAR_TO_PA
'''
# -----------------------------------------------------------------------------
# 5) Reaction rates and PFR ODEs
# -----------------------------------------------------------------------------

def reaction_rates(T: float, P_bar: float, F: np.ndarray, kin: Kinetics) -> np.ndarray:
    """
    r_i [kmol/kg-cat/s] using [Yee2003, Appx A, Table A1]. Partial pressures in bar.
    r1 reversible via K_eq(T) from [Yee2003, Appx A note]. Orders per appendix notes.
    """
    #Converting flows → mole fractions → partial pressures
    Ft = float(F.sum())
    y = (F / Ft) if Ft > 0.0 else np.zeros_like(F)
    p = y * P_bar #partial pressures Elnashaie & Elshishini, Eq. (6.206)

    k = kin.k(T)
    Keb = kin.K_eq_r1(T) #Reversible equilibrium for reaction 1

    pEB  = max(1e-12, p[IDX['EB']])
    pST  = max(1e-12, p[IDX['ST']])
    pH2  = max(1e-12, p[IDX['H2']])
    pETH = max(1e-12, p[IDX['ETH']])
    pH2O = max(1e-12, p[IDX['H2O']])
    pCH4 = max(1e-12, p[IDX['CH4']])
    pCO  = max(1e-12, p[IDX['CO']])

    r = np.zeros(6, dtype=float)
    r[0] = k[0] * (pEB - (pST * pH2) / max(1e-16, Keb))              # r1: EB ⇄ ST + H2
    r[1] = k[1] * pEB                                                # r2: EB → BZ + ETH
    r[2] = k[2] * pEB * pH2                         # r3: EB + H2 → TOL + CH4
    r[3] = k[3] * pH2O * (pETH ** 0.5)                      # r4: 2H2O + ETH → 2CO + 4H2
    r[4] = k[4] * pH2O * pCH4                                        # r5: H2O + CH4 → CO + 3H2
    r[5] = k[5] * (P_bar / (T**3)) * pH2O * pCO                      # r6: H2O + CO → CO2 + H2
    r[1:] = np.maximum(r[1:], 0.0)  # keep r1 reversible only
    return r

def reactor_odes(z: float, y: np.ndarray, bed: CatalystBed, kin: Kinetics) -> np.ndarray:
    """
    PFR pseudo-homogeneous balances. State y = [F(10), T, P].
    Returns dy/dz.
    """
    F = np.maximum(0.0, y[:10].copy())
    T = float(y[10]); P_bar = float(y[11])

    if not np.isfinite(T) or not np.isfinite(P_bar):
        raise ValueError("non-finite state in reactor_odes")
    if (T < 200.0) or (T > 2000.0) or (P_bar < 0.01) or (P_bar > 30.0):
        raise ValueError("state outside bounds in reactor_odes")

    # Rates
    r = reaction_rates(T, P_bar, F, kin)

    # Stoichiometry ν_{species, reaction} — matches [Yee2003, Appx A reactions]
    NU = np.array([
        [-1, -1, -1,  0,  0,  0],  # EB
        [ 1,  0,  0,  0,  0,  0],  # ST
        [ 1,  0, -1,  4,  3,  1],  # H2
        [ 0,  1,  0,  0,  0,  0],  # BZ
        [ 0,  1,  0, -1,  0,  0],  # ETH
        [ 0,  0,  1,  0,  0,  0],  # TOL
        [ 0,  0,  1,  0, -1,  0],  # CH4
        [ 0,  0,  0,  2,  1, -1],  # CO
        [ 0,  0,  0,  0,  0,  1],  # CO2
        [ 0,  0,  0, -2, -1, -1],  # H2O
    ], dtype=float)

    # dF/dz = ρ_b A * ν r * 3600  (kmol/h per m) — form in [Yee’s Appendix A writes balances in terms of conversions X; the form is dXᵢ/dz = (ρ_b A rᵢ)/F₀ (their Eqs. (A1)–(A2))]
    rbA = bed.rho_b * bed.area
    dFdz = rbA * (NU @ r) * HOUR_TO_SEC

    # Energy balance: [Yee2003, Appx A energy], with ΔH_i(T) and Cp_j(T) from [E&E1994]
    Qr = float(np.dot(delta_H_vector(T), r))              # (detlat_H * r_i) (part of the numerator of eq. A3 Yee2003)            # kJ/kg-cat/s
    mCp = sum(F[j] * Cp_species(T, SPECIES[j]) for j in range(10))  # kJ/h/K
    mCp_s = mCp / HOUR_TO_SEC
    dTdz = 0.0 if mCp_s <= 1e-12 else -(rbA * Qr) / mCp_s



    # Momentum balance via Ergun [Yee2003, Appx A]
    # Pressure drop: Ergun
    dPdz = -ergun_dPdz(P_bar, T, F, bed)

    dy = np.zeros_like(y)
    dy[:10] = dFdz #put the 10 species derivatives first (same slots as F)
    dy[10] = dTdz #then temperature T (K)
    dy[11] = dPdz #then pressure P (bar).
    return dy  

# Simple fixed-step RK4
def rk4_integrate(f: Callable[[float, np.ndarray], np.ndarray],
                  z0: float, y0: np.ndarray, zf: float, nstep: int) -> Tuple[np.ndarray, np.ndarray]:
    z = np.linspace(z0, zf, nstep + 1)
    y = np.zeros((nstep + 1, len(y0)), dtype=float)
    y[0, :] = y0
    h = (zf - z0) / nstep
    for i in range(nstep):
        zi, yi = z[i], y[i, :]
        k1 = f(zi, yi)
        k2 = f(zi + 0.5*h, yi + 0.5*h*k1)
        k3 = f(zi + 0.5*h, yi + 0.5*h*k2)
        k4 = f(zi + h,     yi + h*k3)
        y[i+1, :] = yi + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.all(np.isfinite(y[i+1, :])):
            raise ValueError("non-finite state in RK4 step")
    return z, y

# -----------------------------------------------------------------------------
# 6) Flowsheet stubs (HE1 and mixers) + simulation harness
# -----------------------------------------------------------------------------
'''
This block is where the “mini‑flowsheet” around the reactor is coded (the two mixers and the EB preheater HE1), and where the three reactor configurations (SB, SI, DB) are stitched together and evaluated.
'''
def mix_temperatures(F_cold: float, Cp_cold: float, T_cold: float,
                     F_hot: float,  Cp_hot:  float, T_hot:  float) -> float:
    '''
    This computes the temperature  when blending two gas streams with no heat loss to the surroundings (an adiabatic mixer)
    Tarafder et al. explicitly state that the mixer temperatures Tmix1 (reactor inlet) and 𝑇mix2 (the second mix point for SI/DB) are calculated from an energy balance with negligible heat loss Qhot=Qcold.
    '''
    if F_cold <= 0.0 and F_hot <= 0.0: #Guard for a degenerate case: if no flow in either stream, just keep the cold-stream temperature.
        return T_cold
    Qcold = F_cold * Cp_cold * T_cold #Units: F (kmol/h) × Cp(kJ/kmol·K) × T (K) → kJ/h.
    Qhot  = F_hot  * Cp_hot  * T_hot
    denom = F_cold*Cp_cold + F_hot*Cp_hot #The total heat-capacity rate of the combined stream (kJ/h/K).
    return T_cold if denom <= 1e-12 else (Qcold + Qhot) / denom #If the denominator is ~zero, avoid divide-by-zero and fall back to Tcold, Otherwise return the adiabatic mixing temperature
'''
HE1 is the countercurrent exchanger that preheats EB + (a fraction of) saturated steam using the hot reactor effluent, counter-current, with energy balance + LMTD area. Tarafder gives the governing equations (7)–(10) and uses U = 55 kJ h⁻¹ m⁻² K⁻¹. HE1 has no phase change ( so need to use equation 7 in Tarafder2005; HE2 does.
For HE1, the code uses the integral ∫Cp(T)dT because the exchanger spans a larger ΔT. Tarafder’s HE1 model and LMTD equations are given in §3.2 (Eqs. 7–10).
'''
def he1_energy_balance(F_cold_vec: np.ndarray, TC1: float, TC2: float, 
                       F_hot_vec:  np.ndarray, TH1: float, U: float
                      ) -> Tuple[float, float, float, float]:
    """
    HE1 (countercurrent, sensible only):
      Qcold = Σ F_j ∫_{TC1}^{TC2} Cp_j(T) dT  [kJ/h]
      Qhot  = Σ F_j ∫_{TH1}^{TH2} Cp_j(T) dT  [kJ/h] = -Qcold  (solve for TH2)
      A     = Q / (U * LMTD),  ΔT_lm = (ΔT1-ΔT2)/ln(ΔT1/ΔT2),  ΔT1=TH1-TC2, ΔT2=TH2-TC1
    Returns (Qcold, TH2, LMTD, A).
    """
    # cold-side duty from enthalpy integral, This is Tarafder’s Eq. (9) for
    Qcold = mixture_enthalpy_change(F_cold_vec, TC1, TC2)  # >0 if TC2>TC1

    # solve for TH2 so that Qhot releases Qcold
    if Qcold <= 0.0: #If there’s no required cold-side gain, set hot-out = hot-in (no cooling).
        TH2 = TH1 #Programming guard; consistent with Eq. (8): Qhot=Qcold
    else:
        # bracket: TH2 must lie between (TC1+10 K) and TH1 by approach constraint
        T_lo = max(200.0, TC1 + 10.0 + 1e-6) #Bracket TH2 between (TC1 + 10 K) and TH1 to enforce 10 K minimum approach at each end of HE1 (Tarafder constraints Eqs. (33)–(34)).
        T_hi = max(T_lo + 1e-6, TH1 - 1e-6)

        
#####---------------------- root-finding-------------------------------#######
        def balance(T): #build a little helper function, balance(T), that tells the mismatch at a trial temperature 𝑇. f balance(T) = 0, the mismatch is zero → energy is perfectly balanced → that T is the answer. Because this function is smooth and always increases with T (Cp > 0), it can be solved for the zero safely
            # H_hot(T) - H_hot(TH1) + Qcold = 0  →  Qhot = Qcold
            return mixture_enthalpy_change(F_hot_vec, TH1, T) + Qcold #hot-side enthalpy change when cooling from TH1 down to trial T, plus Qcold.

            '''
            Meaning:
            
            If result > 0 → the hot side hasn’t released enough heat yet (we need a lower T).
            
            If result < 0 → it released too much heat (we need a higher T).
            
            If result = 0 → perfect match: Qhot=Qcold
            '''

        # try bisection (monotone in T because dH/dT = ΣF Cp > 0), evaluate the mismatch at two temperatures that bracket the answer.
        g_lo, g_hi = balance(T_lo), balance(T_hi) #T_lo is the lowest allowed (respecting the 10 K approach). and T_hi is near the hot inlet.
        if g_lo * g_hi > 0.0: #If both values have the same sign, the zero isn’t guaranteed to lie between them (not bracketed).
            # switch to a fast Newton strategy (below) from mid-point, [Tnew=T−f(T)/f′(T)]
            T = 0.5*(T_lo + TH1) #Start Newton’s method from a sensible middle guess.
            #Compute the slope of balance(T) at this T
            for _ in range(12): 
                Cp_hot = sum(F_hot_vec[j] * Cp_species(T, SPECIES[j]) for j in range(10))
                if Cp_hot <= 1e-12: break #Safety: avoid divide-by-zero.
                T -= balance(T) / Cp_hot #Newton step: draw a tangent line at the current point and jump to where that line crosses zero.
                T = float(np.clip(T, T_lo, T_hi)) #Keep the guess inside allowed limits (10 K approach, physical range)
            TH2 = T #Newton found the hot outlet temperature.
        #If the signs at the ends are different, the zero must be between them → use bisection:
        else:
            for _ in range(80):
                T_mid = 0.5*(T_lo + T_hi)
                if balance(T_mid) > 0.0:
                    T_hi = T_mid
                else:
                    T_lo = T_mid
            TH2 = 0.5*(T_lo + T_hi)
            '''
            Bisection = guaranteed but slower.
            Newton = fast but can misbehave.
            Combining them gives a solver that is both safe and quick for the HE1 energy-balance equation.

            '''

    # LMTD and area
    dT1 = max(1e-6, TH1 - TC2)  # hot-in – cold-out
    dT2 = max(1e-6, TH2 - TC1)  # hot-out – cold-in
    #computes the log-mean temperature difference (LMTD).
    if abs(dT1 - dT2) < 1e-12: #If ΔT1 ≈ ΔT2, the log-mean tends to that common value; this branch avoids log(1) / 0 round-off.
        LMTD = dT1
    else:
        LMTD = (dT1 - dT2) / math.log(dT1 / dT2)
    A = 0.0 if LMTD <= 1e-12 else Qcold / (U * LMTD) #equestion (10) Tarafder2005
    return Qcold, TH2, LMTD, A


@dataclass
class SimulationResult:#A tidy container for everything  want back from one flowsheet run (reactor + HE1). Easier for the optimizer to read and to debug.
    F_out: np.ndarray #Outlet molar flows of all 10 species (kmol/h).feeds both objectives (styrene flow, selectivity) and gives the hot-stream composition for HE1 duty/LMTD. Case-1 objectives are Fst and Sst.
    T_out: float #Reactor outlet temperature (K)
    P_out: float #Reactor outlet pressure (bar).Tarafder’s constraint Pexit≥1.4 bar
    Fst_out: float#Styrene outlet flow (kmol/h).this is objective J1 in Case 1
    Sst: float#Styrene selectivity (%) at the reactor outlet.objective J2 in Case 1
    Tmix1: float#Inlet mix temperature to the reactor (K).
    Tmix2: float | None#Second mix/reheat temperature (K)or None (SB: not used).
    TH1: float#HE1 hot-in temperature (K).
    TH2: float#HE1 hot-out temperature (K).
    TC1: float#HE1 cold-in temperature TEB(K).
    TC2: float#HE1 cold-out temperature (K).
    LMTD_HE1: float#Log-mean temperature difference for HE1 (K).
    A_HE1: float#Calculated HE1 area (m²).
    Q_HE1: float#HE1 heat duty transferred (kJ/h).
def simulate_reactor(config: str, op: OperatingParams, kin: Kinetics, nstep: int = 300) -> SimulationResult:
    """
    config picks the topology: SB, SI, or DB.  
    op carries all decision/utility inputs. 
    kin holds kinetic params.
    nstep is ODE step count for RK4 along the bed length(s).
    
    Simulate EB preheat + packed bed(s) for config in {'SB',' SI',' DB'}.
    Returns outlet state and objective metrics.
    """
    assert config in ('SB', 'SI', 'DB') # Sanity check: only the three supported topologies are allowed.
    # total steam


    # Determine the T_C2 (out of the HE1), see fig. 1 in Tarafder.
    F0_stm = op.SOR * op.F0_EB#steam‑over‑reactant (SOR)=steam / EB. so, Total steam feed (αF0_stm) = SOR × fresh EB. Used to split into α (saturated) and (1−α) (superheated). see fig1 in Tarafder2005.

    # Cold side to HE1: EB + impurities + (αF0_steam).This is exactly the “premix to HE1” branch in the flowsheet.
    F0 = np.zeros(10, dtype=float)#Start a 10-component flow vector (one slot per species).
    F0[IDX['EB']] = op.F0_EB #Put fresh EB into its slot.
    for s, v in op.F0_impurities.items(): #Add plant impurities in the EB feed (e.g., ST, BZ, TOL). see table A2 in Yee2003.
        F0[IDX[s]] += v
    F_cold_to_HE1 = F0.copy()#Make a separate stream object for HE1’s cold side. (Copy avoids mutating the original baseline vector.)
    F_cold_to_HE1[IDX['H2O']] += op.alpha * F0_stm #Add the α fraction of total steam as saturated steam to HE1’s cold side. The remaining (1−α) will be superheated for mixing at the reactor.
    # Cold-side inlet to HE1 (TC1) must be the adiabatic mix of EB(+impurities) at T_EB
    # with α·saturated steam at T_sat (Tarafder 2005, mixer before HE1).
    F_EBimp = F0.copy()                           # EB + impurities only (no steam)
    F_sat_alpha = op.alpha * F0_stm               # α·steam to premix
    TC1_cold = mix_temperatures(
        F_EBimp.sum(),                            # flow of EB+impurities
        mixture_cp(F_EBimp, op.T_EB),             # Cp of EB+impurities at T_EB
        op.T_EB,                                  # EB feed temperature
        F_sat_alpha,                              # α·steam flow
        Cp_species(op.T_sat, 'H2O'),              # Cp of steam at T_sat
        op.T_sat                                  # saturated steam temperature
    )

    Cp_cold = mixture_cp(F_cold_to_HE1, op.TC2)       #Compute mixture ∑yjCp,j at the cold outlet temp TC2.
    F_cold_total = float(F_cold_to_HE1.sum())        # Total molar flow of the HE1 cold stream. Also needed in the mixing energy balance.

    '''
    the above block builds the HE1 cold-side stream exactly as in the flowsheet (EB + impurities + α·steam) and computes the properties (total flow and Cp at TC2) we need to mix with the superheated portion later to get the reactor inlet temperature Tmix1.
    '''





    
    # Split of superheated steam, Determine how the superheated steam is used:
    if config == 'SI':#SI: split into inlet part (δ) and in-bed injection part (1−δ). See fig.2b 
        assert op.delta is not None and op.lam is not None #Safety: SI needs δ (inlet split) and λ (injection location).
        F_sh_inlet  = op.delta * (1.0 - op.alpha) * F0_stm #(1-α) δ Fo_stm. see fig1 Taradser2005
        F_sh_inject = (1.0 - op.delta) * (1.0 - op.alpha) * F0_stm #The remaining hot steam is reserved for in-bed injection [(1-α)(1-δ)Fo_stm], see fig1 Taradser2005
    else:
        F_sh_inlet, F_sh_inject = (1.0 - op.alpha) * F0_stm, 0.0 #SB/DB: no mid-bed injection. All hot steam goes to the inlet; injected portion is zero.

    # Inlet mixing temperature (HE1 cold outlet with superheated steam)
    Tmix1 = mix_temperatures(F_cold_total, Cp_cold, op.TC2,
                             F_sh_inlet,  Cp_species(op.T_sh, 'H2O'), op.T_sh) #Compute reactor-inlet temperature by adiabatic mixing:cold stream (HE1 cold-out at TC2) + inlet hot steam at Tsh

    # Reactor inlet composition
    Fin = F_cold_to_HE1.copy() #Start from the HE1 cold-side outlet composition (EB+impurities+α steam).
    Fin[IDX['H2O']] += F_sh_inlet #Add the inlet superheated steam to make the true reactor-inlet mixture.

    # Initial state for bed 1, Set the PFR ODE state at z=0:
    y0 = np.zeros(12, dtype=float) #Allocate the ODE state vector [F10,T,P].
    y0[:10] = Fin#first 10 entries = species flows (kmol/h),
    y0[10]  = Tmix1#then temperature Tmix1,
    y0[11]  = op.Pin#then pressure Pin.

    # --- Bed 1 ---
    bed1 = op.bed1 #Select geometry/properties for bed 1.
    f1 = lambda z, y: reactor_odes(z, y, bed1, kin)#Define the ODE right-hand side using bed-1 geometry & kinetics.
    _, y1 = rk4_integrate(f1, 0.0, y0, bed1.L, nstep)#Integrate from z=0 to L₁ with RK4 using nstep steps.

    #Extract bed-1 outlet flows, temperature, and pressure.
    F1_out = y1[-1, :10].copy() #copy() guards against accidental later mutation.
    T1_out = float(y1[-1, 10])
    P1_out = float(y1[-1, 11])



    '''
    the above block
    It routes the superheated steam correctly (all-inlet for SB/DB; split for SI).
    It computes the reactor-inlet temperature T mix1via an energy balance.
    It assembles the inlet state and integrates the PFR ODEs to get the first bed’s outlet, which feeds the rest of the flowsheet (SI injection section or DB second bed, then HE1).
    '''

    if config == 'SB':#in the single-bed case: one packed bed, no mid-bed mixing, no second bed
        F_out, T_out, P_out = F1_out, T1_out, P1_out #Set the final reactor outlet to bed-1’s outlet:
        Tmix2 = None #There is no second mix point in SB (no steam injection, no inter-bed reheat), so nothing to report.
        TH1 = T_out #so, Hot-side inlet temperature to HE1 is the reactor outlet:hot stream into HE1 = reactor effluent → TH1=Tout. see fig. 1 Tarafder2005.
        Qcold, TH2, LMTD, A = he1_energy_balance(
            F_cold_vec=F_cold_to_HE1, TC1=TC1_cold, TC2=op.TC2,
            F_hot_vec=F_out,         TH1=TH1,       U=op.U_HE1
        )



    elif config == 'SI': 
        # inject remaining steam at λL
        z_inj = op.lam * bed1.L #Compute the physical injection location inside bed 1: fraction λ of length L.
        idx = min(max(int(round(z_inj / (bed1.L / nstep))), 1), nstep-1) #Convert that z-position to an RK4 step index. Clamp to [1, nstep-1] so we don’t inject exactly at the inlet (0) or outlet (L).
        F_at_inj = y1[idx, :10].copy()#Grab the gas composition and temperature at the injection point before adding steam.
        T_at_inj = float(y1[idx, 10])

        Tmix2 = mix_temperatures(
            F_at_inj.sum(),                 # flow of in-bed gas
            mixture_cp(F_at_inj, T_at_inj), # its Cp at T_at_inj
            T_at_inj,                       # its temperature
            F_sh_inject,                    # steam to inject
            Cp_species(op.T_sh, 'H2O'),     # steam Cp at T_sh
            op.T_sh                         # steam temperature (superheated)
        )


        Fin2 = F_at_inj.copy(); Fin2[IDX['H2O']] += F_sh_inject #Update the composition by actually adding the injected steam moles.
        y0_2 = y1[idx, :].copy(); y0_2[:10] = Fin2; y0_2[10] = Tmix2 #Reset the ODE state at the injection node: new flows and raised T.

        f2 = lambda z, y: reactor_odes(z, y, bed1, kin)#Continue integrating the rest of the bed from z_inj to L.
        _, y2 = rk4_integrate(f2, z_inj, y0_2, bed1.L, nstep - idx)#Use the remaining steps so total steps ≈ nstep.

        #Final reactor outlet after SI injection section.
        F_out = y2[-1, :10].copy()
        T_out = float(y2[-1, 10])
        P_out = float(y2[-1, 11])

        TH1 = T_out#Set HE1 hot-in to the reactor outlet temperature
        Qcold, TH2, LMTD, A = he1_energy_balance(
            F_cold_vec=F_cold_to_HE1, TC1=TC1_cold, TC2=op.TC2,
            F_hot_vec=F_out,         TH1=TH1,       U=op.U_HE1
        )



    else:
        # DB: inter-bed reheat (no mass addition),in DB mode: two beds in series with a reheater between them.
        assert op.Tmix2 is not None and op.bed2 is not None #Safety check: DB requires a reheat setpoint Tmix2 and a second bed geometry bed2
        y0_2 = y1[-1, :].copy(); y0_2[10] = op.Tmix2#Start bed-2 inlet state from bed-1 outlet (y1[-1]). #Inter-bed reheat: set temperature only to the chosen setpoint Tmix2.Why: DB reheater adds heat only (no mass addition), unlike SI.

        bed2 = op.bed2#Use the second bed’s diameter and L/D (it may differ from bed-1).
        f2 = lambda z, y: reactor_odes(z, y, bed2, kin)#PFR ODEs for bed-2 using its geometry + the same kinetics.
        _, y2 = rk4_integrate(f2, 0.0, y0_2, bed2.L, nstep)#Integrate across bed-2 length from a fresh axial coordinate (0 → L₂).Why: each bed has its own length;  treat z locally per bed.


        #Final reactor outlet after bed-2: flows, temperature, pressure.
        F_out = y2[-1, :10].copy()
        T_out = float(y2[-1, 10])
        P_out = float(y2[-1, 11])
        Tmix2 = op.Tmix2 #Keep Tmix2 in the results for reporting/constraints.

        #Prepare HE1: hot-in TH1 is the reactor outlet;cold-in TC1 is the EB feed temperature.
        TH1 = T_out
        Qcold, TH2, LMTD, A = he1_energy_balance(
            F_cold_vec=F_cold_to_HE1, TC1=TC1_cold, TC2=op.TC2,
            F_hot_vec=F_out,         TH1=TH1,       U=op.U_HE1
        )


        '''
        Run the preheater model (HE1):
        # Solve energy balance (find TH2 so Qhot=Qcold).
        # Compute LMTD and area.

        Why: HE1 heats EB(+α steam) using the reactor effluent; its outputs feed HX constraints (10 K approaches) and give heat-duty/area for feasibility.
        '''



    # Objectives (Tarafder 2005): maximize F_ST,out and S_ST (%)
    Fst_out = F_out[IDX['ST']]#Grab styrene outlet flow (kmol/h) from the outlet vector.This is objective J1 to maximize.
    Fst_in  = F0[IDX['ST']] #Styrene in the feed (kmol/h).EB feed has a small ST impurity; we must not count that as produced ST.
    Feb_in  = F0[IDX['EB']] #EB at inlet and outlet (kmol/h).EB consumed = Feb_in - Feb_out. Used in selectivity.
    Feb_out = F_out[IDX['EB']]
    denom = max(1e-12, Feb_in - Feb_out)#Denominator for selectivity; guard with 1e-12 to avoid divide-by-zero or negative due to tiny numerical noise.
    Sst = max(0.0, 100.0 * (Fst_out - Fst_in) / denom) #Selectivity (%) = (styrene formed) / (EB consumed) × 100. Formed = Fst_out - Fst_in (subtract feed impurity). Clamp at 0.0 to avoid tiny negative values from rounding

    #Package everything the optimizer/constraints need:
    return SimulationResult(
        F_out=F_out, T_out=T_out, P_out=P_out,
        Fst_out=Fst_out, Sst=Sst, Tmix1=Tmix1, Tmix2=Tmix2,
        TH1=TH1, TH2=TH2, TC1=TC1_cold, TC2=op.TC2,
        LMTD_HE1=LMTD, A_HE1=A, Q_HE1=Qcold
    )

# -----------------------------------------------------------------------------
# 7) Bounds, constraints, objective wrapper
# -----------------------------------------------------------------------------

def bounds_for_config(config: str) -> List[Tuple[float, float]]:
    common = [
        (1.4, 2.63),      # Pin [bar]
        (7.0, 20.0),      # SOR [-]
        (27.56, 40.56),   # F0_EB [kmol/h]
        (1.5, 4.0),       # D1 [m]
        (0.7, 1.5),       # L/D1 [-]
        (450.0, 500.0),   # T_EB [K]
        (0.1, 1.0),       # alpha [-]
        (700.0, 900.0),   # TC2 [K]
    ]
    if config == 'SB':
        return common
    if config == 'SI':
        return common + [(0.1, 1.0), (0.1, 1.0)]  # delta, lam
    if config == 'DB':
        return common + [(700.0, 950.0), (1.5, 4.0), (0.7, 1.5)]  # Tmix2, D2, (L/D)2
    raise ValueError("unknown config")

def dv_to_params(config: str, dv: np.ndarray) -> OperatingParams:#Takes a decision vector dv (flat array of numbers from the optimizer) and converts it into a structured OperatingParams object, tailored to the requested config (SB, SI, or DB).


    expected = len(bounds_for_config(config))
    assert dv.size == expected, (
        f"Expected {expected} decision variables for {config}, got {dv.size}"
    )

    
    i = 0#Pointer/index into dv. We’ll read sequential chunks and advance i as we go.
    Pin, SOR, F0_EB = dv[i], dv[i+1], dv[i+2]; i += 3#Read the first 3 entries: inlet pressure, steam/EB ratio, fresh EB.i += 3 advances the pointer so the next reads pick up where we left off.
    D1, L_over_D1   = dv[i], dv[i+1];          i += 2 #Read bed-1 geometry: diameter D1 and slenderness L/D1.
    T_EB, alpha, TC2= dv[i], dv[i+1], dv[i+2]; i += 3 #Read EB feed temperature, premix fraction α, and HE1 cold-out setpoint TC2.
    bed1 = CatalystBed(D=D1, L_over_D=L_over_D1) #Build a CatalystBed object for the first bed, so downstream code can access bed1.D, bed1.L, bed1.area,

    if config == 'SB': #Single-Bed case uses only the common variables.
        return OperatingParams(Pin=Pin, SOR=SOR, F0_EB=F0_EB, bed1=bed1,
                               T_EB=T_EB, alpha=alpha, TC2=TC2)
    if config == 'SI':#For Steam-Injection, pull two extra variables from dv
        delta, lam = dv[i], dv[i+1]; i += 2
        return OperatingParams(Pin=Pin, SOR=SOR, F0_EB=F0_EB, bed1=bed1,
                               T_EB=T_EB, alpha=alpha, TC2=TC2,
                               delta=delta, lam=lam)
    if config == 'DB':#For Double-Bed, read three extra entries
        Tmix2, D2, L_over_D2 = dv[i], dv[i+1], dv[i+2] #Tmix2 (inter-bed reheat setpoint),D2 and L_over_D2 (second-bed geometry).
        bed2 = CatalystBed(D=D2, L_over_D=L_over_D2) #Build a second bed object for the DB configuration.
        return OperatingParams(Pin=Pin, SOR=SOR, F0_EB=F0_EB, bed1=bed1,
                               T_EB=T_EB, alpha=alpha, TC2=TC2,
                               Tmix2=Tmix2, bed2=bed2) #Return OperatingParams including DB-only pieces (reheat setpoint and bed-2 geometry).
    raise ValueError("unknown config")


def constraint_violation(config: str, op: OperatingParams, sim: SimulationResult) -> float:
    """
    Dimensionless, normalized hinge penalties. 0 = feasible.
    Each term is divided by its limit/band to balance units.
    """

    viol = 0.0

    # --- Tmix1 window: 650–925 K ---
    viol += max(0.0, (650.0 - sim.Tmix1) / 650.0)
    viol += max(0.0, (sim.Tmix1 - 925.0) / 925.0)

    # --- Tmix2 windows (only if present) ---
    if config == 'SI' and sim.Tmix2 is not None:
        viol += max(0.0, (650.0 - sim.Tmix2) / 650.0)
        viol += max(0.0, (sim.Tmix2 - 925.0) / 925.0)

    #if config == 'DB' and sim.Tmix2 is not None:
        #viol += max(0.0, (700.0 - sim.Tmix2) / 700.0)
        #viol += max(0.0, (sim.Tmix2 - 950.0) / 950.0)

    # --- Exit pressure: P_out ≥ 1.4 bar ---
    viol += max(0.0, (1.4 - sim.P_out) / 1.4)

    # --- HE1 approach: TH1 - TC2 ≥ 10 K ---
    viol += max(0.0, (10.0 - (sim.TH1 - sim.TC2)) / 10.0)
    # NOTE: TH2−TC1 ≥ 10 K is already enforced inside he1_energy_balance.

    # --- Steam cap: SOR*F0_EB ≤ 453.59 kmol/h ---
    F0_stm = op.SOR * op.F0_EB
    viol += max(0.0, (F0_stm - 453.59) / 453.59)

    return viol



def evaluate(config: str, dv: np.ndarray, kin: Kinetics, nstep: int = 250) -> Tuple[np.ndarray, float]:#Wraps one design point into a simulation + objective/constraint values.
    op = dv_to_params(config, dv)#Decode the flat decision vector into structured OperatingParams

    '''
    BELOW:
    Run the flowsheet (mixers + HE1 + bed(s)).

    If anything blows up (bad physics, ODE failure), return terrible scores:

    big objectives [1e6, 1e6] and big constraint 1e6.

    This penalizes invalid points so the optimizer avoids them.
    '''
    
    
    try:
        sim = simulate_reactor(config, op, kin, nstep=nstep)
    except Exception:
        return np.array([1e6, 1e6]), 1e6

    # Convert maximization to minimization via 1/(1+J)
    #map each J (larger is better) to f = 1/(1+J) (smaller is better).
    f1 = 1.0 / (1.0 + max(1e-9, sim.Fst_out)) #f1 from styrene flow Fst_out
    f2 = 1.0 / (1.0 + max(1e-9, sim.Sst)) #f2 from selectivity Sst (%).
    g = constraint_violation(config, op, sim)
    return np.array([f1, f2]), g

# -----------------------------------------------------------------------------
# 8) PVOS-4C / pymoo-style adapter
# -----------------------------------------------------------------------------

import os as _os

class _PVOSProblem:
    def __init__(self, config="DB", kinetics="YEE2003"):#Store the chosen topology (SB/SI/DB) and build a Kinetics object once.
        self.config = config
        self.KIN = Kinetics(kinetics)
        self.bounds = bounds_for_config(config) #Pull decision-variable bounds, count vars/objs/constraints, and set a readable name.
        self.n_var = len(self.bounds) 
        self.n_obj = 2
        self.n_constr = 1
        self.n_ieq = self.n_ineq = 1
        self.name = f"Tarafder-{config}"
        self.xl = np.array([lo for lo, _ in self.bounds], float)#Vector forms of lower/upper bounds; lb/ub aliases for libs that expect those names.
        self.xu = np.array([hi for _, hi in self.bounds], float)
        self.lb = self.xl; self.ub = self.xu
        self._pf_cache = None

    def evaluate(self, X, return_values_of=None, **kwargs):
        X = np.asarray(X, float)
        single = (X.ndim == 1)
        if single: X = X[None, :]

        F_list, G_list = [], []
        for x in X:
            f, gsum = evaluate(self.config, x, self.KIN)
            F_list.append(np.asarray(f, float).ravel())
            G_list.append(np.array([gsum], float))   # G <= 0 feasible

        F = np.vstack(F_list)
        G = np.vstack(G_list)

        if return_values_of is None:
            return F[0] if single else F

        outs = []
        for key in return_values_of:
            k = key.upper()
            if k == 'F': outs.append(F[0] if single else F)
            elif k == 'G': outs.append(G[0] if single else G)
            else: raise ValueError(f"Unsupported return key: {key}")
        return tuple(outs) if len(outs) > 1 else outs[0]

    def decode(self, x):
        return dv_to_params(self.config, np.asarray(x, float))

    def pareto_front(self, n_points: int = 0, seed: int | None = None):
        if _os.getenv("TARAFDER_ESTIMATE_PF", "0") != "1":
            return None
        if self._pf_cache is not None:
            return self._pf_cache
        n_samples = max(800, n_points or 1200)
        rng = np.random.default_rng(seed)
        X = self.xl + rng.random((n_samples, self.n_var)) * (self.xu - self.xl)
        F, G = self.evaluate(X, return_values_of=['F', 'G'])
        feas = (G <= 0).all(axis=1)
        F = F[feas]
        if F.size == 0:
            self._pf_cache = None
            return None
        nd = self._non_dominated(F)
        nd = nd[np.argsort(nd[:, 0])]
        self._pf_cache = nd
        return nd

    @staticmethod
    def _non_dominated(F):
        n = F.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]: continue
            for j in range(n):
                if i == j or not keep[j]: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):  # j dominates i
                    keep[i] = False
                    break
        return F[keep]

def pvos_problem(config: str = "DB", kinetics: str = "YEE2003"):
    return _PVOSProblem(config, kinetics)

_DEFAULT_CONFIG   = _os.getenv("TARAFDER_CONFIG", "DB")
_DEFAULT_KINETICS = _os.getenv("TARAFDER_KINETICS", "YEE2003")

problem     = pvos_problem(_DEFAULT_CONFIG, _DEFAULT_KINETICS)
problem_SB  = pvos_problem("SB", _DEFAULT_KINETICS)
problem_SI  = pvos_problem("SI", _DEFAULT_KINETICS)
problem_DB  = pvos_problem("DB", _DEFAULT_KINETICS)

__all__ = [
    "CatalystBed", "OperatingParams", "Kinetics", "simulate_reactor",
    "bounds_for_config", "dv_to_params", "evaluate",
    "pvos_problem", "problem", "problem_SB", "problem_SI", "problem_DB"
]

