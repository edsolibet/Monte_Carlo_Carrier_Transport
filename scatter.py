# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:11:19 2020

@author: Carlo
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from Material import GaAs
import func, init

#logging.basicConfig(level = logging.DEBUG)

inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
Q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity 
#Q = init.Q
N = init.N

''' Scatter Data Table '''

# electron relative mass
mG = 0.063*m_0
mL = 0.170*m_0
mX = 0.58*m_0

# Non parabolicity factor
npG = 0.62
npL = 0.5
npX = 0.3

# Acoustic Deformation Potential
DA_G = 7.01
DA_L = 9.2
DA_X = 9.0

# Potential energy difference between valleys (eV)
E_GL = 0.29
E_LG = E_GL
E_GX = 0.48
E_XG = E_GX
E_LX = E_GX - E_GL
E_XL = E_LX

# Equivalent final valleys
B_G = 1
B_L = 4
B_X = 3

E_phonon = 0.03536
vs = 5.22E3 # sound velocity (m/s)

# Deformation potential eV/cm
defpot_GL = 1.8E8
defpot_GX = 10E8
defpot_LG = 1.8E8
defpot_LL = 5E8
defpot_LX = 1E8
defpot_XG = 10E8
defpot_XL = 1E8
defpot_XX = 10E8

# Phonon energies (eV)
EP_GL = 0.0278
EP_GX = 0.0299
EP_LG = 0.0278
EP_LL = 0.029
EP_LX = 0.0293
EP_XG = 0.0299
EP_XL = 0.0293
EP_XX = 0.0299


''' Scattering rates '''

# Acoustic Deformation Potential Sccattering
def ADP_G(Ek, N, m):
    const = np.pi * DA_G**2 * kT / (hbar_eVs * vs**2 * GaAs.rho)
    return const * func.density_of_states(Ek, m) * 1E9

def ADP_L(Ek, N, m):
    const = np.pi * DA_L**2 * kT / (hbar_eVs * vs**2 * GaAs.rho)
    return const * func.density_of_states(Ek, m) * 2E9

def ADP_X(Ek, N, m):
    const = np.pi * DA_X**2 * kT / (hbar_eVs * vs**2 * GaAs.rho)
    return const * func.density_of_states(Ek, m) * 2E9
    

# Polar optical phonon scattering Gamma valley to L valley
# Spherical parabolic bands
def POP_abs(Ek, N, m):
    eps_p = 1/(1/GaAs.eps_inf - 1/GaAs.eps_0)
    k = np.sqrt(2*Q*Ek*m)/hbar
    p_abs = func.Bose_Einstein(E_phonon)*np.arcsinh(np.sqrt(Ek/E_phonon))
    return (Q**2 * E_phonon * k / (8 * np.pi * eps_p * hbar * (Ek + 1E-6))) * (p_abs) * 2

def POP_ems(Ek, N, m):
    eps_p = 1/(1/GaAs.eps_inf - 1/GaAs.eps_0)
    k = np.sqrt(2*Q*Ek*m)/hbar
    p_ems = (Ek>E_phonon)*(func.Bose_Einstein(E_phonon) + 1) * \
        np.arcsinh(np.sqrt(abs(Ek/E_phonon - 1)))
    return (Q**2 * E_phonon * k / (8 * np.pi * eps_p * hbar * (Ek + 1E-6))) * (p_ems) * 2.25
    
# Intervalley Phonon Scattering

# Gamma to L valley absorption
# Spherical parabolic bands
def IV_GL_abs(Ek, N, m = mG): 
    const = (np.pi*(defpot_GL)**2 * B_L) / (GaAs.rho * EP_GL/hbar_eVs)
    p_abs =  (Ek > (E_GL - EP_GL))*func.density_of_states(abs(Ek + EP_GL - E_GL), m) * \
            func.Bose_Einstein(EP_GL)
    return const*(p_abs)*5E15 #5E13

# Gamma to L valley emission
# Spherical parabolic bands
def IV_GL_ems(Ek, N, m = mG):
    const = (np.pi*(defpot_GL)**2 * B_L) / (GaAs.rho * EP_GL/hbar_eVs)
    p_ems = (Ek > (E_GL + EP_GL))*func.density_of_states(abs(Ek - EP_GL - E_GL), m) * \
            (func.Bose_Einstein(EP_GL) + 1)
    return const*(p_ems)*5E15 #5E13

# Gamma to X valley absorption
# Spherical parabolic bands
def IV_GX_abs(Ek, N, m = mG): 
    const = (np.pi*(defpot_GX)**2 * B_X) / (GaAs.rho * EP_GX/hbar_eVs)
    p_abs =  (Ek > (E_GX - EP_GX))*func.density_of_states(abs(Ek + EP_GX - E_GX), m) * \
            func.Bose_Einstein(EP_GX)
    return const*(p_abs)*1.5E14

# Gamma to X valley emission
# Spherical parabolic bands
def IV_GX_ems(Ek, N, m = mG):
    const = (np.pi*(defpot_GX)**2 * B_X) / (GaAs.rho * EP_GX/hbar_eVs)
    p_ems = (Ek > (E_GX + EP_GX))*func.density_of_states(abs(Ek - EP_GX - E_GX), m) * \
            (func.Bose_Einstein(EP_GX) + 1)
    return const*(p_ems)*1.75E14

# L to Gamma valley absorption
# Spherical parabolic bands
def IV_LG_abs(Ek, N, m = mL):
    const = (np.pi*(defpot_LG)**2 * B_G) / (GaAs.rho * EP_LG/hbar_eVs)
    p_abs = func.density_of_states(abs(Ek + EP_LG + E_LG), m) * func.Bose_Einstein(EP_LG)
    return const*(p_abs)*2E12

# L to Gamma valley emission
# Spherical parabolic bands
def IV_LG_ems(Ek, N, m = mL):
    const = (np.pi*(defpot_LG)**2 * B_G) / (GaAs.rho * EP_LG/hbar_eVs)
    p_ems = func.density_of_states(abs(Ek - EP_LG + E_LG), m) * \
            (func.Bose_Einstein(EP_LG) + 1)
    return const*(p_ems)*2E12

# L to L valley absorption
# Spherical parabolic bands
def IV_LL_abs(Ek, N, m = mL):
    const = (np.pi*(defpot_LL)**2 * B_L) / (GaAs.rho * EP_LL/hbar_eVs)
    p_abs = func.density_of_states(abs(Ek + EP_LL), m) * func.Bose_Einstein(EP_LL)
    return const*(p_abs)*1E13

# L to L valley emission
# Spherical parabolic bands
def IV_LL_ems(Ek, N, m = mL):
    const = (np.pi*(defpot_LL)**2 * B_L) / (GaAs.rho * EP_LL/hbar_eVs)
    p_ems = (Ek > EP_LL)*func.density_of_states(abs(Ek - EP_LL), m) * (func.Bose_Einstein(EP_LL) + 1)
    return const*(p_ems)*1E13

# L to X valley absorption
# Spherical parabolic bands
def IV_LX_abs(Ek, N, m = mL):
    const = (np.pi*(defpot_LX)**2 * B_X) / (GaAs.rho * EP_LX/hbar_eVs)
    p_abs = (Ek > (E_LX - EP_LX))*func.density_of_states(abs(Ek + EP_LX - E_LX), m) *\
    func.Bose_Einstein(EP_LX)
    return const*(p_abs)*4E13

# L to X valley emission
# Spherical parabolic bands
def IV_LX_ems(Ek, N, m = mL):
    const = (np.pi*(defpot_LX)**2 * B_X) / (GaAs.rho * EP_LX/hbar_eVs)
    p_ems = (Ek > (E_LX + EP_LX))*func.density_of_states(abs(Ek - EP_LX - E_LX), m) *\
    (func.Bose_Einstein(EP_LX) + 1)
    return const*(p_ems)*4E13

# X to G valley absorption
# Spherical parabolic bands
def IV_XG_abs(Ek, N, m = mX):
    const = (np.pi*(defpot_XG)**2 * B_G) / (GaAs.rho * EP_XG/hbar_eVs)
    p_abs = func.density_of_states(abs(Ek + EP_XG + E_XG), m) * func.Bose_Einstein(EP_XG)
    return const*(p_abs)*3.5E10

# X to G valley emission
# Spherical parabolic bands
def IV_XG_ems(Ek, N, m = mX):
    const = (np.pi*(defpot_XG)**2 * B_G) / (GaAs.rho * EP_XG/hbar_eVs)
    p_ems = func.density_of_states(abs(Ek - EP_XG + E_XG), m) * \
            (func.Bose_Einstein(EP_XG) + 1)
    return const*(p_ems)*2.5E10

# X to L valley absorption
# Spherical parabolic bands
def IV_XL_abs(Ek, N, m = mX):
    const = (np.pi*(defpot_XL)**2 * B_L) / (GaAs.rho * EP_XL/hbar_eVs)
    p_abs = func.density_of_states(abs(Ek + EP_XL + E_XL), m) * func.Bose_Einstein(EP_XL)
    return const*(p_abs)*2E13

# X to L valley emission
# Spherical parabolic bands
def IV_XL_ems(Ek, N, m = mX):
    const = (np.pi*(defpot_XL)**2 * B_L) / (GaAs.rho * EP_XL/hbar_eVs)
    p_ems = func.density_of_states(abs(Ek - EP_XL + E_XL), m) * \
            (func.Bose_Einstein(EP_XL) + 1)
    return const*(p_ems)*1.6E13

# X to X valley absorption
# Spherical parabolic bands
def IV_XX_abs(Ek, N, m = mX):
    const = (np.pi*(defpot_XX)**2 * B_X) / (GaAs.rho * EP_XX/hbar_eVs)
    p_abs = func.density_of_states(abs(Ek + EP_XX), m) * func.Bose_Einstein(EP_XX)
    return const*(p_abs)*5E12

# L to L valley emission
# Spherical parabolic bands
def IV_XX_ems(Ek, N, m = mX):
    const = (np.pi*(defpot_XX)**2 * B_L) / (GaAs.rho * EP_XX/hbar_eVs)
    p_ems = (Ek > EP_XX)*func.density_of_states(abs(Ek - EP_XX), m) * (func.Bose_Einstein(EP_XX) + 1)
    return const*(p_ems)*4E12


# Ionized Impurity Scattering
# Spherical parabolic bands
def IMP(Ek, N, m):
    Ld = 1/func.inv_debye_length(N)
    k = np.sqrt(2*Q*Ek*m)/hbar
    return (2* np.pi * N * Q**2/ (hbar_eVs * GaAs.eps_0**2)) * func.density_of_states(Ek, m) * \
        (1/Ld**2) * (1 / (1/Ld**2 + 4*k**2)) * 1E2 # correction factor

# Alternative equations
#def IMP(Ek):
#    N = 1E17
#    Ld = 1/func.inv_debye_length(N)
#    b = (hbar**2 / (2*GaAs.m_G * Ld**2))
#    coeff =  (2**(2.5) * np.pi * N * q**4) / (np.sqrt(GaAs.m_G) * GaAs.eps_0**2)
#    return coeff * np.sqrt(Ek) / (b**2 * (1 + 4*(Ek/b))) * 1E13

#def IMP(Ek):
#    N = 1E17
#    Ld = 1/func.inv_debye_length(N)
#    g = 8*GaAs.m_G * Ek * Ld**2 / hbar**2
#    coeff = N * q**4/ (np.sqrt(512*GaAs.m_G) * np.pi * hbar * GaAs.eps_0**2)
#    return coeff * (np.log(1 + g) - g/(1+g)) / (Ek+1E-6)**1.5
        
# Carrier-Plasma scattering
# Spherical parabolic bands
def cc(Ek, N, m):
    E_plasma = func.plasmon_freq(N, m)*hbar_eVs
    k = np.sqrt(2*Q*Ek*m)/hbar
    coeff = (Q**2 * E_plasma * k / (8 * np.pi * GaAs.eps_0 * hbar * (Ek + 1E-6)))
    p_abs = func.Bose_Einstein(E_plasma)*np.arcsinh(np.sqrt(Ek/E_plasma))
    p_ems = (Ek>E_plasma)*(func.Bose_Einstein(E_plasma) + 1) * \
        np.arcsinh(np.sqrt(abs(Ek/E_plasma - 1)))
    return  coeff * (p_abs + p_ems)

# Total scattering rate
def total_scatter_G(Ek, dope = 1E16, m = mG):
    return ADP_G(Ek, dope, m) + POP_abs(Ek, dope, m) + POP_ems(Ek, dope, m) + IV_GL_abs(Ek, dope, m) + \
        IV_GL_ems(Ek, dope, m) + IV_GX_abs(Ek, dope, m) + IV_GX_ems(Ek, dope, m) + \
        IMP(Ek, dope, m) + cc(Ek, dope, m)

# Total scattering rate
def total_scatter_L(Ek, dope = 1E16, m = mL):
    return ADP_L(Ek, dope, m) + POP_abs(Ek, dope, m) + POP_ems(Ek, dope, m) + \
        IV_LG_abs(Ek, dope, m) + IV_LG_ems(Ek, dope, m) + IV_LL_abs(Ek, dope, m) + \
        IV_LL_ems(Ek, dope, m) + IV_LX_abs(Ek, dope, m) + IV_LX_ems(Ek, dope, m) + \
        IMP(Ek, dope, m) + cc(Ek, dope, m)
        
# Total scattering rate
def total_scatter_X(Ek, dope = 1E16, m = mX):
    return ADP_X(Ek, dope, m) + POP_abs(Ek, dope, m) + POP_ems(Ek, dope, m) + \
        IV_XG_abs(Ek, dope, m) + IV_XG_ems(Ek, dope, m) + IV_XL_abs(Ek, dope, m) + \
        IV_XL_ems(Ek, dope, m) + IV_XX_abs(Ek, dope, m) + IV_XX_ems(Ek, dope, m) + \
        IMP(Ek, dope, m) + cc(Ek, dope, m)

# Self-scattering Rate
def self_scatter_G(Ek, dope=1E16, m = mG, T = 1E15):
    return T - total_scatter_G(Ek, dope, m)

# Self-scattering Rate
def self_scatter_L(Ek, dope=1E16, m = mL, T = 1E15):
    return T - total_scatter_L(Ek, dope, m)

# Self-scattering Rate
def self_scatter_X(Ek, dope=1E16, m = mX, T = 1E15):
    return T - total_scatter_X(Ek, dope, m)

    
def iso_costheta(Ek, N):
    return np.arccos(1 - 2*np.random.uniform(low = 0, high = 1))

def pop_abs_costheta(Ek, N):
    E_phonon = 0.03536
    eta = 2*np.sqrt(Ek*(Ek + E_phonon))/(np.sqrt(Ek) - np.sqrt(Ek + E_phonon))**2
    r = np.random.uniform(0, 1)
    return np.arccos(((1 + eta) - (1 + 2*eta)**r)/eta)

def pop_ems_costheta(Ek, N):
    E_phonon = 0.03536
    eta = 2*np.sqrt(Ek*(Ek - E_phonon))/(np.sqrt(Ek) - np.sqrt(Ek - E_phonon))**2
    r = np.random.uniform(0, 1)
    return np.arccos(((1 + eta) - (1 + 2*eta)**r)/eta)

#def ion_costheta(E, N):
#    k = np.sqrt(2*GaAs.m_G*q*E/hbar**2)
#    r = np.random.uniform(low = 0, high = 1)
#    LD = 1/func.inv_debye_length(N)
#    return np.arccos(1 - 2*r/(1 + 4 * (k*LD)**2 * (1 - r)))

def ion_costheta(Ek, N):
    b = (1/(2*N))**(1/3)
    eb = Q/(2*GaAs.eps_0*b)
    alpha = 2*np.arctan(eb/(Ek+1E-6))
    r = np.random.uniform(low = 0, high = 1)
    return np.arccos(1-(1 - np.cos(alpha))/(1 - r*(1-0.5*(1-np.cos(alpha)))))

def self_scat_angle(Ek, N):
    return 0


E_plasma_G = func.plasmon_freq(N, mG)*hbar_eVs
E_plasma_L = func.plasmon_freq(N, mL)*hbar_eVs
E_plasma_X = func.plasmon_freq(N, mX)*hbar_eVs
G_scat = {"ADP G": {'mech' : ADP_G, 'dE': E_phonon, 'polar': iso_costheta},
        "POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta},
        "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta},
        "IV_GL Absorption": {'mech': IV_GL_abs, 'dE': EP_GL - E_GL, 'polar': iso_costheta},
        "IV_GL Emission": {'mech': IV_GL_ems, 'dE': -EP_GL - E_GL, 'polar': iso_costheta},
        "IV_GX Absorption": {'mech': IV_GX_abs, 'dE': EP_GX - E_GX, 'polar': iso_costheta},
        "IV_GX Emission": {'mech': IV_GX_ems, 'dE': -EP_GX - E_GX, 'polar': iso_costheta},
        "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta},
        "Carrier-Carrier": {'mech': cc, 'dE': E_plasma_G, 'polar': ion_costheta},
        "Self-Scattering": {'mech' : self_scatter_G, 'dE' : 0, 'polar': self_scat_angle}
        }

L_scat = {"ADP L": {'mech' : ADP_L, 'dE': E_phonon, 'polar': iso_costheta},
        "POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta},
        "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta},
        "IV_LG Absorption": {'mech': IV_LG_abs, 'dE': EP_LG + E_LG, 'polar': iso_costheta},
        "IV_LG Emission": {'mech': IV_LG_ems, 'dE': -EP_LG + E_LG, 'polar': iso_costheta},
        "IV_LL Absorption": {'mech': IV_LL_abs, 'dE': EP_LL, 'polar': iso_costheta},
        "IV_LL Emission": {'mech': IV_LL_ems, 'dE': -EP_LL, 'polar': iso_costheta},
        "IV_LX Absorption": {'mech': IV_LX_abs, 'dE': EP_LX - E_LX, 'polar': iso_costheta},
        "IV_LX Emission": {'mech': IV_LX_ems, 'dE': -EP_LX - E_LX, 'polar': iso_costheta},
        "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta},
        "Carrier-Carrier": {'mech': cc, 'dE': E_plasma_L, 'polar': ion_costheta},
        "Self-Scattering": {'mech' : self_scatter_L, 'dE' : 0, 'polar': self_scat_angle}
        }

X_scat = {"ADP X": {'mech' : ADP_X, 'dE': E_phonon, 'polar': iso_costheta},
        "POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta},
        "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta},
        "IV_XG Absorption": {'mech': IV_XG_abs, 'dE': EP_XG + E_XG, 'polar': iso_costheta},
        "IV_XG Emission": {'mech': IV_XG_ems, 'dE': -EP_XG + E_XG, 'polar': iso_costheta},
        "IV_XL Absorption": {'mech': IV_XL_abs, 'dE': EP_XL + E_XL, 'polar': iso_costheta},
        "IV_XL Emission": {'mech': IV_XL_ems, 'dE': -EP_XL + E_XL, 'polar': iso_costheta},
        "IV_XX Absorption": {'mech': IV_XX_abs, 'dE': EP_XX, 'polar': iso_costheta},
        "IV_XX Emission": {'mech': IV_XX_ems, 'dE': -EP_XX, 'polar': iso_costheta},
        "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta},
        "Carrier-Carrier": {'mech': cc, 'dE': E_plasma_X, 'polar': ion_costheta},
        "Self-Scattering": {'mech' : self_scatter_X, 'dE' : 0, 'polar': self_scat_angle}
        }

param = {0: {'scat': G_scat, 'm': mG, 'np': npG, 'DA': DA_G},
         1: {'scat': L_scat, 'm': mL, 'np': npL, 'DA': DA_L},
         2: {'scat': X_scat, 'm': mX, 'np': npX, 'DA': DA_X}
         }

def plot_scat_rate(scat, elec_energy, N, m):
    y = np.zeros((len(scat.keys()),len(elec_energy)))
    for e, energy in enumerate(elec_energy):
        for s, sc in enumerate(list(scat.keys())):
            y[s][e] = scat[sc]['mech'](energy, N, m)
    tot = y[0:-1].sum(axis = 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(scat.keys()) - 1):
        ax.plot(elec_energy, y[i], label = list(scat.keys())[i])
    ax.plot(elec_energy, tot, label = 'Total Scattering Rate')
    ax.plot(elec_energy, y[-1], label = 'Self Scattering')
    ax.legend()
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(r'Scattering rate $\Gamma$ (s$^{-1}$)')
    #ax2.set_yscale('log')
    
def choose_mech(E, scat, N, m):
    r2 = np.random.uniform(low = 0, high = 1)
    scatter_rate = []
    scatter_mech = []
    T = 0
    for i in scat.keys():
        T += scat[i]['mech'](E, N, m)
        scatter_rate.append(scat[i]['mech'](E,N,m))
        scatter_mech.append(i)
    scatter_rate = np.array(scatter_rate)/T
    #print (T)
    i = 0
    #print (scatter_mech)
    #print (scatter_rate)
    while r2 >= sum(scatter_rate[:i+1]): i+=1
    #print (scatter_mech[i])
    return scatter_mech[i]

# Calculate new k state after scattering
# Determine azimuthal and polar angles after scattering
def scatter_angles(coords, N, scat, mass):
    kx, ky, kz = coords[0:3]
    k0 = np.sqrt(kx**2 + ky**2 + kz**2)
    p0 = np.arccos(kz/k0)
    a0 = np.arcsin(kx/(k0*np.sin(p0)))
    azimuthal = 2*np.pi*np.random.uniform(low = 0, high = 1)
    E0 = func.calc_energy(kx, ky, kz, mass)
    #print ('####')
    #print (E0)
    polar = scat['polar'](E0, N)
    kp = np.sqrt(2*mass*Q*(E0 + scat['dE'])/hbar**2)
    kxr = kp*np.sin(polar)*np.cos(azimuthal)
    kyr = kp*np.sin(polar)*np.sin(azimuthal)
    kzr = kp*np.cos(polar)
    kxp = kxr*np.cos(a0)*np.cos(p0) - kyr*np.sin(a0) + kzr*np.cos(p0)*np.sin(a0)
    kyp = kxr*np.sin(a0)*np.cos(p0) + kyr*np.cos(a0) + kzr*np.sin(p0)*np.sin(a0)
    kzp = -kxr*np.sin(p0) + kzr*np.cos(p0)
    coords[0:3] = [kxp, kyp, kzp]
    return coords

def choose_scat(val, drift_coords, N):
    if val == 0:
        mech = choose_mech(func.calc_energy(*drift_coords[0:3], param[val]['m']), param[val]['scat'], N, param[val]['m'])       
        drift_coords = scatter_angles(drift_coords, N, param[val]['scat'][mech], param[val]['m'])
        if mech in ["IV_GL Absorption", "IV_GL Emission"]:
            print ('#####')
            print (mech)
            val = 1
        elif mech in ["IV_GX Absorption", "IV_GX Emission"]:
            print ('-----')
            print (mech)
            val = 2
        else:
            val = 0
    elif val == 1:
        mech = choose_mech(func.calc_energy(*drift_coords[0:3], param[val]['m']), param[val]['scat'], N, param[val]['m'])         
        drift_coords = scatter_angles(drift_coords, N, param[val]['scat'][mech], param[val]['m'])
        if mech in ["IV_LG Absorption", "IV_LG Emission"]:
            print ('#####')
            print (mech)
            val = 0
        elif mech in ["IV_LX Absorption", "IV_LX Emission"]:
            print ('*****') 
            print (mech)
            val = 2
        else:
            val = 1
    elif val == 2:
        mech = choose_mech(func.calc_energy(*drift_coords[0:3], param[val]['m']), param[val]['scat'], N, param[val]['m'])          
        drift_coords = scatter_angles(drift_coords, N, param[val]['scat'][mech], param[val]['m'])
        if mech in ["IV_XG Absorption", "IV_XG Emission"]:
            print ('-----')
            print (mech)
            val = 0
        elif mech in ["IV_XL Absorption", "IV_XL Emission"]:
            print ('*****')
            print (mech)
            val = 1
        else:
            val = 2 
    return drift_coords, val
            
        

    
