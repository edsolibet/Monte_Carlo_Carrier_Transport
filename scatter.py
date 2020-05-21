# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:11:19 2020

@author: Carlo
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from Material import GaAs
import func

#logging.basicConfig(level = logging.DEBUG)

inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity 

''' Scattering rates '''

# Polar optical phonon scattering Gamma valley to L valley
# Spherical parabolic bands
def POP_abs(Ek, N):
    E_phonon = 0.03536
    eps_p = 1/(1/GaAs.eps_inf - 1/GaAs.eps_0)
    k = np.sqrt(2*q*Ek*GaAs.m_G)/hbar
    p_abs = func.Bose_Einstein(E_phonon)*np.arcsinh(np.sqrt(Ek/E_phonon))
    return (q**2 * E_phonon * k / (8 * np.pi * eps_p * hbar * (Ek + 1E-6))) * (p_abs)

def POP_ems(Ek, N):
    E_phonon = 0.03536
    eps_p = 1/(1/GaAs.eps_inf - 1/GaAs.eps_0)
    k = np.sqrt(2*q*Ek*GaAs.m_G)/hbar
    p_ems = (Ek>E_phonon)*(func.Bose_Einstein(E_phonon) + 1) * \
        np.arcsinh(np.sqrt(abs(Ek/E_phonon - 1)))
    return (q**2 * E_phonon * k / (8 * np.pi * eps_p * hbar * (Ek + 1E-6))) * (p_ems)
    
# Intervalley Phonon Scattering

# Gamma to L valley absorption
# Spherical parabolic bands
def IV_GL_abs(Ek, N): 
    Eif = 0.0278
    E_GL = 0.29
    const = (np.pi*(GaAs.d_L*1E8)**2 * GaAs.B_L) / (GaAs.rho * Eif/hbar_eVs)
    p_abs =  (Ek > (E_GL - Eif))*func.density_of_states(abs(Ek + Eif - E_GL)) * \
            func.Bose_Einstein(Eif)
    return const*(p_abs)*1E16

# Gamma to L valley emission
# Spherical parabolic bands
def IV_GL_ems(Ek, N):
    Eif = 0.0278
    E_GL = 0.29
    const = (np.pi*(GaAs.d_L*1E8)**2 * GaAs.B_L) / (GaAs.rho * Eif/hbar_eVs)
    p_ems = (Ek > (E_GL + Eif))*func.density_of_states(abs(Ek - Eif - E_GL)) * \
            (func.Bose_Einstein(Eif) + 1)
    return const*(p_ems)*1E16

# L to Gamma valley absorption
# Spherical parabolic bands
def IV_LG_abs(Ek, N):
    Eif = 0.0278
    E_LG = 0.29
    const = (np.pi*(GaAs.d_L*1E8)**2 * GaAs.B_G) / (GaAs.rho * Eif/hbar_eVs)
    p_abs = func.density_of_states(abs(Ek + Eif + E_LG)) * func.Bose_Einstein(Eif)
    return const*(p_abs)*1E16

# L to Gamma valley emission
# Spherical parabolic bands
def IV_LG_ems(Ek, N):
    Eif = 0.0278
    E_LG = 0.29
    const = (np.pi*(GaAs.d_L*1E8)**2 * GaAs.B_G) / (GaAs.rho * Eif/hbar_eVs)
    p_ems = (Ek > (E_LG - Eif))*func.density_of_states(abs(Ek - Eif + E_LG)) * \
            (func.Bose_Einstein(Eif) + 1)
    return const*(p_ems)*1E16

# Ionized Impurity Scattering
# Spherical parabolic bands
def IMP(Ek, N):
    N = 1E17
    Ld = 1/func.inv_debye_length(N)
    k = np.sqrt(2*q*Ek*GaAs.m_G)/hbar
    return (2* np.pi * N * q**2/ (hbar_eVs * GaAs.eps_0**2)) * func.density_of_states(Ek) * \
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
def cc(Ek, N):
    E_plasma = func.plasmon_freq(N)*hbar_eVs
    k = np.sqrt(2*q*Ek*GaAs.m_G)/hbar
    coeff = (q**2 * E_plasma * k / (8 * np.pi * GaAs.eps_0 * hbar * (Ek + 1E-6)))
    p_abs = func.Bose_Einstein(E_plasma)*np.arcsinh(np.sqrt(Ek/E_plasma))
    p_ems = (Ek>E_plasma)*(func.Bose_Einstein(E_plasma) + 1) * \
        np.arcsinh(np.sqrt(abs(Ek/E_plasma - 1)))
    return  coeff * (p_abs + p_ems)

# Total scattering rate
def total_scatter(Ek, dope = 1E16):
    return POP_abs(Ek, dope) + POP_ems(Ek, dope) + IV_GL_abs(Ek, dope) + \
        IV_GL_ems(Ek, dope)+ IV_LG_abs(Ek, dope) + IV_LG_ems(Ek, dope) + \
        IMP(Ek, dope) + cc(Ek, dope)

# Self-scattering Rate
def self_scatter(Ek, dope=1E16, T = 3E14):
    return T - total_scatter(Ek, dope)

def plot_scat_rate(scat, elec_energy, N):
    y = np.zeros((len(scat.keys()),len(elec_energy)))
    for e, energy in enumerate(elec_energy):
        for s, sc in enumerate(list(scat.keys())):
            y[s][e] = scat[sc]['mech'](energy, N)
    tot = y[0:-1].sum(axis = 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    for i in range(len(scat.keys())):
#        ax.plot(elec_energy, y[i], label = list(scat.keys())[i])
    ax.plot(elec_energy, tot, label = 'Total Scattering Rate')
    ax.plot(elec_energy, y[-1], label = 'Self Scattering')
    ax.legend()
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(r'Scattering rate $\Gamma$ (s$^{-1}$)')
    #ax2.set_yscale('log')
    
def choose_scatter(E, scat, N = 1E16):
    r2 = np.random.uniform(low = 0, high = 1)
    scatter_rate = []
    scatter_mech = []
    T = 0
    for i in scat.keys():
        T += scat[i]['mech'](E, N)
        scatter_rate.append(scat[i]['mech'](E,N))
        scatter_mech.append(i)
    scatter_rate = np.array(scatter_rate)/T
    i = 0
    while r2 >= sum(scatter_rate[:i+1]): i+=1
    #print (scatter_mech[i])
    return scatter_mech[i]
    
def iso_costheta(Ek):
    return np.arccos(1 - 2*np.random.uniform(low = 0, high = 1))

def pop_abs_costheta(Ek):
    E_phonon = 0.03536
    eta = 2*np.sqrt(Ek*(Ek + E_phonon))/(np.sqrt(Ek) - np.sqrt(Ek + E_phonon))**2
    r = np.random.uniform(0, 1)
    return np.arccos(((1 + eta) - (1 + 2*eta)**r)/eta)

def pop_ems_costheta(Ek):
    E_phonon = 0.03536
    eta = 2*np.sqrt(Ek*(Ek - E_phonon))/(np.sqrt(Ek) - np.sqrt(Ek - E_phonon))**2
    r = np.random.uniform(0, 1)
    return np.arccos(((1 + eta) - (1 + 2*eta)**r)/eta)

#def ion_costheta(E, N):
#    k = np.sqrt(2*GaAs.m_G*q*E/hbar**2)
#    r = np.random.uniform(low = 0, high = 1)
#    LD = 1/func.inv_debye_length(N)
#    return np.arccos(1 - 2*r/(1 + 4 * (k*LD)**2 * (1 - r)))

def ion_costheta(Ek, N = 1E16):
    b = (1/(2*N))**(1/3)
    eb = q/(2*GaAs.eps_0*b)
    alpha = 2*np.arctan(eb/(Ek+1E-6))
    r = np.random.uniform(low = 0, high = 1)
    return np.arccos(1-(1 - np.cos(alpha))/(1 - r*(1-0.5*(1-np.cos(alpha)))))

def self_scat_angle(Ek):
    return 0

E_phonon = 0.03536
Eif = 0.0278 # intervalley phonon energy / hbar_eVs
E_GL = 0.29
scat = {"POP Absorption": {'mech' : POP_abs, 'dE': E_phonon, 'polar': pop_abs_costheta},
        "POP Emission": {'mech': POP_ems, 'dE': -E_phonon, 'polar': pop_ems_costheta},
        "IV_GL Absorption": {'mech': IV_GL_abs, 'dE': Eif, 'polar': iso_costheta},
        "IV_GL Emission": {'mech': IV_GL_ems, 'dE': -Eif, 'polar': iso_costheta},
        "IV_LG Absorption": {'mech': IV_LG_abs, 'dE': Eif, 'polar': iso_costheta},
        "IV_LG Emission": {'mech': IV_LG_ems, 'dE': -Eif, 'polar': iso_costheta},
        "Ionized Impurity": {'mech': IMP, 'dE': 0, 'polar': ion_costheta},
        "Carrier-Carrier": {'mech': cc, 'dE': 0, 'polar': ion_costheta},
        "Self-Scattering": {'mech' : self_scatter, 'dE' : 0, 'polar': self_scat_angle}
        }

# Calculate new k state after scattering
# Determine azimuthal and polar angles after scattering
def scatter_angles(coords, N, scat):
    kx, ky, kz = coords[0:3]

    k0 = np.sqrt(kx**2 + ky**2 + kz**2)
    p0 = np.arccos(kz/k0)
    a0 = np.arcsin(kx/(k0*np.sin(p0)))
    azimuthal = 2*np.pi*np.random.uniform(low = 0, high = 1)
    E0 = func.calc_energy(kx, ky, kz)
    polar = scat['polar'](E0)
    kp = np.sqrt(2*GaAs.m_G*q*(E0 + scat['dE'])/hbar**2)
    kxr = kp*np.sin(polar)*np.cos(azimuthal)
    kyr = kp*np.sin(polar)*np.sin(azimuthal)
    kzr = kp*np.cos(polar)
    kxp = kxr*np.cos(a0)*np.cos(p0) - kyr*np.sin(a0) + kzr*np.cos(p0)*np.sin(a0)
    kyp = kxr*np.sin(a0)*np.cos(p0) + kyr*np.cos(a0) + kzr*np.sin(p0)*np.sin(a0)
    kzp = -kxr*np.sin(p0) + kzr*np.cos(p0)
    coords[0:3] = [kxp, kyp, kzp]
    return coords

    
