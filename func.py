# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:51:33 2020

@author: Carlo
"""

from __future__ import division
import numpy as np
import math as m
from Material import GaAs

inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity 

# calculate energy in eV
# to show histogram, extract kx, ky, kz from coords using ki = coords[;,i]
def calc_energy(kx, ky, kz):
    E = (kx**2 + ky**2 + kz**2)*(hbar)**2/(2*GaAs.m_G*q)
    if m.isnan(E):
        print ('kx', kx)
        print ('ky', ky)
        print ('kz', kz)
        raise Exception('NaN error: NaN value detected')
    #print ('Energy: ', E)
    return E

# Bose Einstein distribution
def Bose_Einstein(E):
    return 1/(np.exp(E/kT) - 1)

# Density of states
def density_of_states(E):
    return 2 * (2*GaAs.m_G)**(1.5) * m.sqrt(E*(1 + GaAs.nonparab*E)) * \
            (1 + 2*GaAs.nonparab*E) / (4*np.pi**2 * hbar_eVs**3)
          
def wv(eV):
    return 2*np.pi/(hbar*2*np.pi/np.sqrt(2*eV*q*GaAs.m_G))
            
''' Free flight '''

def free_flight(G0 = 3E14):
    r1 = np.random.uniform(0, 1)
    return (-1/G0)*np.log(r1)

''' Carrier drift '''

def carrier_drift(coord, elec_field, dt2):
    kx, ky, kz, x, y, z = coord
    
    kx1 = kx
    ky1 = ky
    kz1 = kz - q*dt2*elec_field*1E2/hbar
    
    x1 = x + hbar*kx*dt2/GaAs.m_G
    y1 = y + hbar*ky*dt2/GaAs.m_G
    #z1 = z + (calc_energy(kx1, ky1, kz1) - calc_energy(kx, ky, kz))/(-elec_field*1E2)
    z1 = z + (hbar*kz - 0.5*q*elec_field*1E2*dt2)*(dt2/GaAs.m_G)
    
    coord = [kx1, ky1, kz1, x1, y1, z1]    
    return coord

def velocity(coords):
    kx, ky, kz = coords[0:3]
    return (hbar/GaAs.m_G)*np.sqrt(kx**2 + ky**2 + kz**2)

# kT = 1.38E-23 * 300 K = 0.0258 eV * q
# N should be in m^-3 hence 100**3
def inv_debye_length(N):
    return np.sqrt(q * N * 100**3 / (GaAs.eps_0*kT))

def plasmon_freq(N):
    return np.sqrt(q**2 * N * 100**3 / (GaAs.eps_0*GaAs.m_G))



    
    