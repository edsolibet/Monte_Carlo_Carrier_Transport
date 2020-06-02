# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:30:32 2020

@author: Carlo
"""
from __future__ import division, print_function

import numpy as np

inf = float('inf')

inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity 

class Material:
    """
    Semiconductor material with the following properties.
    
    m_G: Gamma valley effective mass, m_0
    
    m_L: L valley effective mass, m_0
    
    m_h: hole effective mass, m_0
    
    E_G: Band gap, eV
    
    E_LG: L valley offset, eV
    
    LO: LO-phnon energy, eV
    
    rho: mass density
    
    B_L: number of final L valley
    
    B_G: number of final gamma valley
    
    d_L: Deformation potential, eV/angstrom
    
    eps_0: static dielectric constant (epsilon / epsilon0)
    
    eps_inf: dielectric constant at infinity
    
    alpha: absorption coefficient, cm-1
    
    nonparab: non-parabolicity paramater, eV-1
    
    name = a string describing the material (for plot labels etc.)
    """

    def __init__(self, m_G, m_L, m_h, E_G, E_LG, LO, rho, B_L, B_G, d_L, eps_0, eps_inf, alpha, nonparab, name=''):
        self.m_G = m_G*m_0
        self.m_L = m_L*m_0
        self.m_h = m_h*m_0
        self.E_G = E_G
        self.E_LG = E_LG
        self.LO = LO
        self.rho = rho
        self.B_L = B_L
        self.B_G = B_G
        self.d_L = d_L 
        self.eps_0 = eps_0*eps0
        self.eps_inf = eps_inf*eps0
        self.alpha = alpha
        self.nonparab = nonparab
        self.name = name
        

GaAs = Material(m_G = 0.067,
                m_L = 0.35,
                m_h = 0.5,
                E_G = 1.43,
                E_LG = 0.29,
                LO = 0.035,
                rho = 5360,
                B_L = 4,
                B_G = 1,
                d_L = 0.6,
                eps_0=13.1,
                eps_inf=10.9,
                alpha = 1.2E4,
                nonparab = 0.64,
                name='GaAs')


class Layer:
    """
    Layer of semiconductor with the following properties...
    
    matl = a material (an object with Material class)
    
    n_or_p = a string, either 'n' or 'p', for the doping polarity
    
    doping = density of dopants in cm^-3
    
    thickness = thickness of the layer in nm
    """
    def __init__(self, matl, thickness, n_or_p = None, doping = None):
        self.matl = matl
        self.thickness = thickness
        if n_or_p is not None:
            self.n_or_p = n_or_p
        if doping is not None:
            self.doping = doping
            
            
def derivative(x, y):
    der = np.diff(y)/np.diff(x)
    x2 = (x[:-1] + x[1:])*0.5
    return {'y':der, 'x': x2}