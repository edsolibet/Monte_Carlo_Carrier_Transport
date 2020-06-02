# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:38:19 2020

@author: Carlo
"""

from __future__ import division
import logging
logging.basicConfig(level = logging.DEBUG)
import numpy as np
from Material import GaAs
from scipy.stats import gamma

kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
Q = 1.602E-19 # Electron charge in Coulombs
N = 1E16
    
''' --- Initialization --- '''

def init_coord(num_carr, mean_nrg = 0.1):
    # initialize wave vectors, x1E9 m^-1
    print ("Initializing initial energy (eV).")
    shape, scale = mean_nrg, kT
    e = gamma.rvs(a = shape, loc = 0, scale = scale, size = num_carr)
    
    print ("Initializing initial wave vectors (m^-1).")
    kx = []
    ky = []
    kz = []
    for i in e:
        r = np.sqrt(2*GaAs.m_G*Q*i/hbar**2) 
        alpha = np.random.normal(0, 2*np.pi)
        beta = np.random.normal(0, 2*np.pi)
        kx.append(r*np.cos(alpha)*np.sin(beta))
        ky.append(r*np.sin(alpha)*np.sin(beta))
        kz.append(r*np.cos(beta))
       
    #initialize positions (x, y, z) in angstroms
    
    print ("Initializing carrier positions in nm.")
    pos_mu, pos_sigma = 0, 0.1
    x = np.random.normal(pos_mu, pos_sigma, int(num_carr))*1E-9
    y = np.random.normal(pos_mu, pos_sigma, int(num_carr))*1E-9
    z = abs(np.random.normal(pos_mu, pos_sigma, int(num_carr)))*1E-9
#    print ("Sample position: (%0.3g, %0.3g, %0.3g)" %(x[0], y[0], z[0]))
    
    coords = np.zeros((num_carr,6))
    for i in range(num_carr):
        coords[i][0] = kx[i]
        coords[i][1] = ky[i]
        coords[i][2] = kz[i]
        coords[i][3] = x[i]
        coords[i][4] = y[i]
        coords[i][5] = z[i]
    
    return coords

#def init_charge_density(lx):
#    dx = lx/64
#    grid = np.zeros(shape = (64, 1))
#    return rho
#
#def init_potential(lx, ly, lz):
#    pot = np.zeros(lx, ly, lz)
#    return pot