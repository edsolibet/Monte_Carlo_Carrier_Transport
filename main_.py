# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:26:38 2020

@author: Master
"""
from __future__ import division

import datetime
import logging

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s \
                    - %(message)s')
logging.debug('Start of Program')
startTime = datetime.datetime.now()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import init_
import scatter_
import scipy.stats as stats

''' --- Constants --- '''
inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity

''' --- Parameters --- '''

param = init_.Parameters
dev = init_.Device
mat = dev.Materials
logging.debug('Parameters loaded')

''' --- Inputs --- '''
max_nrg = 2 # eV
div = 10000 # energy discretization
# Generates energy values database
dfEk = init_.init_energy_df(max_nrg, div)

''' --- Initialization --- '''
# init_coords: 1 x 6 x num_carr arrays for kx, ky, kz, x, y, z v
init_coords = init_.init_coords()
coords = np.copy(init_coords)
valley = np.zeros(dev.num_carr).astype(int)
EF = init_.init_elec_field()
scat_tables = [scatter_.calc_scatter_table(dfEk, ndx) for ndx in range(len(dev.layers))] # Need scatter tables for each material
if len(init_coords) == len(valley) and len(init_coords) >0:
    logging.debug('Initial coordinates and valley assignments generated.')

''' --- Functions --- '''

def calc_energy(k, val, mat):
    '''
    Calculates the energy from the given wavevector coordinates k, valley, and material
    Converts the calculated energy to its nonparabolic equivalent
    Searches for the closest energy value in the preconstructed energy discretization
    
    k : list of kx, ky, kz coordinates of a particle
    val : valley assignment of particle
    mat : material class
    
    return : (1) float, nonparabolic energy value found within energy database
             (2) int, index of energy in energy database
    '''   
    E = sum(np.square(k))*(hbar)**2/(2*mat.mass[val]*q)
    if np.isnan(E):
        raise Exception('NaN error: NaN value detected')
    nonparab = mat.nonparab[val]
    E_np = (-1 + np.sqrt(1 + 4*nonparab*E))/(2*nonparab)
    ndx = (dfEk - E_np).abs().idxmin()
    #return dfEk.iloc[ndx].values.tolist()[0][0], ndx.iloc[0]
    return E_np, ndx.iloc[0]
    
def free_flight(G0 = 3E14):
    '''
    Calculates a random free flight duration based on the given total scattering rate G0
    
    G0 : total scattering rate; must be greater than calculated maximum scattering rate 
    
    return: float, free flight duration of single particle
    '''
    r1 = np.random.uniform(0, 1)
    return (-1/G0)*np.log(r1)

def cyclic_boundary(r, dim):
    ''' Performs cyclic boundary conditions for the particles position with respect to 
    device dimensions in x- and y- directions
    
    r : position coordinates
    dim : device dimensions
    
    return: array, r with applied cyclic boundary conditions
    '''
    r[:2] = r[:2] - dim[:2]*(r[:2] >= dim[:2]) + dim[:2]*(r[:2]<=0)
    if False in (coords[0][3:5] <= dim[:2]) == (coords[0][3:5] >= 0).all():
        raise Exception('Invalid particle position')
    else:
        return r

def specular_refl(k, r, dim):
    ''' Calculates the z- components of position and wavevector when specular reflection occurs
    
    k : wavevectors
    r : positions
    dim : total device dimensions
    
    return (1) : array, updated z- wavevector
           (2) : array, updated z- position
    '''
    #needs total dim instead of material dim
    while not (r[2] >= 0 and r[2] <= dim[2]):
        k[2] = -1*(r[2] > dim[2] or r[2] < 0)*k[2] + (r[2] > 0 and r[2] < dim[2])*k[2]
        r[2] = 2*dim[2]*(r[2] >= dim[2]) + -1*(r[2] >= dim[2] or r[2] <= 0)*r[2] + (r[2] > 0 and r[2] < dim[2])*r[2]   
    return k, r

def where_am_i(layers, dist):
    '''
    distance is the distance from the start of layer 0.
    
    layers is a list of each layer; each element should be a Layer object.
    
    Return a dictionary {'current_layer':X, 'distance_into_layer':Y}.
    (Note: X is a Layer object, not an integer index.)
    '''
    if dist < 0:
        raise ValueError('Point is outside all layers')
    layer_index = 0
    while layer_index <= (len(layers) - 1):
        current_layer = layers[layer_index]
        if dist <= current_layer.lz:
            return {'current_layer': layer_index,
                    'dist': dist}
        else:
            dist -= current_layer.lz
            layer_index+=1
    raise ValueError('Point is outside all layers. Distance = ' + str(dist))

def carrier_drift(coord, elec_field, dt2, mass):
    '''
    Performs the carrier drift sequence for each set of coordinates in a given electric field
    at a particular time duration dt2 of each particle
    Takes into account the mass of the particle in a given valley
    
    coord : wavevector and position coordinates (kx, ky, kz, x, y, z)
    elec_field : electric field magnitudes in each direction
    dt2 : drift duration
    mass: particle mass in a given valley
    
    return: list, updated coordinates (kx, ky, kz, x, y, z)
    '''
    
    k = coord[:3]
    r = coord[3:]
    
    k_ = k - q*dt2*elec_field*1E2/hbar
    r_ = r + (hbar*k - 0.5*q*elec_field*1e2*dt2)*(dt2/mass)
    ndx = where_am_i(dev.layers, r_[2])['current_layer']
    # checks if cyclic boundary conditions are satisfied
    r_ = cyclic_boundary(r_, dev.dim[ndx])
    # checks if specular reflection is satisfied
    tot_dim = np.append(np.max(dev.dim, axis = 0)[:2], np.sum(dev.dim, axis = 0)[2])
    k_, r_ = specular_refl(k_, r_, tot_dim)
    
    coord[:3] = k_
    coord[3:] = r_
    
    # Value check
    if True in r_ > 1:
        raise Exception('Invalid position coordinates')
    if True in np.abs(k_) < 1e5:
        raise Exception('Invalid wavevector coordinates')
    return coord

''' --- Main Monte Carlo Transport Sequence --- '''
# Generate quantity list placeholders num_carr (rows) x pts (columns)
t_range = [i * param.dt for i in range(param.pts)]
v_hist = np.zeros((dev.num_carr, param.pts))
x_hist = np.zeros((dev.num_carr, param.pts))
y_hist = np.zeros((dev.num_carr, param.pts))
z_hist = np.zeros((dev.num_carr, param.pts))
e_hist = np.zeros((dev.num_carr, param.pts))
val_hist = np.zeros((dev.num_carr, param.pts))

# Generate initial free flight durations for all particles
dtau = [free_flight(1E15) for i in range(dev.num_carr)]

for c, t in enumerate(t_range):
    logging.debug('Time: %0.4g' %t)
    # Carrier photoexcitation 
    
    # Start transport sequence, iterate over each particle
    for carr in range(dev.num_carr):
        dte = dtau[carr]
        # time of flight longer than time interval dt
        if dte >= param.dt:
            dt2 = param.dt
            # get elec field at coordinate
            mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
            drift_coords = carrier_drift(coords[carr], dev.elec_field, dt2, mat[mat_i].mass[valley[carr]])
        # time of flight shorter than time interval dt
        else:
            dt2 = dte
            # get elec field at coordinate
            mat_i = where_am_i(dev.layers, coords[carr][5])['current_layer']
            drift_coords = carrier_drift(coords[carr], dev.elec_field, dt2, mat[mat_i].mass[valley[carr]])
            # iterate free flight until approaching dt
            while dte < param.dt:
                dte2 = dte
                mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
                drift_coords, valley[carr] = scatter_.scatter(drift_coords, scat_tables[mat_i][valley[carr]], valley[carr], dfEk, mat_i)
                dt3 = free_flight(mat[0].tot_scat)
                # Calculate remaining time dtp before end of interval dt
                dtp = param.dt - dte2
                if dt3 <= dtp: # free flight after scattering is less than dtp
                    dt2 = dt3
                else: # free flight after scattering is longer than dtp
                    dt2 = dtp
                # get elec field at coordinate
                mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
                drift_coords = carrier_drift(drift_coords, dev.elec_field, dt2, mat[mat_i].mass[valley[carr]])
                dte = dte2 + dt3
        dte -= param.dt
        dtau[carr] = dte
        coords[carr] = drift_coords
        mat_i = where_am_i(dev.layers, drift_coords[5])['current_layer']
        e_hist[carr][c] = calc_energy(drift_coords[0:3], valley[carr], mat[mat_i])[0]
        v_hist[carr][c] = drift_coords[2]*param.hbar/mat[mat_i].mass[valley[carr]]
        
''' --- Plots --- '''
fig1, ax1 = plt.subplots()
ax1.plot(np.array(t_range)*1E12, v_hist.mean(axis=0))
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Mean Velocity (m/s)')
ax1.set_xlim([0, t_range[-1]*1E12])

fig2, ax2 = plt.subplots()
ax2.plot(np.array(t_range)*1E12, e_hist.mean(axis = 0))
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Mean Energy (eV)')
ax2.set_xlim([0, t_range[-1]*1E12])