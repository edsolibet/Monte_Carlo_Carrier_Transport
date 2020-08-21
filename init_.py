# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:31:29 2020

@author: Master
"""
from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import maxwell



class Layer:
    """
    Layer of semiconductor with the following properties...
    
    matl = a material (an object with Material class)
    
    n_or_p = a string, either 'n', 'p' or 'i', for the doping polarity
    
    doping = density of dopants in cm^-3
    
    thickness = thickness of the layer in nm
    """
    def __init__(self, matl, doping, lx, ly, lz):
        self.matl = matl
        self.doping = doping
        self.lx = lx
        self.ly = ly
        self.lz = lz
        

def where_am_i(layers, dist):
    """
    distance is the distance from the start of layer 0.
    
    layers is a list of each layer; each element should be a Layer object.
    
    Return a dictionary {'current_layer':X, 'distance_into_layer':Y}.
    (Note: X is a Layer object, not an integer index.)
    """
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

class Parameters:
    ''' Simulation parameters '''
    dt = 2E-15 # time step
    pts = 25 # number of time intervals
    
    ''' Physical Constants '''
    inf = float('inf') # infinity
    kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
    m_0 = 9.11E-31 # Electron mass in kg
    q = 1.602E-19 # Electron charge in Coulombs
    hbar = 1.055E-34 # Reduced planck's constant in Joules sec
    hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
    eps0 = 8.854E-12 # Vacuum permittivity 


class GaAs:
    
    EG = 1.424 # energy bandgap
    eps_0 = 13.1*Parameters.eps0
    eps_inf = 10.9*Parameters.eps0
    rho = 5360 
    E_phonon = 0.03536  # phonon energy eV
    vs = 5.22E3 # sound velocity (m/s)
    dope = 1E16 # doping concentration
    alpha = 1.1E6 # absorption coefficient
    tot_scat = 4E14 # total scattering rate
    
    # keys are valley index (0 = Gamma, 1 = L, 2 = X)
    # electron relative mass
    mass = {0: 0.063*Parameters.m_0, 1: 0.170*Parameters.m_0, 2: 0.58*Parameters.m_0}
    # Non parabolicity factor
    nonparab = {0: 0.610, 1: 0.461, 2: 0.204}
    # Acoustic Deformation Potential 
    DA = {0: 7.01, 1: 9.2, 2: 9.0}
    
    # Potential energy difference between valleys (eV) 
    E_val = {"GL": 0.29, "LG": 0.29, "GX": 0.48, "XG": 0.48, 
              "LX": 0.19, "XL": 0.19}
    # Equivalent final valleys
    B = {0: 1, 1: 4, 2: 3}
    
    # Deformation potential eV/cm
    defpot = {"GL": 1.8E8, "LG": 1.8E8, "GX": 10E8, "XG": 10E8, 
              "LX": 1E8, "XL": 1E8, "LL": 1E8, "XX": 10E8}
    
    # Phonon energies (eV)
    EP = {"GL": 0.0278, "LG": 0.0278, "GX": 0.0299, "XG": 0.0299, 
              "LX": 0.0293, "XL": 0.0293, "LL": 0.029, "XX": 0.0299}
 
    
class Device:

    def inv_debye_len(mat):
        ''' Calculates the inverse debye length
        
        Parameters
        ----------
        dope : float
            doping concentration of the material
        mat : class
            material class

        Returns
        -------
        float
            inverse debye length
        '''
        return np.sqrt(Parameters.q * mat.dope * 1E2**3 / (mat.eps_0 * Parameters.kT))
    
    ''' --- Device geometry and material composition --- '''

    layer0 = Layer(matl=GaAs, doping=1E17, lx = 10E-6, ly = 10E-6, lz=3E-6)
    #layer1 = Layer(matl=GaAs, doping=1E17, lx = 10E-6, ly = 10E-6, lz=1E-6)
    layers = [layer0]
    Materials = [layer.matl for layer in layers]
    
    num_carr = 100
    elec_field = np.array([0, 0, -5000]) # Ex, Ey, Ez (V/cm)
    dim = [np.array([layer.lx, layer.ly, layer.lz]) for layer in layers]
    seg = []
    for ndx, layer in enumerate(layers):
        seg.append((dim[ndx]*inv_debye_len(layer.matl)).astype(int))
    
    dl = []    
    for ndx, layer in enumerate(layers):
        dl.append(dim[ndx]/seg[ndx])
    mean_energy = 0.1


def init_energy_df(max_nrg = 2, div = 10000):
    ''' Generates energy discretization database
    Inputs:
        max_nrg : maximum energy in eV
        div : number of divisions/resolution of energy
    '''
    return pd.DataFrame(np.linspace(0, max_nrg, div), columns = ['Ek'])

def init_elec_field():
    ''' Generates initial electric field grid
    
    '''
    tot_seg = np.append(np.max(Device.seg, axis = 0)[:2], np.sum(Device.seg, axis = 0)[2])
    return np.tile(Device.elec_field, np.append(tot_seg.astype(int), 1)) 
    

def init_coords(layers = Device.layers, num_carr = Device.num_carr, mean_nrg = Device.mean_energy):
    #initialize positions (x, y, z) in angstroms
    
    # print ("Initializing carrier positions in nm.")
    #pos_mu, pos_sigma = 0, Device.dim[2]/6
    x = np.random.uniform(0, np.max(Device.dim, axis = 0)[0], int(num_carr))
    y = np.random.uniform(0, np.max(Device.dim, axis = 0)[1], int(num_carr))
    z = np.random.uniform(0, np.max(Device.dim, axis = 0)[2], int(num_carr))
    #z = abs(np.random.normal(pos_mu, pos_sigma, int(num_carr)))
    # print ("Sample position: (%0.3g, %0.3g, %0.3g)" %(x[0], y[0], z[0]))
    
    # initialize wave vectors, x1E9 m^-1
    # print ("Initializing initial energy (eV).")
    mean, scale = mean_nrg, 1E-7
    e = maxwell.rvs(loc = mean * Parameters.kT, scale = scale, size = num_carr)
    
    # print ("Initializing initial wave vectors (m^-1).")
    kx = []
    ky = []
    kz = []
    for ndx, i in enumerate(e):
        mat = layers[where_am_i(layers, z[ndx])['current_layer']].matl
        k = np.sqrt(2*mat.mass[0] * Parameters.q * i/Parameters.hbar**2) 
        alpha = np.random.normal(0, 2*np.pi)
        beta = np.random.normal(0, 2*np.pi)
        kx.append(k*np.cos(alpha)*np.sin(beta))
        ky.append(k*np.sin(alpha)*np.sin(beta))
        kz.append(k*np.cos(beta))
       
    
    coords = np.zeros((num_carr,6))
    # remove for loop
    for i in range(num_carr):
        coords[i][0] = kx[i]
        coords[i][1] = ky[i]
        coords[i][2] = kz[i]
        coords[i][3] = x[i]
        coords[i][4] = y[i]
        coords[i][5] = z[i]
    
    return coords