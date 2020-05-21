# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:38:19 2020

@author: Carlo
"""

from __future__ import division
import logging
logging.basicConfig(level = logging.DEBUG)
import datetime
import numpy as np

start_time = datetime.datetime.now()

''' Initialization '''

def init(num_carr):
    # initialize wave vectors, x1E9 m^-1
    
    print ("Initializing wave vectors (m^-1).")
    wv_mu, wv_sigma = 0, 1.5
    kx = np.random.normal(wv_mu, wv_sigma, int(num_carr))*1E8
    ky = np.random.normal(wv_mu, wv_sigma, int(num_carr))*1E8
    kz = np.random.normal(wv_mu, wv_sigma, int(num_carr))*1E8
#    print ("Sample wavevector: (%0.3g, %0.3g, %0.3g)" %(kx[0], ky[0], kz[0]))
    
    #initialize positions (x, y, z) in angstroms
    
    print ("Initializing carrier positions in nm.")
    pos_mu, pos_sigma = 0, 1
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

print("--- %s seconds ---" % (datetime.datetime.now() - start_time).total_seconds())