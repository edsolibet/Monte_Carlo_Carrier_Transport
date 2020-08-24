# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 01:06:56 2020

@author: Master
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from timeit import default_timer as timer
import init_
from mpl_toolkits.mplot3d import Axes3D

dev = init_.Device
param = init_.Parameters

# Generate grid with coordinates at centers of cells
grid = [np.arange(1, dev.tot_seg[j]*2 + 1, 2)*dev.dl[j]/2 for j in range(3)]

coords = init_.init_coords()

pd.options.display.precision = 15
dfpos = pd.DataFrame(list(zip(*coords[:,3:].T)))
dfmesh = pd.DataFrame(list(product(*grid)))
rho = np.ones(len(dfmesh))*dev.Materials[0].dope*np.prod(dev.dl)*(-100**3)
rho_init = np.copy(rho)

def get_near(x):
    sub = np.product(abs(dfmesh - x) <= dev.dl/2, axis = 1)
    #nbors = dfmesh.iloc[list(sub[sub==1].index),:]
    try:
        #rho[list(sub[sub==1].index)] = qsup*np.product(1 - abs(nbors-x)/dim, axis = 1)
        #print(list(sub[sub==1].index))
        rho[list(sub[sub==1].index)] += dev.Materials[0].dope*dev.vol*(100**3)/dev.num_carr
    except:
        pass
    
start_charge = timer()
dfpos.apply(get_near, axis = 1)
rho_ = rho.reshape(dev.tot_seg)
end_charge = timer()
print ('total time: ', end_charge - start_charge)


u = np.random.random(rho_.shape)
# Apply Dirichlet boundary conditions on all surfaces
u_ = np.pad(u, pad_width = 1, mode = 'constant', constant_values = 0)
# Apply surface potential boundary condition
surf_val = 0.6
u_[:,:,0] = np.ones(u_[:,:,0].shape)*surf_val
rho_ = np.pad(rho_, pad_width = 1, mode = 'constant', constant_values = 0)

def get_adj(mat, i, j, k, dim):
    xy = (dim[0]/dim[1])**2
    xz = (dim[0]/dim[2])**2
    adj_cells = [-2*(1 + xy + xz)*mat[i][j][k], mat[i+1][j][k], mat[i-1][j][k], 
                 xy*mat[i][j+1][k], xy*mat[i][j-1][k], xz*mat[i][j][k+1], 
                 xz*mat[i][j][k-1]]
    return adj_cells, -2*(1 + xy + xz)

tol = 1e-6
err = 1
err_hist = []
relx = 1.5
num = 1
p = 0.5*np.cos(np.pi/dev.tot_seg[0]) + 0.5*np.cos(np.pi/dev.tot_seg[1]) + 0.5*np.cos(np.pi/dev.tot_seg[2])
while err > tol:
    print (f"Iteration {num}")
    u_prev = np.copy(u_)
    for k, j, i in product(range(1, dev.tot_seg[2] + 1), range(1, dev.tot_seg[1] + 1), range(1, dev.tot_seg[0] + 1)):
        adj_cells, e = get_adj(u_, i, j, k, dev.dl)
        du = (np.sum(adj_cells) - dev.dl[0]**2*rho_[i][j][k]/(dev.Materials[0].eps_0))*relx/e
        u_[i][j][k] -= du
        if abs(du) >= 1e3:
            print (f"{i},{j},{k}")
            raise ValueError("Too large du")
    relx = 1/(1-0.25*(p**2) * relx)
    err = np.sqrt(np.square(np.subtract(u_, u_prev)).mean(axis = None))
    err_hist.append(err)
    num += 1

    
# Calculate electric field
ef_x = np.zeros((dev.tot_seg[0] + 2, dev.tot_seg[1]+2, dev.tot_seg[2]+2), float)
ef_y = np.zeros((dev.tot_seg[0] + 2, dev.tot_seg[1]+2, dev.tot_seg[2]+2), float)
ef_z = np.zeros((dev.tot_seg[0] + 2, dev.tot_seg[1]+2, dev.tot_seg[2]+2), float)
for k in range(1, dev.tot_seg[2] + 1):
    for j in range(1, dev.tot_seg[1] + 1):
        for i in range(1, dev.tot_seg[0] + 1):
            ef_x[i][j][k] = -0.5*(u_[i+1][j][k] - u_[i-1][j][k])/dev.dl[0]
            ef_y[i][j][k] = -0.5*(u_[i][j+1][k] - u_[i][j-1][k])/dev.dl[1]
            ef_z[i][j][k] = -0.5*(u_[i][j][k+1] - u_[i][j][k-1])/dev.dl[2]

def plot_particles(coords):
    x, y, z = coords[:,3:].T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

