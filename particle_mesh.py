# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:38:32 2020

@author: Carlo
"""
from __future__ import division
import numpy as np
from Material import GaAs
import func, init
import matplotlib.pyplot as plt


kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
q = 1.602E-19 # Electron charge in Coulombs
eps0 = 8.854E-12 # Vacuum permittivity 

# Device parameters
lx = 3E-6
ly = 3E-6
lz = 500E-9
N = 1E16
rho_sheet = N
rho_0 = 0
num_carr = 10000
V = lx*ly*lz

num_carr = 5000 # number of carriers/particles
elec_field = -5000 # electric field magnitude in V/cm
dt = 2E-15 # time step must be smaller than 1/plasmon freq i.e. 0.16 ps for N = 1E16 cm^-3
N = init.N # Doping concentration (cm-3)
pts = 750 # Data points
mean_energy = 0.25

qsup = q*N*V*(1E2**3)/num_carr # charge of superparticle
# Mesh size dx_i must be larger than the v_max x dt 
# but smaller than the inverse debye length; it is also dependent on the relevant
# dimensions of the system
seg_z = int(lz*func.inv_debye_length(N))
seg_x = int(lx*func.inv_debye_length(N))
seg_y = int(ly*func.inv_debye_length(N))
dz = lz/seg_z
dx = lx/seg_x
dy = ly/seg_y

# Generate num_carr particles with coordinates (x, z) within lx, lz
z = np.random.uniform(0, lz, num_carr)
y = np.random.uniform(0, ly, num_carr)
x = np.random.uniform(0, lx, num_carr)

rho = np.zeros((seg_z, seg_x, seg_y))
mesh_grid_z = np.arange(1, seg_z*2 + 1, 2)*dz/2
mesh_grid_y = np.arange(1, seg_y*2 + 1, 2)*dy/2
mesh_grid_x = np.arange(1, seg_x*2 + 1, 2)*dx/2
xx, yy, zz = np.meshgrid(mesh_grid_y, mesh_grid_z, mesh_grid_x)

def in_range(x, y, z, xx, yy, zz, i, j, k, carr):
    z = (abs(zz[k][j][i]-z[carr]) <= dz)
    y = (abs(yy[k][j][i]-y[carr]) <= dy)
    x = (abs(xx[k][j][i]-x[carr]) <= dx)
    return (x and y and z)

def CIC(x, y, z, xx, yy, zz, i, j, k, carr):
    z = (1 - abs(zz[k][j][i]-z[carr])/dz)
    y = (1 - abs(yy[k][j][i]-y[carr])/dy)
    x = (1 - abs(xx[k][j][i]-x[carr])/dx)
    return x * y * z

# Charge density assignment scheme: Cloud-in-cell
# For each mesh grid point (midpoint), if particle location is less then dx/2 away from 
# mesh grid point, assign carrier density to that mesh point
for k in range(len(zz)):
    for j in range(len(yy[0])):
        for i in range(len(xx[0])):
            print ("z index %d, y index: %d, x index %d " %(k, j, i))
            count = 0
            for carr in range(len(z)):
                count +=  in_range(x, y, z, xx, yy, zz, i, j, k, carr) * \
                CIC(x, y, z, xx, yy, zz, i, j, k, carr)
            rho[k][j][i] = qsup*count + rho_0


#def calc_lpot(pot, dz):
#    lpot = np.zeros(len(pot))
#    for i in range(len(pot)):
#        lpot[i] = (pot[(i-1) % len(pot)] + pot[(i+1) % len(pot)] - 2*pot[i])/dz**2
#    return lpot

tol = 1E-6
u_bound = np.random.random((seg_z + 2, seg_y + 2, seg_x + 2))

# Apply Dirichlet boundary conditions on all surfaces
top_val = elec_field*lz
u_bound[0, :, :] = np.ones((seg_y+2, seg_x+2)) * top_val
u_bound[-1, :, :] = np.zeros((seg_y+2, seg_x+2))
side_val = 0
u_bound[:, :, 0] = np.zeros((seg_z+2, seg_y+2))*side_val
u_bound[:, :, -1] = np.zeros((seg_z+2, seg_y+2))*side_val
u_bound[:, 0, :] = np.zeros((seg_z+2, seg_x+2))*side_val
u_bound[:, -1, :] = np.zeros((seg_z+2, seg_x+2))*side_val
rho_bound = np.zeros((seg_z + 2, seg_y + 2, seg_x + 2))
rho_bound[1:seg_z + 1, 1:seg_y + 1, 1:seg_x + 1] = rho

def get_adj(mat, k, j, i, dx, dy, dz):
    rat_xy = (dx/dy)**2
    rat_xz = (dx/dz)**2
    adj_cells = [-2*(1 + rat_xy + rat_xz)*mat[k][j][i], mat[k][j][i+1], mat[k][j][i-1], 
                 rat_xy*(mat[k][j+1][i]), rat_xy*(mat[k][j-1][i]), rat_xz*(mat[k+1][j][i]), 
                 rat_xz*(mat[k-1][j][i])]
    return adj_cells, -3*(1 + rat_xy + rat_xz)

def on_boundary(mat, i, j):
    M, N = mat.shape
    if i == 0:
        return True
    if i == M - 1:
        return True
    if j == 0: 
        return True
    if j == N - 1:
        return True
    else:
        return False
    
u = np.copy(u_bound)
err = 1
p = 0.5*np.cos(np.pi/seg_z) + 0.5*np.cos(np.pi/seg_y) + 0.5*np.cos(np.pi/seg_x)
relax = 1
num = 1
err_hist = []
while err > tol:
    print (f"Iteration {num}")
    u_prev = np.copy(u)
    for k in range(1, seg_z+1):
        for j in range(1, seg_y+1):
            for i in range(1, seg_x+1):
                adj_cells, e = get_adj(u, k, j, i, dx, dy, dz)
                du = (np.sum(adj_cells) - (dx**2 * rho_bound[k][j][i]/(q*GaAs.eps_0)))*relax/e
                u[k][j][i] = u_prev[k][j][i] - du
                if abs(du) > 1E3:
                    print (f"({k},{i})")
                    raise Exception("Too large du")
    relax = 1/(1-(0.0625*p**2 * relax))
    err = np.sqrt(np.square(np.subtract(u, u_prev)).mean(axis = None))
    err_hist.append(err)
    num += 1
    
# Calculate electric field
ef_x = np.zeros((seg_z + 2, seg_y+2, seg_x+2), float)
ef_y = np.zeros((seg_z + 2, seg_y+2, seg_x+2), float)
ef_z = np.zeros((seg_z + 2, seg_y+2, seg_x+2), float)
for k in range(1, seg_z + 1):
    for j in range(1, seg_y + 1):
        for i in range(1, seg_x+1):
            ef_x[k][j][i] = 0.5*(u[k][j][i+1] - u[k][j][i-1])/dx
            ef_y[k][j][i] = 0.5*(u[k][j+1][i] - u[k][j-1][i])/dy
            ef_z[k][j][i] = 0.5*(u[k+1][j][i] - u[k-1][j][i])/dz

#fig, ax = plt.subplots()
#ax.plot(list(range(0, len(err_hist))), err_hist)



    


    




        
    



  

