# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:38:32 2020

@author: Carlo
"""
from __future__ import division
import numpy as np
from Material import GaAs
import func


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
num_carr = 1000
V = lx*ly*lz

qsup = q*N*V*(1E2**3)/num_carr # charge of superparticle
# Mesh size dx_i must be larger than the v_max x dt 
# but smaller than the inverse debye length; it is also dependent on the relevant
# dimensions of the system
seg_z = int(lz*func.inv_debye_length(N))
seg_x = int(lx*func.inv_debye_length(N))
dz = lz/seg_z
dx = lx/seg_x

z = np.random.uniform(0, lz, num_carr)
x = np.random.uniform(0, lx, num_carr)

rho = np.zeros((seg_z, seg_x))
mesh_grid_z = np.arange(1, seg_z*2 + 1, 2)*dz/2
mesh_grid_x = np.arange(1, seg_x*2 + 1, 2)*dx/2
xx, zz = np.meshgrid(mesh_grid_x, mesh_grid_z)

# Charge density assignment scheme: Cloud-in-cell
# For each mesh grid point (midpoint), if particle location is less then dx/2 away from 
# mesh grid point, assign carrier density to that mesh point
for i in range(len(zz)):
    for j in range(len(xx[0])):
        print ("z index %d, x index: %d " %(i,j))
        count = 0
        for carr in range(len(z)):
            count += ((abs(zz[i][j]-z[carr]) <= dz) and (abs(xx[i][j]-x[carr]) <= dx)) * \
            (1 - abs(zz[i][j]-z[carr])/dz)*(1 - abs(xx[i][j]-x[carr])/dx)
        rho[i][j] = qsup*count + rho_0
    
''' --- UNTIL HERE --- '''

def calc_lpot(pot, dz):
    lpot = np.zeros(len(pot))
    for i in range(len(pot)):
        lpot[i] = (pot[(i-1) % len(pot)] + pot[(i+1) % len(pot)] - 2*pot[i])/dz**2
    return lpot

#prev_pot = np.zeros(seg)
#pot = np.random.random(seg)*10
#pot[0:5] = 10
#pot[-5:-1] = 0
#tol = 1E-3
#count = 0
#while abs(sum(pot - prev_pot)) >= tol:
#    print ("Iteration: %d" %count)
#    print ("Error: %0.3f" %(abs(sum(pot - prev_pot))))
#    prev_pot = np.copy(pot)
#    for k in range(len(pot)):
#        pot[k] = 0.5*(prev_pot[(k-1) % len(pot)] + prev_pot[(k+1) % len(pot)] - rho[k]*dz**2/(q*GaAs.eps_0))
#    pot[0:5] = 10
#    pot[-5:-1] = 0
#    count += 1
#print (abs(sum(pot - prev_pot)))

# Create matrix with dimensions J x J x (I * I)
A = np.zeros((seg_z, seg_z, seg_x, seg_x), np.float)
T = np.identity(seg_x) * -4

# Fill in submatrix
for i in range(seg_x - 1):
    T[i][i+1] = -1
    T[i+1][i] = -1

# Create identity submatrix with dimension J
B = np.identity(seg_x)

# Fill matrix with submatrix B
for i in range(seg_z - 1):
    A[i][i+1] = B
    A[i+1][i] = B
    A[i][i] = T
    
# Invert A
A_inv = A.getI()


    


    




        
    



  

