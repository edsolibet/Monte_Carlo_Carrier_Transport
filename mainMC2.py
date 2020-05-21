#! python3
"""
Created on Wed May  6 15:13:10 2020

@author: Carlo
"""
from __future__ import division
import datetime
import logging

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(levelname)s \
                    - %(message)s')
logging.debug('Start of Program')
startTime = datetime.datetime.now()

import numpy as np
import matplotlib.pyplot as plt
from Material import GaAs
import init, func, scatter, os, openpyxl


''' --- Constants --- '''
inf = float('inf') # infinity
kT = 0.02585 # Boltzmann constant times the semiconductor temperature, expressed in eV
m_0 = 9.11E-31 # Electron mass in kg
q = 1.602E-19 # Electron charge in Coulombs
hbar = 1.055E-34 # Reduced planck's constant in Joules sec
hbar_eVs = 6.582E-16 # Reduced Planck's constant in eV sec
eps0 = 8.854E-12 # Vacuum permittivity 

''' --- Simulation parameters --- '''

logging.debug('Initiatlization of parameters')

num_carr = 10000 # number of carriers/particles
elec_field = -500 # electric field magnitude in V/cm
dt = 2E-15 # time step
N = 1E16 # Doping concentration (cm-3)
pts = 1500 # Data points

''' --- Initialization --- '''

# init_coords: 1 x 6 x num_carr arrays for kx, ky, kz, x, y, z
init_coords = init.init(num_carr)
coords = np.copy(init_coords)
logging.debug('Initial coordinates generated.')

# Generate Excel file
dirPath = 'C:/Users/Carlo/Documents/Research/THz emission mechanisms in InGaAs/'
folderName = 'Monte Carlo Simulation Results'
os.chdir(dirPath + folderName)
keyword = "Steady-state Transport Kernel"

num = 1
while True:
    filename = datetime.datetime.today().strftime("%Y-%m-%d") + ' ' + keyword + \
    ' (' + str(num) + ').xlsx'
    if not os.path.exists(filename):
        break
    num += 1
wb = openpyxl.Workbook()
wb.save(os.getcwd() + '\\' + filename)
wb = openpyxl.load_workbook(os.getcwd() + '\\' + filename)
sheet = wb[wb.sheetnames[0]]

''' --- Main Monte Carlo Transport Kernel --- '''
# Generate variable lists
t_range = [i*dt for i in range(pts)]
v_hist = np.zeros((num_carr, pts))
z_hist = np.zeros((num_carr, pts))
e_hist = np.zeros((num_carr, pts))

dtau = [func.free_flight() for i in range(num_carr)]
for c, t in enumerate(t_range):
    sheet['A' + str(c + 2)] = t*1E12
    logging.debug('Time: %0.4g' %t)    
    for carr in range(num_carr):
        dte = dtau[carr]
        if dte >= dt:
            dt2 = dt
            drift_coords = func.carrier_drift(coords[carr], elec_field, dt2)
        else:
            dt2 = dte
            drift_coords = func.carrier_drift(coords[carr], elec_field, dt2)
            while dte < dt:
                dte2 = dte
                drift_coords = func.carrier_drift(coords[carr], elec_field, dte)
                mech = scatter.choose_scatter(func.calc_energy(*drift_coords[0:3]), scatter.scat)            
                drift_coords = scatter.scatter_angles(drift_coords, N, scatter.scat[mech])    
                dt3 = func.free_flight()
                dtp = dt - dte2
                if dt3 <= dtp:
                    dt2 = dt3
                else:
                    dt2 = dtp
                drift_coords = func.carrier_drift(drift_coords, elec_field, dt2)
                dte = dte2 + dt3
        dte -= dt
        dtau[carr] = dte
        coords[carr] = drift_coords
        energy = func.calc_energy(*drift_coords[0:3])
        e_hist[carr][c] = energy
        v_hist[carr][c] = np.sqrt(2*q*energy/GaAs.m_G)
        z_hist[carr][c] = drift_coords[5]
    sheet['B' + str(c+2)] = np.mean(z_hist[:,c])*1E9
    sheet['C' + str(c+2)] = np.mean(v_hist[:,c])
    sheet['D' + str(c+2)] = np.mean(e_hist[:,c])

  
''' --- Plots --- '''
fig1, ax1 = plt.subplots()
ax1.plot(np.array(t_range)*1E12, v_hist.mean(axis=0))
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Mean Velocity (m/s)')
ax1.set_xlim([0, t_range[-1]])

fig2, ax2 = plt.subplots()
ax2.plot(np.array(t_range)*1E12, e_hist.mean(axis = 0))
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Mean Energy (eV)')
ax2.set_xlim([0, t_range[-1]])

''' --- End Simulation --- '''
total_time = datetime.datetime.now() - startTime
mins = int(total_time.total_seconds() / 60)
secs = total_time.total_seconds() % 60
print("---%s minutes, %s seconds ---" % (mins, secs))

''' --- Excel Workbook Inputs --- '''

xl_input = {'Number of Carriers' : num_carr,
            'Electric field (V/cm)': elec_field,
            'Impurity Doping (cm-3)': N,
            'Time step (ps)': dt,
            'Data Points': pts,
            'Simulation Time (ps)': dt*pts
            }

# Series Heading titles
sheet['A1'] = 'Time (ps)'
sheet['B1'] = 'Average z pos. (nm)'
sheet['C1'] = 'Average velocity (m/s)'
sheet['D1'] = 'Average energy (eV)'

# Simulation parameter inputs
for i, key in enumerate(list(xl_input.keys())):
    sheet['F' + str(i+2)] = key
    sheet['G' + str(i+2)] = xl_input[key]
sheet['F' + str(len(xl_input.keys()) + 1)] = 'Actual Simulation Time'
sheet['G' + str(len(xl_input.keys()) + 1)] = f'{mins} mins, {secs:.2f} s'

# Set column width and freeze first row
sheet.column_dimensions['F'].width = 21
sheet.freeze_panes = 'A2'

# Generate Excel chart
refObjx = openpyxl.chart.Reference(sheet, min_col = 1, min_row = 1, max_col = 1, max_row = sheet.max_row)
seriesObjx = openpyxl.chart.series(refObjx, title = 'Time (ps)')
refObjy = openpyxl.chart.Reference(sheet, min_col = 3, min_row = 1, max_col = 3, max_row = sheet.max_row)
seriesObjy = openpyxl.chart.series(refObjy, title = 'Mean velocity (m/s)')
chartObj = openpyxl.chart.ScatterChart()
chartObj.append(seriesObjx)
chartObj.append(seriesObjy)
sheet.add_chart(chartObj, 'I2')
wb.save(os.getcwd() + '\\' + filename)