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

num_carr = 5000 # number of carriers/particles
elec_field = -5000 # electric field magnitude in V/cm
dt = 2E-15 # time step must be smaller than 1/plasmon freq i.e. 0.16 ps for N = 1E16 cm^-3
N = init.N # Doping concentration (cm-3)
pts = 750 # Data points
mean_energy = 0.25
#dx = 1/func.inv_debye_length(N)
#Q = q*N/num_carr # charge of superparticles for charge neutrality

''' --- Initialization --- '''

# init_coords: 1 x 6 x num_carr arrays for kx, ky, kz, x, y, z v
init_coords = init.init_coord(num_carr, mean_energy)
coords = np.copy(init_coords)
valley = np.zeros(num_carr)
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
val_hist = np.zeros((num_carr, pts))

dtau = [func.free_flight(1E15) for i in range(num_carr)]
for c, t in enumerate(t_range):
    sheet['A' + str(c + 2)] = t*1E12
    logging.debug('Time: %0.4g' %t)    
    for carr in range(num_carr):
        dte = dtau[carr]
        if dte >= dt:
            dt2 = dt
            drift_coords = func.carrier_drift(coords[carr], elec_field, dt2, scatter.param[valley[carr]]['m'])
        else:
            dt2 = dte
            drift_coords = func.carrier_drift(coords[carr], elec_field, dt2, scatter.param[valley[carr]]['m'])
            while dte < dt:
                dte2 = dte
                drift_coords = func.carrier_drift(drift_coords, elec_field, dte, scatter.param[valley[carr]]['m'])
                drift_coords, valley[carr] = scatter.choose_scat(valley[carr], drift_coords, N)
                dt3 = func.free_flight(1E15)
                dtp = dt - dte2
                if dt3 <= dtp:
                    dt2 = dt3
                else:
                    dt2 = dtp
                drift_coords = func.carrier_drift(drift_coords, elec_field, dt2, scatter.param[valley[carr]]['m'])
                dte = dte2 + dt3
        dte -= dt
        dtau[carr] = dte
        coords[carr] = drift_coords
        energy = func.calc_energy(*drift_coords[0:3], scatter.param[valley[carr]]['m'])
        e_hist[carr][c] = energy
        v_hist[carr][c] = coords[carr][2]*hbar/scatter.param[valley[carr]]['m']
        z_hist[carr][c] = drift_coords[5]
        val_hist[carr][c] = valley[carr]
        print (energy)
    sheet['B' + str(c+2)] = np.mean(z_hist[:,c])*1E9
    sheet['C' + str(c+2)] = np.mean(v_hist[:,c])
    sheet['D' + str(c+2)] = np.mean(e_hist[:,c])
    sheet['E' + str(c+2)] = val_hist[:,c].tolist().count(0)
    sheet['F' + str(c+2)] = val_hist[:,c].tolist().count(1)
    sheet['G' + str(c+2)] = val_hist[:,c].tolist().count(2)

     
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

G_val = np.zeros(pts)
L_val = np.zeros(pts)
X_val = np.zeros(pts)
for i in range(pts):
    G_val[i] = val_hist[:,i].tolist().count(0)
    L_val[i] = val_hist[:,i].tolist().count(1)
    X_val[i] = val_hist[:,i].tolist().count(2)
    
fig3, ax3 = plt.subplots()
ax3.plot(np.array(t_range)*1E12, G_val, label = r"$\Gamma$ pop.")
ax3.plot(np.array(t_range)*1E12, L_val, label = r"L pop.")
ax3.plot(np.array(t_range)*1E12, X_val, label = r"X pop.")
ax3.set_xlabel('Time (ps)')
ax3.set_ylabel('Valley population')
ax3.set_xlim([0, t_range[-1]*1E12])

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
sheet['E1'] = 'Gamma-Valley Population'
sheet['F1'] = 'L-Valley Population'
sheet['G1'] = 'X-Valley Population'

# Simulation parameter inputs
for i, key in enumerate(list(xl_input.keys())):
    sheet['I' + str(i+2)] = key
    sheet['J' + str(i+2)] = xl_input[key]
sheet['I' + str(len(xl_input.keys()) + 1)] = 'Actual Simulation Time'
sheet['J' + str(len(xl_input.keys()) + 1)] = f'{mins} mins, {secs:.2f} s'

# Set column width and freeze first row
sheet.column_dimensions['I'].width = 21
sheet.freeze_panes = 'A2'

# Generate Excel chart
#refObjx = openpyxl.chart.Reference(sheet, min_col = 1, min_row = 1, max_col = 1, max_row = sheet.max_row)
#seriesObjx = openpyxl.chart.series(refObjx, title = 'Time (ps)')
#refObjy = openpyxl.chart.Reference(sheet, min_col = 3, min_row = 1, max_col = 3, max_row = sheet.max_row)
#seriesObjy = openpyxl.chart.series(refObjy, title = 'Mean velocity (m/s)')
#chartObj = openpyxl.chart.ScatterChart()
#chartObj.append(seriesObjx)
#chartObj.append(seriesObjy)
#sheet.add_chart(chartObj, 'I2')
wb.save(os.getcwd() + '\\' + filename)