from State import State
from Response import Response
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Use glob to get the list of DATA files
datafiles = glob.glob('DATA*.pkl')

# Prepare for computation
S = State (filename=datafiles[0])

# Init
sequence = S.get_sequence()
correctors = S.get_correctors()['names']

# Pick all bpms following the first corrector
bpms = [ bpm for bpm in S.get_bpms()['names'] if sequence.index(bpm) > sequence.index(correctors[0]) ]

# Pick all correctors preceding the last bpm
hcorrs = [ corr for corr in S.get_hcorrectors_names() if sequence.index(corr) < sequence.index(bpms[-1]) ]
vcorrs = [ corr for corr in S.get_vcorrectors_names() if sequence.index(corr) < sequence.index(bpms[-1]) ]

# Read all orbits
Bx = np.empty((0,len(bpms)))
By = np.empty((0,len(bpms)))
Cx = np.empty((0,len(hcorrs)))
Cy = np.empty((0,len(vcorrs)))
datafiles_p = [f for f in datafiles if f[-9] == 'p']
for datafile_p in datafiles_p:
    datafile_m = datafile_p[:-9] + 'm' + datafile_p[-8:]
    if os.path.exists(datafile_m):
        Sp = State(filename=datafile_p)
        Sm = State(filename=datafile_m)
        Op = Sp.get_orbit (bpms)
        Om = Sm.get_orbit (bpms)
        Cx_p = Sp.get_correctors(hcorrs)['bact']
        Cy_p = Sp.get_correctors(vcorrs)['bact']
        Cx_m = Sm.get_correctors(hcorrs)['bact']
        Cy_m = Sm.get_correctors(vcorrs)['bact']
        if 0:
            O_x = Op['x'] - Om['x']
            O_y = Op['y'] - Om['y']
            C_x = Cx_p - Cx_m
            C_y = Cy_p - Cy_m
            Bx = np.vstack((Bx, O_x))
            By = np.vstack((By, O_y))
            Cx = np.vstack((Cx, C_x))
            Cy = np.vstack((Cy, C_y))
            print(Cx, Bx)
        else:
            Bx = np.vstack((Bx, Op['x']))
            Bx = np.vstack((Bx, Om['x']))
            By = np.vstack((By, Op['y']))
            By = np.vstack((By, Om['y']))
            Cx = np.vstack((Cx, Cx_p))
            Cx = np.vstack((Cx, Cx_m))
            Cy = np.vstack((Cy, Cy_p))
            Cy = np.vstack((Cy, Cy_m))
    else:
        print(f"Data file '{datafile_m}' does not exist, ignoring counterpart '{datafile_p}' for response matrix computation")

# Compute the response matrices
ones_column_x = np.ones((Cx.shape[0], 1))
ones_column_y = np.ones((Cy.shape[0], 1))

# Add the column of ones to the matrix
Cx = np.hstack((Cx, ones_column_x))
Cy = np.hstack((Cy, ones_column_y))

Rxx = np.transpose(np.linalg.lstsq(Cx, Bx, rcond=None)[0])
Rxy = np.transpose(np.linalg.lstsq(Cy, Bx, rcond=None)[0])
Ryx = np.transpose(np.linalg.lstsq(Cx, By, rcond=None)[0])
Ryy = np.transpose(np.linalg.lstsq(Cy, By, rcond=None)[0])

# Reference trajectory
'''
Bx = Rxx[:,-1]
By = Ryy[:,-1]
'''

Bx = np.mean(Bx,axis=0).reshape(-1,1)
By = np.mean(By,axis=0).reshape(-1,1)

# Response matrices
Rxx = Rxx[:,:-1]
Rxy = Rxy[:,:-1]
Ryx = Ryx[:,:-1]
Ryy = Ryy[:,:-1]

# Zero the response of all bpms preceeding the correctors
for corr in hcorrs:
     bpm_indexes = [ bpms.index(bpm) for bpm in bpms if sequence.index(bpm) < sequence.index(corr) ]
     Rxx[bpm_indexes, hcorrs.index(corr)] = 0
     Ryx[bpm_indexes, hcorrs.index(corr)] = 0

for corr in vcorrs:
     bpm_indexes = [ bpms.index(bpm) for bpm in bpms if sequence.index(bpm) < sequence.index(corr) ]
     Rxy[bpm_indexes, vcorrs.index(corr)] = 0
     Ryy[bpm_indexes, vcorrs.index(corr)] = 0

# Save on disk
R = Response()
R.bpms = bpms
R.hcorrs = hcorrs
R.vcorrs = vcorrs
R.Rxx = Rxx
R.Rxy = Rxy
R.Ryx = Ryx
R.Ryy = Ryy
R.Bx = Bx
R.By = By

R.save('response2.pkl')

# Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Reference trajectory')

# Plot on the first subplot
ax1.plot(Bx, label='Bx')
ax1.set_xlabel('BPMs [#]')
ax1.set_ylabel('Bx')
ax1.legend()

# Plot on the second subplot
ax2.plot(By, label='By')
ax2.set_xlabel('BPMs [#]')
ax2.set_ylabel('By')
ax2.legend()

plt.tight_layout()
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 10))

x = np.array(range(len(hcorrs)))
y = np.array(range(len(bpms)))
X, Y = np.meshgrid(x, y)

ax1.plot_surface(X, Y, Rxx, cmap='viridis')
ax1.set_title('$R_{xx}$')
ax1.set_xlabel('Corrector [#]')
ax1.set_ylabel('BPM [#]')

ax3.plot_surface(X, Y, Ryx, cmap='viridis')
ax3.set_title('$R_{yx}$')
ax3.set_xlabel('Corrector [#]')
ax3.set_ylabel('BPM [#]')

x = np.array(range(len(vcorrs)))
y = np.array(range(len(bpms)))
X, Y = np.meshgrid(x, y)

ax2.plot_surface(X, Y, Rxy, cmap='viridis')
ax2.set_title('$R_{xy}$')
ax2.set_xlabel('Corrector [#]')
ax2.set_ylabel('BPM [#]')

ax4.plot_surface(X, Y, Ryy, cmap='viridis')
ax4.set_title('$R_{yy}$')
ax4.set_xlabel('Corrector [#]')
ax4.set_ylabel('BPM [#]')

plt.tight_layout()
plt.show()


