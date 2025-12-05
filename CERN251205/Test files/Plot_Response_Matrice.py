import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

with open('/userhome/atfop1/CERN-Flight_Simulator/flight-simulator-live/Data/new_SYSID_20250619_090834/response2.pkl', 'rb') as f:
    R = pickle.load(f)

bpms=R["bpms"]
hcorrs=R["hcorrs"]
vcorrs=R["vcorrs"]
Rxx=R["Rxx"]
Rxy=R["Rxy"]
Ryx=R["Ryx"]
Ryy=R["Ryy"]
Bx=R["Bx"]
By=R["By"]

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
