import sys
sys.path.append('.')
sys.path.append('Interfaces/ATF2')

from State import State

import matplotlib.pyplot as plt
import numpy as np
import glob
import os

if 1:
    from Interfaces.ATF2.InterfaceATF2_DR_RFTrack import InterfaceATF2_DR_RFTrack
    I = InterfaceATF2_DR_RFTrack()
else:
    from Interfaces.ATF2.InterfaceATF2_Ext_RFTrack import InterfaceATF2_Ext_RFTrack
    I = InterfaceATF2_Ext_RFTrack()

project_name = I.get_name()
print(f"Selected interface: {project_name}")

# TEST 1 - Read one bpm
mb1x_disp_x, mb1x_disp_y = I.get_target_dispersion("MB1X")
print('disp x @ MB1X = ', mb1x_disp_x, ' m')
print('disp y @ MB1X = ', mb1x_disp_y, ' m')

# TEST 2 - Read several bpms
names = I.get_bpms_names()
disp_x, disp_y = I.get_target_dispersion(names[:5])
print('name of bpms = ', names)
print('disp x first 5 BPMS = ', disp_x, ' m')
print('disp y first 5 BPMS = ', disp_y, ' m')

# TEST 3 - Read all bpms
target_disp_x, target_disp_y = I.get_target_dispersion()

plt.figure(1)
plt.plot(target_disp_x, 'b-', linewidth=2, label='disp x')
plt.plot(target_disp_y, 'r-', linewidth=2, label='disp y')
plt.legend()
plt.xlabel('BPM [#]')
plt.ylabel('target dispersion [m]')
plt.show()
