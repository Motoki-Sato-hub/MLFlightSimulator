import sys
sys.path.append('/userhome/alatina/flight-simulator')

from State import State
from Response import Response
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

R = Response('response2.pkl')

# Use glob to get the list of DATA files
datafiles = glob.glob('DATA*.pkl')

# Read all orbits
datafiles_p = [f for f in datafiles if f[-10] == 'p']
for datafile_p in datafiles_p:
    datafile_m = datafile_p[:-10] + 'm' + datafile_p[-9:]
    if os.path.exists(datafile_m):
        Sp = State(datafile_p)
        Sm = State(datafile_m)
        Op = Sp.get_orbit (R.bpms)
        Om = Sm.get_orbit (R.bpms)
        Cx_p = Sp.get_correctors(R.hcorrs)['bact']
        Cy_p = Sp.get_correctors(R.vcorrs)['bact']
        Cx_m = Sm.get_correctors(R.hcorrs)['bact']
        Cy_m = Sm.get_correctors(R.vcorrs)['bact']
        
        dC_x = (Cx_p - Cx_m)/2
        dC_y = (Cy_p - Cy_m)/2
        
        Bx = np.hstack((R.Rxx,R.Rxy)) @ np.hstack((dC_x,dC_y)).T
        By = np.hstack((R.Ryx,R.Ryy)) @ np.hstack((dC_x,dC_y)).T
        
        dO_x = (Op['x'] - Om['x'])/2
        dO_y = (Op['y'] - Om['y'])/2
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        ax1.clear()
        ax2.clear()
               
        ax1.set_title(datafile_p)
        ax1.plot(Bx,label='Bx')
        ax1.plot(dO_x,label='Ox')
        ax1.legend()

        ax2.set_title(datafile_p)
        ax2.plot(By,label='By')
        ax2.plot(dO_y,label='Oy')
        ax2.legend()
        plt.show()
        
    else:
        print(f"Data file '{datafile_m}' does not exist, ignoring counterpart '{datafile_p}' for response matrix computation")


