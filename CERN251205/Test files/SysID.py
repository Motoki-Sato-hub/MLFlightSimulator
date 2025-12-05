# from InterfaceATF2_Linac import InterfaceATF2_Linac
from Interfaces.ATF2.InterfaceATF2_DR import InterfaceATF2_DR
# from InterfaceATF2_Ext_RFTrack import InterfaceATF2_Ext_RFTrack
from State import State
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import signal
import os

# Connect to interface ATF LINAC, DR or ATF2 beamline
# I = InterfaceATF2_Linac(nsamples=3)
I = InterfaceATF2_DR(nsamples=3)
# I = InterfaceATF2_Ext(nsamples=3)
# I = InterfaceATF2_Ext_RFTrack(jitter=0.0, bpm_resolution=0.0, nsamples=1)

# Create the working environment
project_name = 'new_SYSID'
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
dir_name = f"Data/{project_name}_{time_str}"
os.makedirs (dir_name)
os.chdir (dir_name)

# What response matrix
DFS = False

# Create a machine
S = State (interface=I)

# Save the reference file
F = S.save (basename='machine_status')

# Install CTRL-C signal handler
def signal_handler(sig, frame, var):
    print('Caught CTRL-C, exiting gracefully!')
    S = var[0]
    F = var[1]
    DFS = var[2]
    S.load(F)
    try:
        S.push(I)
        if DFS:
            I.reset_energy()
    except:
        pass
    exit(0)

signal.signal(signal.SIGINT, partial(signal_handler, var=(S,F,DFS)))

# The list of correctors to use 
C = [
'ZV1R', 'ZH1R', 'ZV2R', 'ZH2R',
 'ZV3R', 'ZH3R', 'ZV4R', 'ZH4R',
 'ZV5R', 'ZH5R', 'ZV6R', 'ZH6R',
# 'ZV7R', 'ZH7R', 'ZV8R', 'ZH8R',
# 'ZV9R', 'ZH9R', 'ZH10R', 'ZV10R',
# 'ZH11R', 'ZV11R', 'ZH12R', 'ZV12R',  
# 'ZV13R', 'ZH13R', 'ZV14R', 'ZH14R', 
# 'ZV15R', 'ZV16R', 'ZH15R', 'ZV17R', 
# 'ZH16R', 'ZV18R', 'ZH17R', 'ZV19R', 
# 'ZH18R', 'ZV20R', 'ZH19R', 'ZV21R', 
# 'ZH20R', 'ZV22R', 'ZH21R', 'ZV23R', 
# 'ZH22R', 'ZV24R', 'ZH23R', 'ZV25R', 
# 'ZH24R', 'ZV26R', 'ZH25R', 'ZV27R', 
# 'ZH26R', 'ZV28R', 'ZH27R', 'ZV29R', 
# 'ZH28R', 'ZV30R', 'ZH29R', 'ZV31R', 
# 'ZH30R', 'ZV32R', 'ZH31R', 'ZH32R', 
# 'ZV33R', 'ZV34R', 'ZH33R', 'ZH34R', 
# 'ZV35R', 'ZV36R', 'ZH35R', 'ZV37R',
# 'ZH36R', 'ZV38R', 'ZH37R', 'ZV39R', 
# 'ZV40R', 'ZH38R', 'ZH41R', 'ZV39R', 
# 'ZH42R', 'ZV43R', 'ZH40R', 'ZV44R', 
# 'ZH41R', 'ZV45R', 'ZH42R', 'ZV46R', 
# 'ZH43R', 'ZV47R', 'ZH44R', 'ZV48R', 
# 'ZH45R', 'ZV49R', 'ZH46R', 'ZV50R', 
# 'ZH47R', 'ZV51R', 'ZH48R'
]

C = S.get_correctors()['names']
print(C)

# The list of bmps to use
B = [
'MB1R', 'MB2R', 'MB3R', 'MB4R','MB5R', 'MB6R', 'MB7R', 'MB8R',
'MB9R', 'MB10R', 'MB11R', 'MB12R','MB13R', 'MB14R', 'MB15R', 'MB16R', 'MB17R', 'MB18R', 'MB19R', 'MBX1', 'MBX2', 'MB21R', 'MB22R',
'MB23R', 'MB24R','MB25R', 'MB26R',  'MB27R', 'MB28R', 'MB29R', 'MB30R', 'MB31R', 'MB32R', 'MB33R',
'MB34R', 'MB35R', 'MB36R', 'MB37R','MB38R', 'MB39R', 'MB40R', 'MB41R', 'MB42R', 'MB43R', 'MB44R', 'MB45R', 
# 'MB46R', 'MB47R', 'MB48R', 'MB49R', 'MB50R', 'MB51R', 'MB52R', 'MB53R','MB54R', 'MB55R', 'MB56R', 'MB57R', 'MB58R', 'MB59R', 'MB60R', 'MB61R', 'MB62R', 'MB63R', 'MB64R', 'MB65R', 'MB66R', 'MB67R', 'MB68R', 'MB69R', 'MB70R', 'MB71R', 'MB72R', 'MB73R', 'MB74R', 'MB76R', 'MB77R', 'MB78R', 'MB79R', 'MB80R', 'MB81R','MB82R', 'MB83R', 'MB84R', 'MB85R', 'MB86R', 'MB87R', 'MB88R', 'MB89R', 'MB90R','MB91R', 'MB92R', 'MB93R', 'MB94R','MB95R', 'MB96R', 'MB97R', 'MB98R'
]

B = S.get_bpms()['names']

# Extra functions
def plot_orbit(orbit, figure):
    plt.figure(figure)
    plt.clf()
    errx = orbit['stdx'] / np.sqrt(orbit['stdx'].size)
    erry = orbit['stdy'] / np.sqrt(orbit['stdy'].size)
    plt.errorbar (range(orbit['nbpms']), np.transpose(orbit['x']), yerr=errx, lw=2, capsize=5, capthick=2, label="X")
    plt.errorbar (range(orbit['nbpms']), np.transpose(orbit['y']), yerr=erry, lw=2, capsize=5, capthick=2, label="Y")
    plt.legend (loc='upper left')
    plt.xlabel ('Bpm [#]')
    plt.ylabel ('Position [mm]')
    plt.draw()
    plt.pause(0.1)  

# Turn on interactive mode
plt.ion()

# Kick to achieve 1mm max excursion
kicks = 0.2 * np.ones(len(C), dtype=float) # kicks to excite 1mm oscillation
max_oscillation = 0.1 # mm

if DFS:
    I.change_energy()

# 10 loops to measure the response matrix
print("Press CTRL-C to interrupt the program.")
Niter = 2
for iter in range (Niter):
    print(f'Iteration {iter}/{Niter}')
    for icorr, corrector in enumerate(C):

        print('corrector = ', corrector)
        
        # initial value
        corr = S.get_correctors (corrector)
        kick = kicks[icorr]

        # '+' excitation 
        print(f"Corrector {corrector} '+' excitation...")
        I.push(corrector, corr['bdes'] + kick)
        S.pull(I)
        S.save(filename=f'DATA_{corrector}_p{iter:04d}.pkl')
        Op = S.get_orbit(B)
        plot_orbit(Op, 1)
        
        # '-' excitation 
        print(f"Corrector {corrector} '-' excitation...")
        I.push(corrector, corr['bdes'] - kick)
        S.pull(I)
        S.save(filename=f'DATA_{corrector}_m{iter:04d}.pkl')
        Om = S.get_orbit (B)
        plot_orbit(Om, 2)
        
        # reset corrector
        I.push(corrector, corr['bdes'])
        
        # Orbit difference
        Diff_x = (Op['x'] - Om['x']) / 2.0
        Diff_y = (Op['y'] - Om['y']) / 2.0
        nsamples = Op['stdx'].size
        Err_x = np.sqrt(np.square(Op['stdx']) + np.square(Om['stdx'])) / np.sqrt(nsamples)
        Err_y = np.sqrt(np.square(Op['stdy']) + np.square(Om['stdy'])) / np.sqrt(nsamples)
        
        # Tunes the kickers omplitude
        if corrector in S.get_hcorrectors_names():
            kicks[icorr] *= max_oscillation / np.max(np.absolute(Diff_x))
        else:
            kicks[icorr] *= max_oscillation / np.max(np.absolute(Diff_y))

        # weighted average
        kicks[icorr] = 0.8 * kicks[icorr] + 0.2 * kick
        np.savetxt('kicks.txt', kicks, delimiter='\n')

        # Plot orbit    
        plt.figure(3)
        plt.clf()
        plt.errorbar (range(Op['nbpms']), Diff_x, yerr=Err_x, lw=2, capsize=5, capthick=2, label="X")
        plt.errorbar (range(Op['nbpms']), Diff_y, yerr=Err_y, lw=2, capsize=5, capthick=2, label="Y")
        plt.legend (loc='upper left')
        plt.xlabel ('Bpm [#]')
        plt.ylabel ('Orbit [mm]')
        plt.title (f"Corrector '{corrector}'")
        plt.draw()
        plt.pause(0.1)

if DFS:
    I.reset_energy()

print('Done!')

