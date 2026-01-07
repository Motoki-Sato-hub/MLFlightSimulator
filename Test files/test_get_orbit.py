from InterfaceATF2 import InterfaceATF2
from State import State
import matplotlib.pyplot as plt
import numpy as np
import os

import time

# Set up environment
data_path = '../Data'
try:
    os.mkdir(data_path)
except:
    pass

# Define interface    
I = InterfaceATF2(nsamples=1)

# Prepare for data taking
S = State(I)
S.pull()

names = [
    "MB2X",
    "MQF1X",
    "MQD2X",
    "MQF3X"
]

O = S.get_orbit(names)

print(O['faulty'])

plt.figure()
plt.errorbar(range(O['nbpms']), np.transpose(O['x']), yerr=O['stdx'], lw=2, capsize=5, capthick=2)
plt.errorbar(range(O['nbpms']), np.transpose(O['y']), yerr=O['stdy'], lw=2, capsize=5, capthick=2)
plt.show()
