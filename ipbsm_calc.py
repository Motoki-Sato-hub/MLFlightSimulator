# ============================================================
#  ipbsm_calc.py
# Author Motoki Sato
# Date 2025/12/4
# ============================================================

import numpy as np

PI = np.pi
LAMBDA = 532e-9  # laser wavelength [m]

def sigma_from_array(data):
    if len(data) < 2:
        return -10000.0
    mean = np.mean(data)
    return np.sqrt(np.mean(data ** 2) - mean ** 2)


def sigma_clip(data, cut=2.0):
    sig = -100.0
    ncut = 1

    while len(data) > 1 and ncut > 0:
        ndata = len(data)
        mean = np.mean(data)
        sig = np.sqrt(np.mean(data ** 2) - mean ** 2)

        new_data = data[np.abs(data - mean) < sig * cut]
        ncut = ndata - len(new_data)
        data = new_data

    return sig

def pitch_from_angle(degMode):
    return LAMBDA/2.0/np.sin(np.deg2rad(degMode/2.0))

def sigmay_from_modulation(modu, degMode):
    pitch = pitch_from_angle(degMode)
    return pitch/PI * np.sqrt(0.5*np.log(np.abs(np.cos(np.deg2rad(degMode))) / modu))

def modulation_from_sigmay(sigy, degMode):
    pitch = pitch_from_angle(degMode)
    return np.abs(np.cos(np.deg2rad(degMode))) * np.exp(-2*(PI*sigy/pitch)**2)

def sigmayIPBSM(modu, degMode):
    wavelength = 532e-9
    pitch = wavelength / (2.0 * np.sin(np.deg2rad(degMode / 2.0)))

    C = np.abs(np.cos(np.deg2rad(degMode)))
    val = 0.5 * np.log(C / modu)

    return pitch / np.pi * np.sqrt(val)

def macropartIPBSM_direct(data, degMode):
    pitch = pitch_from_angle(degMode)
    phase = 2*PI*data/pitch
    Pterm = np.sum(np.cos(phase))
    Qterm = np.sum(np.sin(phase))
    Cfac = np.cos(np.deg2rad(degMode))
    modulation = abs(Cfac) * np.sqrt(Pterm**2 + Qterm**2) / len(data)
    return modulation

def FuncIPBSMbeamsize(data):

    RMSbeamsize = np.std(data)  # [m]
    RMSbeamsize_nm = RMSbeamsize*1e9

    # mode selection
    if RMSbeamsize_nm >= 600:
        degMode = 2
    elif RMSbeamsize_nm >= 400:
        degMode = 4
    elif RMSbeamsize_nm >= 200:
        degMode = 8
    elif RMSbeamsize_nm >= 70:
        degMode = 30
    else:
        degMode = 174

    ModIPBSM = macropartIPBSM_direct(data, degMode)
    SigIPBSM = sigmay_from_modulation(ModIPBSM, degMode)
    return degMode, ModIPBSM, SigIPBSM

def BeamStatistics(track_bsizes):

    track_bsizes = np.array(track_bsizes)
    mean = np.mean(track_bsizes)
    stdev = np.std(track_bsizes, ddof=1)
    sdom = stdev / np.sqrt(len(track_bsizes))
    return mean, stdev, sdom