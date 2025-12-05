# ============================================================
#  ipbsm_calc.py
# Author Motoki Sato
# Date 2025/12/4
# ============================================================

import numpy as np

PI = np.pi
LAMBDA = 532e-9  # laser wavelength [m]

# ============================================================
#  RMS beam size (simple)
# ============================================================
def sigma_from_array(arr):
    """Compute RMS beam size: sqrt(<y^2> - <y>^2)."""
    if len(arr) < 2:
        return -10000.0
    mean = np.mean(arr)
    return np.sqrt(np.mean(arr ** 2) - mean ** 2)


# ============================================================
#  Core beam size (SAD sigmac)
# ============================================================
def core_sigma(arr, cut=2.0):

    data = arr.copy()
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


# ============================================================
#  sigmayIPBSM : analytic inversion from modulation → sigma
# ============================================================
def sigmayIPBSM(modu, degMode):
    """
    SAD: sigmayIPBSM[modu,degMode]
    pitch = λ / (2 sin(θ/2)), λ = 532 nm
    output: sigma in [m]
    """
    wavelength = 532e-9
    pitch = wavelength / (2.0 * np.sin(np.deg2rad(degMode / 2.0)))

    C = np.abs(np.cos(np.deg2rad(degMode)))
    val = 0.5 * np.log(C / modu)

    return pitch / np.pi * np.sqrt(val)


# ============================================================
#  macropartIPBSMdirect : modulation calculation
# ============================================================
def macropartIPBSM_direct(data, degMode):
    """
    data: ndarray of y positions [m]
    degMode: mode angle
    """
    pitch = pitch_from_angle(degMode)
    phase = 2*PI*data/pitch
    Pterm = np.sum(np.cos(phase))
    Qterm = np.sum(np.sin(phase))
    Cfac = np.cos(np.deg2rad(degMode))
    modulation = abs(Cfac) * np.sqrt(Pterm**2 + Qterm**2) / len(data)
    return modulation


# ============================================================
#  FuncIPBSMbeamsize : main entry point
# ============================================================
def FuncIPBSMbeamsize(data):
    """
    data: ndarray of y positions [m]
    return: (degMode, modulation, sigma_y [m])
    """
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
    """
    track_bsizes: list or ndarray of beam sizes (float)
    return: (mean, stdev, sdom)
    """
    track_bsizes = np.array(track_bsizes)
    mean = np.mean(track_bsizes)
    stdev = np.std(track_bsizes, ddof=1)
    sdom = stdev / np.sqrt(len(track_bsizes))
    return mean, stdev, sdom