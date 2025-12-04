# ============================================================
#  ipbsm_calc.py
# Author Motoki Sato
# Date 2025/12/4
# ============================================================

import numpy as np


# ============================================================
#  RMS beam size (simple)
# ============================================================
def sigma_from_array(arr):
    """Compute RMS beam size: sqrt(<x^2> - <x>^2)."""
    if len(arr) < 2:
        return -10000.0
    mean = np.mean(arr)
    return np.sqrt(np.mean(arr ** 2) - mean ** 2)


# ============================================================
#  Core beam size (SAD sigmac)
# ============================================================
def core_sigma(arr, cut=1.5):
    """
    SAD sigmac algorithm:
    iterative trimming using |x - mean| < sigma * cut
    """
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
def macropartIPBSMdirect(data, degMode):
    """
    SAD macropartIPBSMdirect[data,degMode]
    Computes IPBSM modulation from macro-particle distribution.
    """
    wavelength = 532e-9
    pitch = wavelength / (2.0 * np.sin(np.deg2rad(degMode / 2.0)))

    # Phase term 2π y / pitch
    phase = 2.0 * np.pi * data / pitch

    Pterm = np.sum(np.cos(phase))
    Qterm = np.sum(np.sin(phase))

    Cfac = np.cos(np.deg2rad(degMode))  # mode-angle correction
    Modulation = Cfac * np.sqrt(Pterm * Pterm + Qterm * Qterm) / len(data)

    return Modulation


# ============================================================
#  FuncIPBSMbeamsize : main entry point
# ============================================================
def FuncIPBSMbeamsize(data):
    """
    SAD FuncIPBSMbeamsize[data] の完全移植版。

    Input:
        data : 1D numpy array [m]

    Output:
        degMode : 8 / 30 / 174 (depends on RMS beam size)
        ModIPBSM : IPBSM modulation
        SigIPBSM : extracted beam size [m]
    """
    # RMS beam size in nm
    RMS_nm = sigma_from_array(data) * 1e9

    # Determine degMode exactly as SAD does
    if RMS_nm >= 300:
        degMode = 8
    if RMS_nm < 300:
        degMode = 30
    if RMS_nm < 70:
        degMode = 174

    # Compute modulation
    ModIPBSM = macropartIPBSMdirect(data, degMode)

    # Compute beam size from modulation
    SigIPBSM = sigmayIPBSM(ModIPBSM, degMode)

    return degMode, ModIPBSM, SigIPBSM
