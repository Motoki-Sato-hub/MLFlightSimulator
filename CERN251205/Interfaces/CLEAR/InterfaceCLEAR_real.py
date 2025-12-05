import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
class CLEAR_real_machine:
    def JapcReadback(japc_address, japc_selector):
        try:
            import jpype
            from pyjapc import PyJapc
            japc = PyJapc(incaAcceleratorName='CTF', selector='')
            if not jpype.isJVMStarted():
                jpype.startJVM()
            time.sleep(1)
            try:
                japc.setSelector(japc_selector)
                data = japc.getParam(japc_address)
                return data
            except:
                print(f'Hardware {japc_address} not found')
                return None
        except ImportError:
            print('pyjapc not found. Try install pyjapc first')
        return data

    def SimulationReedback(japc_address, lattice_configuration):
        try:
            import RF_Track
            if japc_address in list(lattice_configuration.keys()):
                data = list(lattice_configuration.values())
                return data
            else:
                print(f'Hardware {japc_address} not found in the simulation')
                return None
        except ImportError:
            print('RF_Track not found. Try install RF_Track first')

    def acq_params(japc_address, japc_selector = 'VIRTUAL', lattice_configuration = 'CLEAR2024'):
        if  japc_selector == 'VIRTUAL':
            data = SimulationReedback[japc_address, lattice_configuration]
        else:
            data = JapcReadback[japc_address, japc_selector]
        return data


    japc_address_des = 'CA.DHG0225/Acquisition#currentAverage'
    japc_address_act = JapcReadback.DHG0225

    japc_selector = 'SCT.USER.SETUP'
    japc.setSelector(japc_selector)
    bdes = japc.getParam(japc_address_des)

    time.sleep(1)
    japc_selector = japc_address_act[1]
    bdact = japc.getParam(japc_address_act[0])
    correctors = { "names": japc_address_des[3:10], "bdes": bdes, "bact": bdact }

    #data = JapcReadback[japc_address, japc_selector]