import sys, time, math,os
import numpy as np
from epics import PV, ca

class InterfaceATF2_Ext:
    def get_name(self):
        return 'ATF2_Ext'

    def __init__(self, nsamples=1):
        self.nsamples = nsamples
        # Bpms and correctors in beamline order
        sequence = [
            "MB2X", "ZV1X", "MQF1X", "ZV2X", "MQD2X", "MQF3X", "ZH1X", "ZV3X", "MQF4X",
            "ZH2X", "MQD5X", "ZV4X", "ZV5X", "MQF6X", "MQF7X", "ZH3X", "MQD8X", "ZV6X",
            "MQF9X", "ZH4X", "FONTK1", "ZV7X", "FONTP1", "MQD10X", "ZH5X", "MQF11X",
            "FONTK2", "ZV8X", "FONTP2", "MQD12X", "ZH6X", "MQF13X", "MQD14X", "FONTP3",
            "ZH7X", "MQF15X", "ZV9X", "MQD16X", "ZH8X", "MQF17X", "ZV10X", "MQD18X",
            "ZH9X", "MQF19X", "ZV11X", "MQD20X", "ZH10X", "MQF21X", "IPT1", "IPT2",
            "IPT3", "IPT4", "MQM16FF", "ZH1FF", "ZV1FF", "MQM15FF", "MQM14FF", "FB2FF",
            "MQM13FF", "MQM12FF", "MQM11FF", "MQD10BFF", "MQD10AFF", "MQF9BFF",
            "MSF6FF", "MQF9AFF", "MQD8FF", "MQF7FF", "MQD6FF", "MQF5BFF", "MSF5FF",
            "MQF5AFF", "MQD4BFF", "MSD4FF", "MQD4AFF", "MQF3FF", "MQD2BFF", "MQD2AFF",
            "MSF1FF", "MQF1FF", "MSD0FF", "MQD0FF", "PREIP", "IPA", "IPB", "IPC", "M-PIP"
        ]
        # ATF2' BPMs Epics names
        # https://atf.kek.jp/atfbin/view/ATF/EPICS_DATABASE
        '''
        monitors = [
            "MB1X", "MB2X", "MQF1X", "MQD2X", "MQF3X", "MQF4X", "MQD5X", "MQF6X",
            "MQF7X", "MQD8X", "MQF9X", "MQD10X", "MQF11X", "MQD12X", "MQF13X",
            "MQD14X", "MQF15X", "MQD16X", "MQF17X", "MQD18X", "MQF19X", "MQD20X",
            "MQF21X", "IPBPM1", "IPBPM2", "nBPM1", "nBPM2", "nBPM3", "MQM16FF",
            "MQM15FF", "MQM14FF", "MFB2FF", "MQM13FF", "MQM12FF", "MFB1FF",
            "MQM11FF", "MQD10BFF", "MQD10AFF", "MQF9BFF", "MSF6FF", "MQF9AFF",
            "MQD8FF", "MQF7FF", "MQD6FF", "MQF5BFF", "MSF5FF", "MQF5AFF",
            "MQD4BFF", "MSD4FF", "MQD4AFF", "MQF3FF", "MQD2BFF", "MQD2AFF",
            "MSF1FF", "MQF1FF", "MSD0FF", "MQD0FF", "M1&2IP", "MPIP", "MDUMP",
            "ICT1X", "ICTDUMP", "MW1X", "MW1IP", "MPREIP", "MIPA", "MIPB"
        ]
        '''
        monitors = [
            "MB1X", "MB2X", "MQF1X", "MQD2X", "MQF3X", "MQF4X", "MQD5X", "MQF6X",
            "MQF7X", "MQD8X", "MQF9X", "MQD10X", "MQF11X", "MQD12X", "MQF13X",
            "MQD14X", "MQF15X", "MQD16X", "MQF17X", "MQD18X", "MQF19X",
            "MQF21X", "MQM16FF",
            "MQM15FF", "MQM14FF",
            "MQM11FF", "MQD10BFF", "MQD10AFF", "MQF9AFF",
            "MQD8FF", "MQF7FF", "MQF5BFF", "MQD4BFF", "MQF3FF", "MQD2BFF", "MQD2AFF",
            "MSF1FF", "MPREIP", "MIPB"
            ]

        # Use list comprehension to filter out strings starting with 'Z' or 'z'
        monitors_from_sequence = [string for string in sequence if not string.lower().startswith('z')]
        # Check if the bpms in the config files are known to Epics
        bpm_ok = all(bpm in monitors for bpm in monitors_from_sequence)
        if not bpm_ok:
            bpms_unknown = [bpm for bpm in monitors_from_sequence if bpm not in monitors]
            print(f'Unknown bpms {bpms_unknown} removed from list')
        # Only retain BPMs in config file which are known by Epics
        sequence_filtered = [element for element in sequence if (element in monitors) or element.lower().startswith('z')]
        # Subset of BPMs and correctors from the config file
        self.sequence = sequence_filtered
        self.bpms = [string for string in self.sequence if not string.lower().startswith('z')]
        self.corrs = [string for string in self.sequence if string.lower().startswith('z')]
        # Index of the selected BPMs in the Epics PV ATF2:monitors
        self.bpm_indexes = [index for index, string in enumerate(monitors) if string in self.bpms]
        # Bunch current monitors
        self.ict_names = [
            'gun:GUNcharge', 'l0:L0charge', 'linacbt:LNEcharge', 'linacbt:BTMcharge',
            'ext:EXTcharge', 'linacbt:BTEcharge', 'BIM:DR:nparticles', 'BIM:IP:nparticles'
        ]
        self.laser_intensity = PV('RFGun:LasetIntensity1:Read').get()

    def change_energy(self, delta_freq=None, **kwargs):
        PV('RAMP:CONTROL_ON_SW').put(1)
        time.sleep(2)
        ### delta_freq MUST MATCH :MI2: to EPICS --> means "MINUS2"
        delta_freq = +4 # kHz
        # PV('RAMP:MI2:ONOFF_SW').put(1)
        PV('RAMP:PL4:ONOFF_SW').put(1)
        time.sleep(2)
        DR_freq = 714e3; # 714 MHz in kHz
        DR_momentum_compaction = 2.1e-3
        dP_P = -delta_freq / DR_freq / DR_momentum_compaction
        return dP_P
    
    def reset_energy(self,**kwargs):
        PV('RAMP:CONTROL_OFF_SW').put(0)
        time.sleep(2)

    def change_intensity(self, laserintensity,**kwargs):
        print(f'Changing laser intensity to {laserintensity}...')
        self.laser_intensity = float(PV('RFGun:LaserIntensity1:Read').get())
        laser_intensity = laserintensity * 100 * 5 # Korysko dixit: 100 for percent, 5 convesion factor
        PV('RFGun:LaserIntensity1:Write').put(laser_intensity)
        time.sleep(3)
        return self

    def reset_intensity(self,**kwargs):
        print('Resetting laser intensity...')
        self.change_intensity(laserintensity=self.laser_intensity / 500)
        return self
    
    def get_sequence(self):
        return self.sequence

    def get_bpms_names(self):
        return self.bpms

    def get_correctors_names(self):
        return self.corrs

    def get_hcorrectors_names(self):
        return [string for string in self.corrs if (string.lower().startswith('zh')) or (string.lower().startswith('zx'))]

    def get_vcorrectors_names(self):
        return [string for string in self.corrs if string.lower().startswith('zv')]

    def get_elements_position(self,names):
        return [index for index, string in enumerate(self.sequence) if string in names]
    
    def get_bpms_S(self):
        p = PV('LINAC:monitors')
        a = p.get().reshape((-1, 20))
        return np.asarray(a[self.bpm_indexes, 4], dtype=float)


    def get_element_S(self, name):
        raise NotImplementedError("get_element_S not available in ATF2_Ext")

    def get_ipbsm_state(self):
        raise NotImplementedError("IPBSM not available in ATF2_Ext")

    def get_linear_knob_names(self):
        return []

    def get_nonlinear_knob_names(self):
        return []


    def get_target_dispersion(self, names=None):
        with open('Interfaces/ATF2/Ext_ATF2/ATF2_EXT_FF_v5.2.twiss', "r") as file:
            lines = [line.strip() for line in file if line.strip()]

        star_symbol = next(i for i, line in enumerate(lines) if line.startswith("*"))
        dollar_sign = next(i for i, line in enumerate(lines) if line.startswith("$") and i > star_symbol)
        columns = lines[star_symbol].lstrip("*").split()

        DX_column = columns.index("DX")
        DY_column = columns.index("DY")
        elements_names = columns.index("NAME")

        target_disp_x, target_disp_y = [], []
        for line in lines[dollar_sign + 1:]:
            data = line.split()
            bpms_name = data[elements_names].strip('"')

            if names == None or bpms_name in names:
                target_disp_x.append(float(data[DX_column]))
                target_disp_y.append(float(data[DY_column]))

        return target_disp_x, target_disp_y
    
    def get_icts(self):
        print("Reading ict's...")
        charge = []
        for ict in self.ict_names:
            pv = PV(f'{ict}')
            charge.append(pv.get())
        names = [ self.ict_names ] if type(self.ict_names) == str else self.ict_names
        charge = np.array(charge)
        icts = { "names": names, "charge": charge }
        return icts

    def get_correctors(self):
        print("Reading correctors' strengths...")
        bdes, bact = [], []
        for corrector in self.corrs:
            pv_des = PV(f'{corrector}:currentWrite')
            pv_act = PV(f'{corrector}:currentRead')
            bdes.append(pv_des.get())
            bact.append(pv_act.get())
        names = [ self.corrs ] if type(self.corrs) == str else self.corrs
        bdes = np.array(bdes)
        bact = np.array(bact)
        correctors = { "names": names, "bdes": bdes, "bact": bact }
        return correctors
    
    def get_bpms(self):
        print('Reading bpms...')
        p = PV('LINAC:monitors')
        x, y, tmit = [], [], []
        for sample in range(self.nsamples):
            print(f'Sample = {sample}')
            a = p.get().reshape((-1, 20))
            status = a[self.bpm_indexes, 0]
            # Set elements that are not equal to 1 to zero
            status[status != 1] = 0
            x.append(a[self.bpm_indexes, 1])
            y.append(a[self.bpm_indexes, 2])
            tmit.append(status * a[self.bpm_indexes, 3])
            time.sleep(1)
        names = [ self.bpms ] if type(self.bpms) == str else self.bpms
        x = np.vstack(x) / 1e3 # mm
        y = np.vstack(y) / 1e3 # mm
        tmit = np.vstack(tmit)
        bpms = { "names": names, "x": x, "y": y, "tmit": tmit }
        return bpms

    def push(self, names, corr_vals):
        if type(corr_vals) == float:
            corr_vals = np.array([corr_vals])
        if type(names) == str:
            names = [names]
        if len(names) != corr_vals.size:
            print('Error: len(names) != len(corr_vals) in push(names, corr_vals)') 
        for corrector, corr_val in zip(names, corr_vals):
            pv_des = PV(f'{corrector}:currentWrite')
            pv_des.put(corr_val)
        time.sleep(1)
    
    def vary_correctors(self, names, corr_vals):
        if type(corr_vals) is float:
            corr_vals = np.array([corr_vals])
        if type(names) == str:
            names = [names]
        if len(names) != corr_vals.size:
            print('Error: len(names) != len(corr_vals) in vary_correctors(names, corr_vals)') 
        for corrector, corr_val in zip(names, corr_vals):
            pv_des = PV(f'{corrector}:currentWrite')
            curr_val = pv_des.get()
            pv_des.put(curr_val + corr_val)
        time.sleep(1)
