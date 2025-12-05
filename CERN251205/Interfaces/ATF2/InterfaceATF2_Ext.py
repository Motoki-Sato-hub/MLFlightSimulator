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

    def change_energy(self, delta_freq=None, **kwargs):
        # some parsing to extract delta_freq
        delta_freq = -2

        # test for valid range and integer value
        if delta_freq != np.int32(delta_freq):
            raise Exception('DR frequency change is not an integer: %s',delta_freq)

        if (-5 > delta_freq) or (delta_freq > 5):
            raise Exception('DR frequency change is out of a safe range: %d', delta_freq)

        pv = PV('atf:rfRamp:sw')
        pv.put(1 if delta_freq else 0)
        time.sleep(2)

        pv = PV('atf:rfRamp:freq:set')
        pv.put(delta_freq)
        time.sleep(2)

    def reset_energy(self,**kwargs):
        pv = PV('atf:rfRamp:sw')
        pv.put(0)
        time.sleep(2)

        pv = PV('atf:rfRamp:freq:set')
        pv.put(0)
        time.sleep(2)

    def change_intensity(self, laserintensity =0.1, ang_offset = 2.0, **kwargs):

        start = time.perf_counter()

        laserintensity = 0.1

        if laserintensity <= 0.0:
            raise ValueError('Laser intensity should be strict positive')

        elif laserintensity >= 1.0:
            raise ValueError(f'Laser intensity should be below 1: {laserintensity}')

        elif laserintensity >= 0.30:
            raise ValueError(f'Laser intensity should be below 0.3 - internal warning {laserintensity}')

        else:
            ang_offset = 2.0

            angle = 180.0 / np.pi / 4.0 * math.acos(2.0 * laserintensity - 1.0) - ang_offset

            ### calculate current angle

            # read x_counter

            x_counter = PV('INJ:LaserIntensityXcount').get()

            #os.system('caget -# 1 INJ:LaserIntensityXcount > /tmp/bba_laserxcount.txt')
            #[name, dummy, x_counter] = np.genfromtxt('/tmp/bba_laserxcount.txt', '%s %f %f')

            print(x_counter)

            angle_read = np.mod(x_counter, 72000) / 200.0 + ang_offset

            if not hasattr(self, '_angle_before'):
                self._angle_before = angle_read

            pulse = np.int32((angle - angle_read) * 200)

            if pulse >= 0:
                print('laser up')
                PV('INJ:setLaserIntUpAngle').put(int(pulse))
                #os.system(f"caput INJ:setLaserIntUpAngle {int(pulse)}")
            else:
                print('laser down')
                pulse = -pulse
                PV('INJ:setLaserIntDownAngle').put(int(pulse))
                #os.system(f'caput INJ:setLaserIntDownAngle {int(pulse)}')

            PV('INJ:setLaserIntSend').put(1)
            #os.system('caput INJ:setLaserIntSend 1')
            time.sleep(0.5)
            PV('INJ:setLaserIntSend').put(1)
            #os.system('caput INJ:setLaserIntSend.PROC 1')

            time.sleep(3)
            elapsed = time.perf_counter() - start
            print('InterfaceATF2::ChangeBunchCharge()', elapsed)
            return self


    def reset_intensity(self, ang_offset=None,**kwargs):
        start = time.perf_counter()

        ang_offset = 2.0

        angle = self._angle_before

        ### calculate current angle

        # read x_counter

        x_counter = PV('INJ:LaserIntensityXcount').get()
        print(x_counter)

        #os.system('caget -# 1 INJ:LaserIntensityXcount > /tmp/bba_laserxcount.txt')
        #[name, dummy, x_counter] = np.genfromtxt('/tmp/bba_laserxcount.txt', '%s %f %f')

        angle_read = np.mod(x_counter, 72000) / 200.0 + ang_offset

        pulse = int((angle - angle_read) * 200)

        if pulse >= 0:
            print('laser up')
            PV('INJ:setLaserIntUpAngle').put(int(pulse))
            #os.system(f"caput INJ:setLaserIntUpAngle {int(pulse)}")
        else:
            print('laser down')
            pulse = -pulse
            PV('INJ:setLaserIntDownAngle').put(int(pulse))
            #os.system(f'caput INJ:setLaserIntDownAngle {int(pulse)}')

        PV('INJ:setLaserIntSend').put(1)
        #os.system('caput INJ:setLaserIntSend 1')
        time.sleep(0.5)

        PV('INJ:setLaserIntSend.PROC').put(1)
        #os.system('caput INJ:setLaserIntSend.PROC 1')

        time.sleep(3)
        elapsed = time.perf_counter() - start
        print('InterfaceATF2::ChangeBunchCharge()', elapsed)
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
