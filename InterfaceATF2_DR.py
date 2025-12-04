import numpy as np
import time, math

from epics import PV, ca

class InterfaceATF2_DR:
    def get_name(self):
        return 'ATF2_DR'

    def __init__(self, nsamples=1):
        self.nsamples = nsamples
        # Bpms and correctors in beamline order
        sequence = [
            'MB1R', 'MB2R', 'ZV1R', 'ZH1R', 'MB3R', 'MB4R', 'ZV2R', 'ZH2R',
 'MB5R', 'MB6R', 'ZV3R', 'ZH3R', 'MB7R', 'MB8R', 'ZV4R', 'ZH4R',
 'MB9R', 'MB10R', 'ZV5R', 'ZH5R', 'MB11R', 'MB12R', 'ZV6R', 'ZH6R',
 'MB13R', 'MB14R', 'ZV7R', 'ZH7R', 'MB15R', 'MB16R', 'ZV8R', 'ZH8R',
 'MB17R', 'MB18R', 'ZV9R', 'ZH9R', 'MB19R', 'MBX1', 'MBX2', 'MB21R', 'MB22R', 'ZH10R', 'ZV10R',
 'MB23R', 'ZH11R', 'MB24R','ZV11R', 'MB25R', 'ZH12R', 'MB26R','ZV12R',  
 'MB27R', 'ZV13R', 'MB28R', 'ZH13R', 'MB29R', 'ZV14R', 'MB30R', 'ZH14R', 'ZV15R',
 'MB31R', 'ZV16R', 'ZH15R', 'MB32R', 'MB33R', 'ZV17R', 'ZH16R',
 'MB34R', 'ZV18R', 'MB35R', 'ZH17R', 'MB36R', 'MB37R', 'ZV19R', 'ZH18R',
 'MB38R', 'MB39R', 'ZV20R', 'ZH19R', 'MB40R', 'MB41R', 'ZV21R', 'ZH20R',
 'MB42R', 'MB43R', 'ZV22R', 'ZH21R', 'MB44R', 'MB45R', 'ZV23R', 'ZH22R',
 'MB46R', 'MB47R', 'ZV24R', 'ZH23R', 'MB48R', 'MB49R', 'ZV25R', 'ZH24R',
 'MB50R', 'MB51R', 'ZV26R', 'ZH25R', 'MB52R', 'MB53R', 'ZV27R', 'ZH26R',
 'MB54R', 'MB55R', 'ZV28R', 'ZH27R', 'MB56R', 'MB57R', 'ZV29R', 'ZH28R',
 'MB58R', 'MB59R', 'ZV30R', 'ZH29R', 'MB60R', 'MB61R', 'ZV31R', 'ZH30R',
 'MB62R', 'MB63R', 'ZV32R', 'ZH31R', 'MB64R', 'MB65R', 'ZH32R', 'ZV33R',
 'MB66R', 'ZV34R', 'MB67R', 'ZH33R', 'MB68R', 'MB69R', 'ZH34R', 'ZV35R',
 'MB70R', 'ZV36R', 'MB71R', 'ZH35R', 'ZV37R', 'MB72R', 'ZH36R', 'MB73R', 'ZV38R',
 'MB74R', 'ZH37R', 'ZV39R', 'MB76R', 'ZV40R', 'MB77R', 'ZH38R',
 'MB78R', 'ZH41R', 'MB79R', 'ZV39R', 'ZH42R', 'MB80R', 'ZV43R', 'ZH40R',
 'MB81R','MB82R', 'ZV44R', 'ZH41R', 'MB83R', 'ZV45R', 'MB84R', 'ZH42R', 'MB85R', 'MB86R',
 'ZV46R', 'ZH43R', 'MB87R', 'MB88R', 'ZV47R', 'ZH44R', 'MB89R', 'MB90R',
 'ZV48R', 'ZH45R', 'MB91R', 'MB92R', 'ZV49R', 'ZH46R', 'MB93R', 'MB94R',
 'ZV50R', 'ZH47R', 'MB95R', 'MB96R', 'ZV51R', 'ZH48R', 'MB97R', 'MB98R'
        ]

        # ATF2' BPMs Epics names
        # https://atf.kek.jp/atfbin/view/ATF/EPICS_DATABASE
        monitors = [
            'MB1R', 'MB2R', 'MB3R', 'MB4R','MB5R', 'MB6R', 'MB7R', 'MB8R',
'MB9R', 'MB10R', 'MB11R', 'MB12R','MB13R', 'MB14R', 'MB15R', 'MB16R', 'MB17R', 'MB18R', 'MB19R', 'MBX1', 'MBX2', 'MB21R', 'MB22R',
'MB23R', 'MB24R','MB25R', 'MB26R',  'MB27R', 'MB28R', 'MB29R', 'MB30R', 'MB31R', 'MB32R', 'MB33R',
'MB34R', 'MB35R', 'MB36R', 'MB37R','MB38R', 'MB39R', 'MB40R', 'MB41R', 'MB42R', 'MB43R', 'MB44R', 'MB45R', 
'MB46R', 'MB47R', 'MB48R', 'MB49R', 'MB50R', 'MB51R', 'MB52R', 'MB53R','MB54R', 'MB55R', 'MB56R', 'MB57R', 'MB58R', 'MB59R', 'MB60R', 'MB61R', 'MB62R', 'MB63R', 'MB64R', 'MB65R', 'MB66R', 'MB67R', 'MB68R', 'MB69R', 'MB70R', 'MB71R', 'MB72R', 'MB73R', 'MB74R', 'MB76R', 'MB77R', 'MB78R', 'MB79R', 'MB80R', 'MB81R','MB82R', 'MB83R', 'MB84R', 'MB85R', 'MB86R', 'MB87R', 'MB88R', 'MB89R', 'MB90R','MB91R', 'MB92R', 'MB93R', 'MB94R','MB95R', 'MB96R', 'MB97R', 'MB98R'
        ]
        monitors = ['MB1R', 'MB3R', 'MB4R', 'MB5R', 'MB7R', 'MB8R', 'MB9R', 'MB10R', 'MB11R', 'MB12R', 'MB13R', 'MB14R', 'MB15R', 'MB16R', 'MB18R', 'MB21R', 'MB22R', 'MB24R', 'MB25R', 'MB26R', 'MB27R', 'MB28R', 'MB29R', 'MB31R', 'MB33R', 'MB34R', 'MB35R', 'MB36R', 'MB37R', 'MB38R', 'MB40R', 'MB41R', 'MB42R', 'MB43R', 'MB44R', 'MB45R', 'MB48R', 'MB49R', 'MB50R', 'MB51R', 'MB52R', 'MB53R', 'MB54R', 'MB55R', 'MB56R', 'MB58R', 'MB59R', 'MB61R', 'MB62R', 'MB63R', 'MB64R', 'MB65R', 'MB66R', 'MB67R', 'MB68R', 'MB69R', 'MB71R']
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
        self.phase_kl1 = PV('CM1L:phaseRead').get()
        self.laser_intensity = PV('RFGun:LasetIntensity1:Read').get()

    def change_energy(self, rel_phase=5,scale=None, **kwargs):
        pv = PV('CM1L:phaseWrite')
        pv.put(self.phase_kl1 + rel_phase)
        time.sleep(1)

    def reset_energy(self,**kwargs):
        pv = PV('CM1L:phaseWrite')
        pv.put(self.phase_kl1)
        time.sleep(1)
        
    def change_intensity(self, laserintensity=0.15,**kwargs):
        print(f'Changing laser intensity to {laserintensity}...')
        self.laser_intensity = float(PV('RFGun:LasetIntensity1:Read').get())
        laser_intensity = laserintensity * 100 * 5 # Korysko dixit: 100 for percent, 5 convesion factor
        PV('RFGun:LaserIntensity1:Write').put(laser_intensity)
        time.sleep(3)
        
        return self

    def reset_intensity(self,**kwargs):
        print('Resetting laser intensity...')
        self.change_intensity(self, laserintensity=self.laser_intensity / 500)
        return self

    def get_sequence(self, *args):
        return self.sequence

    def get_bpms_names(self, *args):
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
            if 0: # Reading the icts is time consuming and unnecessary for SysID and BBA
                charge.append(pv.get())
            else:
                charge.append(1.0)
        print("ICT's read.")
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
        p = PV('DR:monitors')
        x, y, tmit = [], [], []
        for sample in range(self.nsamples):
            print(f'Sample = {sample}')
            a = p.get().reshape((-1, 10))
            a = a[a[:,0]==1,:]
            a = a[a[:,3]>0,:]
            x.append(a[:, 1])
            y.append(a[:, 2])
            tmit.append(a[:, 3])
            time.sleep(1)
        names = [ self.bpms ] if type(self.bpms) == str else self.bpms
        print(x)
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
