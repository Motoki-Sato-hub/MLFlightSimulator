from datetime import datetime
import numpy as np
import pickle

class State:
    def __init__(self, interface=None, filename=None):
        if filename is not None:
            self.load(filename)
        elif interface is not None:
            self.pull(interface)

    def pull (self, interface):
        self.correctors = interface.get_correctors()
        self.bpms = interface.get_bpms()
        self.icts = interface.get_icts()
        self.sequence = interface.get_sequence()
        self.hcorrectors_names = interface.get_hcorrectors_names()
        self.vcorrectors_names = interface.get_vcorrectors_names()
        self.timestamp = datetime.now()

    def push(self, interface):
        interface.push(self.correctors['names'], self.correctors['bdes']) #sets the desired current for one or more correctors

    def get_sequence(self):
        return self.sequence #from rf track

    def get_correctors(self, names=None):
        if names is not None:
            corr_indexes = np.array([index for index, string in enumerate(self.correctors['names']) if string in names])
            correctors = {
                "names": np.array(self.correctors['names'])[corr_indexes],
                "bdes": np.array(self.correctors['bdes'])[corr_indexes],
                "bact": np.array(self.correctors['bact'])[corr_indexes]
            }
        else:
            correctors = self.correctors
        return correctors

    def get_hcorrectors_names(self):
        return self.hcorrectors_names

    def get_vcorrectors_names(self):
        return self.vcorrectors_names

    def get_bpms(self, names=None):
        if names is not None:
            bpm_indexes = np.array([index for index, string in enumerate(self.bpms['names']) if string in names])
            bpms = {
                "names": np.array(self.bpms['names'])[bpm_indexes],
                "x": np.array(self.bpms['x'])[:,bpm_indexes],
                "y": np.array(self.bpms['y'])[:,bpm_indexes],
                "tmit": np.array(self.bpms['tmit'])[:,bpm_indexes],
                "S": np.array(self.bpms['S'])[bpm_indexes]
            }
        else:
            bpms = self.bpms
        return bpms         
    
    def get_bpms_names(self, interface):
        return interface.get_bpms_names()

    def get_icts(self, names=None):
        icts = self.icts
        if names is not None:
            ict_indexes = np.array([index for index, string in enumerate(icts['names']) if string in names])
            icts = {
                "names": np.array(self.icts['names'])[ict_indexes],
                "charge": np.array(self.icts['charge'])[ict_indexes]
            }
        return icts         

    def get_orbit(self, names=None):
        bpms = self.get_bpms(names)
        x = np.mean(bpms['x'],axis=0) # mm
        y = np.mean(bpms['y'],axis=0) # mm
        stdx = np.std(bpms['x'],axis=0) # mm #standard deviation
        stdy = np.std(bpms['y'],axis=0) # mm
        tmit = np.mean(bpms['tmit'],axis=0)
        faulty = np.isnan(x) | np.isnan(y)
        x[faulty] = np.nan
        y[faulty] = np.nan
        orbit = { "names": names, "x": x, "y": y, "stdx": stdx, "stdy": stdy, "tmit": tmit, "faulty": faulty, "nbpms": len(bpms["names"]) }
        return orbit


    def change_energy(self, interface):
        interface.change_energy()
        pass

    def reset_energy(self, interface):
        interface.reset_energy()
        pass

    def change_intensity(self, interface):
        interface.change_intensity()
        pass

    def reset_intensity(self, interface, *args):
        interface.reset_intensity(*args)
        pass
    """""
    def push(self, interface, names, corr_vals):
        interface.push(names, corr_vals)
    """""

    def vary_correctors(self, interface, names, corr_vals):
        interface.vary_correctors(names, corr_vals)

    def measure_orbit(self, interface, bpm_names=None):
        self.pull(interface)
        orbit = self.get_orbit(bpm_names)
        try:
            s = np.asarray(interface.get_bpms_S(), dtype=float)
        except Exception:
            s = np.arange(len(orbit["x"]), dtype=float)

        self.orbit = {
            "s": s,
            **orbit,
            "timestamp": self.timestamp,
        }
        return self.orbit
    
    def get_element_S(self, interface, name):
        return interface.get_element_S(name)
    
    def init_knobs(self, interface):
        self.linear_knobs = {k: 0.0 for k in interface.get_linear_knob_names()}
        self.nonlinear_knobs = {k: 0.0 for k in interface.get_nonlinear_knob_names()}

    def set_linear_knob(self, interface, name, value):
        if not hasattr(self, "linear_knobs"):
            self.init_knobs(interface)
        interface.set_linear_knob(name, value)
        self.linear_knobs[name] = value
        #self.pull(interface)


    def set_nonlinear_knob(self, interface, name, value):
        if not hasattr(self, "nonlinear_knobs"):
            self.init_knobs(interface)
        interface.set_nonlinear_knob(name, value)
        self.nonlinear_knobs[name] = value
        #self.pull(interface)

    def apply_qmag_current(self, interface, name, dA):
        interface.apply_qmag_current(name, dA)
        #self.pull(interface)

    def apply_sum_knob(self, interface, dA):
        interface.apply_sum_knob(dA)
        #self.pull(interface)

    def measure_ipbsm(self, interface):
        state = interface.get_ipbsm_state()
        self.ipbsm = {
            **state,
            "timestamp": self.timestamp,
        }
        return self.ipbsm
    
    def get_ipbsm_state(self, interface):
        return interface.get_ipbsm_state()



    def load(self, filename):
        from glob import glob
        f = glob(f'{filename}*')
        try:
            with open(f[0], "rb") as pickle_file:
                data = pickle.load(pickle_file)
            self.sequence = data['sequence']
            self.correctors = data['correctors']
            self.bpms = data['bpms']
            self.icts = data['icts']
            """
            self.correctors = {
                "names": data['correctors']['names']
                "bdes": data['correctors']['bdes']
                "bact": data['correctors']['bact']
            }
            self.bpms = {
                "names": data['bpms']['names'],
                "x": data['bpms']['x'],
                "y": data['bpms']['y'],
                "tmit": data['bpms']['tmit']
            }
            self.icts = {
                "names": data['icts']['names'],
                "charge": data['icts']['charge']
            }
            """
            self.hcorrectors_names = data['hcorrectors_names']
            self.vcorrectors_names = data['vcorrectors_names']
            self.timestamp = datetime.strptime(data['timestamp'], "%Y/%m/%d, %H:%M:%S")
        except Exception:
            raise Exception(f"Could not load {filename}")

    def save(self, basename=None, filename=None):
        if basename is not None:
            time_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{basename}_{time_str}.pkl"
        correctors = {
            'names': self.correctors['names'],
            'bdes': self.correctors['bdes'], #setpoint for a corrector
            'bact': self.correctors['bact'] #readback for a corr
        }
        bpms = {
            'names': self.bpms['names'],
            'x': self.bpms['x'],
            'y': self.bpms['y'],
            'tmit': self.bpms['tmit'] #it's a local intensity at each bpm
        }
        icts = { #tells us about intensity
            'names': self.icts['names'],
            'charge': self.icts['charge'] #intensity
        }
        state = {
            "sequence": self.sequence,
            "correctors": correctors,
            "bpms": bpms,
            "icts": icts,
            "hcorrectors_names": self.hcorrectors_names,
            "vcorrectors_names": self.vcorrectors_names,
            "timestamp": self.timestamp.strftime("%Y/%m/%d, %H:%M:%S")
        }
        with open(filename, "wb") as file:
            pickle.dump(state, file)
        return filename
            
    def push(self, interface):
        interface.push(self.correctors['names'], self.correctors['bdes']) #restores, because errors would add up i think
