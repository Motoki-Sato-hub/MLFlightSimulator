import RF_Track as rft
import numpy as np
import time

class InterfaceATF2_Ext_RFTrack:
    def get_name(self):
        return 'ATF2_Ext_RFT'

    def __init__(self, population=2e10, jitter=0.0, bpm_resolution=0.0, nsamples=1):
        self.lattice = rft.Lattice('Ext_ATF2/ATF2_EXT_FF_v5.2.twiss')
        self.lattice.set_bpm_resolution(bpm_resolution)
        self.sequence = [ e.get_name() for e in self.lattice['*'] ]
        self.bpms = [ e.get_name() for e in self.lattice.get_bpms() ]
        self.corrs = [ e.get_name() for e in self.lattice.get_correctors() ]
        self.Pref = 1.2999999e3 # 1.3 GeV/c
        self.population = population
        self.jitter = jitter
        self.nsamples = nsamples
        self.__setup_beam0()
        self.__track_bunch()

    def __setup_beam0(self):
        T = rft.Bunch6d_twiss()
        T.emitt_x = 5.2 # mm.mrad normalised emittance
        T.emitt_y = 0.03 # mm.mrad
        T.beta_x = 6.848560987 # m
        T.beta_y = 2.935758992 # m
        T.alpha_x = 1.108024744
        T.alpha_y = -1.907222942
        T.sigma_t = 8 # mm/c
        T.sigma_pt = 0.8 # permille
        Nparticles = 10000 # number of macroparticles
        self.B0 = rft.Bunch6d_QR(rft.electronmass, self.population, -1, self.Pref, T, Nparticles)
        
    def __setup_beam1(self):
        # Beam for DFS - Reduced energy
        Pref = 0.98 * self.Pref # 98% of nominal energy
        T = rft.Bunch6d_twiss()
        T.emitt_x = 2e-3 # mm.mrad normalised emittance
        T.emitt_y = 1.179228346e-5 # mm.mrad
        T.beta_x = 6.848560987 # m
        T.beta_y = 2.935758992 # m
        T.alpha_x = 1.108024744
        T.alpha_y = -1.907222942
        T.sigma_t = 8 # mm/c
        T.sigma_pt = 0.8 # permille
        Nparticles = 10000 # number of macroparticles
        self.B0 = rft.Bunch6d_QR(rft.electronmass, self.population, -1, Pref, T, Nparticles)

    def __setup_beam2(self):
        # Beam for WFS - Reduced bunch charge
        population = 0.90 * self.population # 90% of nominal charge
        T = rft.Bunch6d_twiss()
        T.emitt_x = 2e-3 # mm.mrad normalised emittance
        T.emitt_y = 1.179228346e-5 # mm.mrad
        T.beta_x = 6.848560987 # m
        T.beta_y = 2.935758992 # m
        T.alpha_x = 1.108024744
        T.alpha_y = -1.907222942
        T.sigma_t = 8 # mm/c    
        T.sigma_pt = 0.8 # permille
        Nparticles = 10000 # number of macroparticles
        self.B0 = rft.Bunch6d_QR(rft.electronmass, population, -1, self.Pref, T, Nparticles)

    def __track_bunch(self):
        I0 = self.B0.get_info()
        dx = self.jitter*I0.sigma_x
        dy = self.jitter*I0.sigma_y
        dz, roll = 0.0, 0.0
        pitch = self.jitter*I0.sigma_py
        yaw   = self.jitter*I0.sigma_px
        B0_offset = self.B0.displaced(dx, dy, dz, roll, pitch, yaw)
        self.lattice.track(B0_offset)

    def change_energy(self, *args):
        self.__setup_beam1()

    def reset_energy(self, *args):
        self.__setup_beam0()

    def change_intensity(self, *args):
        pass

    def reset_intensity(self, *args):
        pass

    def get_sequence(self):
        return self.sequence

    def get_bpms_names(self):
        return self.bpms

    def get_correctors_names(self):
        return self.corrs

    def get_hcorrectors_names(self):
        return [string for string in self.corrs if string.lower().startswith('zh')]

    def get_vcorrectors_names(self):
        return [string for string in self.corrs if string.lower().startswith('zv')]

    def get_elements_position(self,names):
        return [index for index, string in enumerate(self.sequence) if string in names]

    def get_icts(self):
        print("Reading ict's...")
        charge = [ bpm.get_total_charge() for bpm in self.lattice.get_bpms() ]
        icts = {
            "names": self.bpms,
            "charge": charge
        }        
        return icts

    def get_correctors(self):
        print("Reading correctors' strengths...")
        bdes = np.zeros(len(self.corrs))
        for i,corrector in enumerate(self.corrs):
            if corrector[:2] == "ZH":
                bdes[i] = (self.lattice[corrector].get_strength()[0]*10)  # gauss*m
            elif corrector[:2] == "ZV":
                bdes[i] = (self.lattice[corrector].get_strength()[1]*10)  # gauss*m
        correctors = { "names": self.corrs, "bdes": bdes, "bact": bdes }
        return correctors
    
    def get_bpms(self):
        print('Reading bpms...')
        x = np.zeros((self.nsamples, len(self.bpms)))
        y = np.zeros(x.shape)
        tmit = np.zeros(x.shape)
        for i in range(self.nsamples):
            for j,bpm in enumerate(self.bpms):
                b = self.lattice[bpm]
                reading = b.get_reading()
                x[i,j] = reading[0]
                y[i,j] = reading[1]
                tmit[i,j] = b.get_total_charge()
        bpms = { "names": self.bpms, "x": x, "y": y, "tmit": tmit }
        return bpms


    # TO DO
    def push(self, names, corr_vals):
        if not isinstance(names, list):
            names = [ names ] # makes it a list
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].set_strength(val/10, 0.0)  # T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].set_strength(0.0, val/10)  # T*mm
        self.__track_bunch()
    
    def vary_correctors(self, names, corr_vals):
        if not isinstance(names, list):
            names = [ names ] # makes it a list
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].vary_strength(val/10, 0.0)  # T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].vary_strength(0.0, val/10)  # T*mm
        self.__track_bunch