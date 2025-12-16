import RF_Track as rft
import numpy as np
import time
from . import ipbsm_calc
from .knobs import KnobSystem
from LogConsole_BBA import LogConsole
from datetime import datetime

class InterfaceATF2_Ext_RFTrack():
    def get_name(self):
        return 'ATF2_Ext_RFT'

    def __init__(self, population=2e10, jitter=0.0, bpm_resolution=0.0, nsamples=1):
        self.log = print
        self.twiss_path = 'Interfaces/ATF2/Ext_ATF2/ATF2_EXT_FF_v5.2.twiss'
        self.lattice = rft.Lattice(self.twiss_path)
        self.lattice.set_bpm_resolution(bpm_resolution)
        for s in self.lattice['*OTR*']:
            screen = rft.Screen()
            screen.set_name(s.get_name())
            s.replace_with(screen)
        Scr = rft.Screen()
        self.lattice['IP'].replace_with(Scr)
        self.sequence = [ e.get_name() for e in self.lattice['*']]
        self.bpms = [ e.get_name() for e in self.lattice.get_bpms()]
        self.corrs = [ e.get_name() for e in self.lattice.get_correctors()]
        self.screens = [ e.get_name() for e in self.lattice.get_screens()]
        self.Pref = 1.2999999e3 # 1.3 GeV/c
        self.population = population
        self.jitter = jitter
        self.nsamples = nsamples
        self.__setup_beam0()
        self.__track_bunch()
        self.dfs_test_energy = 0.98
        self.wfs_test_charge = 0.90

        
        self.knobs = KnobSystem(self.lattice, p_ref=-self.Pref)
        
        self.kl_per_A = {
            "ZH100RX": 0.0007311,
            "ZH101RX": 0.0002322,
            "ZV100RX": 0.0002764,
            "ZV1X": 0.0003276,
            "ZX1X": 0.0,
            "ZV2X": 0.0003276,
            "ZH1X": 0.0003018,
            "ZV3X": 0.0003276,
            "ZH2X": 0.0003018,
            "ZV4X": 0.0003276,
            "ZX2X": 0.0,
            "ZV5X": 0.0003276,
            "ZH3X": 0.0003018,
            "ZV6X": 0.0003276,
            "ZH4X": 0.0003018,
            "ZV7X": 0.0003276,
            "ZH5X": 0.0003018,
            "ZV8X": 0.0003276,
            "ZH6X": 0.0003018,
            "ZH7X": 0.0003018,
            "ZV9X": 0.0003276,
            "ZH8X": 0.0003018,
            "ZV10X": 0.0003276,
            "ZH9X": 0.0003018,
            "ZV11X": 0.0003276,
            "ZH10X": 0.0003018,
            "ZH1FF": 0.0003018,
            "ZV1FF": 0.0003276,
            "IPKICK": 0.0,
            "QS1X": -0.0051397,
            "QS2X": -0.0051397,
            "QF6X": -0.0216857,
            "QF1FF": -0.0061125,
            "QD0FF": 0.0070313,

        }

        self.Qmagnames = ['QS1X', 'QF1X', 'QD2X', 'QF3X', 'QF4X', 'QD5X', 'QF6X', 'QS2X', 'QF7X', 'QD8X', 'QF9X', 'QK1X', 'QD10X', 'QF11X', 'QK2X', 'QD12X', 'QF13X', 'QD14X', 'QF15X', 'QK3X', 'QD16X', 'QF17X', 'QK4X', 'QD18X', 'QF19X', 'QD20X', 'QF21X', 'QM16FF', 'QM15FF', 'QM14FF', 'QM13FF', 'QM12FF', 'QM11FF', 'QD10BFF', 'QD10AFF', 'QF9BFF', 'QF9AFF', 'QD8FF', 'QF7FF', 'QD6FF', 'QF5BFF', 'QF5AFF', 'QD4BFF', 'QD4AFF', 'QF3FF', 'QD2BFF', 'QD2AFF', 'QF1FF', 'QD0FF']
    
    def _I_to_KL(self, name, I):
        if name not in self.kl_per_A:
            raise KeyError(f"kl_per_A is not defined for corrector '{name}'")
        return np.asarray(I, dtype=float) * self.kl_per_A[name]

    def _KL_to_I(self, name, kl):
        if name not in self.kl_per_A:
            raise KeyError(f"kl_per_A is not defined for corrector '{name}'")
        return np.asarray(kl, dtype=float) / self.kl_per_A[name]


    
    def _build_lattice(self):
        self.lattice = rft.Lattice(self.twiss_path)
        Scr = rft.Screen()
        self.lattice['IP'].replace_with(Scr)
        self.lattice.set_bpm_resolution(self.bpm_resolution)

        self.sequence = [e.get_name() for e in self.lattice["*"]]
        self.bpms = [e.get_name() for e in self.lattice.get_bpms()]
        self.corrs = [e.get_name() for e in self.lattice.get_correctors()]

        self.__setup_beam0()
        self.__track_bunch()

    def log_messages(self,console):
        self.log=console or print

    

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
        Pref= self.dfs_test_energy * self.Pref
        #Pref = 0.98 * self.Pref # 98% of nominal energy
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
        self.B0 = rft.Bunch6d_QR(rft.electronmass, self.population, -1, Pref, T, Nparticles)

    def __setup_beam2(self):
        # Beam for WFS - Reduced bunch charge
        population= self.wfs_test_charge * self.population
        #population = 0.90 * self.population # 90% of nominal charge
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
        I=B0_offset.get_info()
        # print("Emittance after tracking:")
        # print(f"εx = {I.emitt_x}[mm.rad]")
        # print(f"εy = {I.emitt_y}[mm.rad]")
        # print(f"εz = {I.emitt_z}[mm.permille]")

        self.log("Emittance after tracking:")
        self.log(f"εx = {I.emitt_x}[mm.rad]")
        self.log(f"εy = {I.emitt_y}[mm.rad]")
        self.log(f"εz = {I.emitt_z}[mm.permille]")

    def __ensure_tracked(self):
        if getattr(self, "_needs_tracking", False):
            self.__track_bunch()
            self._needs_tracking = False

    def change_energy(self, grad=None, **kwargs):
        self.__setup_beam1()
        self.__track_bunch()

    def reset_energy(self, grad=1,**kwargs):
        self.__setup_beam0( )
        self.__track_bunch()

    def change_intensity(self, grad=None, **kwargs): #reduced charge
        self.__setup_beam2()
        self.__track_bunch()

    def reset_intensity(self, grad=1,**kwargs):
        self.__setup_beam0()
        self.__track_bunch()

    def get_sequence(self):
        return self.sequence

    def get_bpms_names(self):
        return self.bpms

    def get_screens_names(self):
        return self.screens

    def get_correctors_names(self):
        return self.corrs

    def get_hcorrectors_names(self):
        return [string for string in self.corrs if (string.lower().startswith('zh')) or (string.lower().startswith('zx'))]

    def get_vcorrectors_names(self):
        return [string for string in self.corrs if string.lower().startswith('zv')]

    def get_elements_position(self,names):
        return [index for index, string in enumerate(self.sequence) if string in names]

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
        #print("Reading ict's...")
        self.log("Reading ict's...")
        charge = [ bpm.get_total_charge() for bpm in self.lattice.get_bpms() ]
        icts = {
            "names": self.bpms,
            "charge": charge
        }        
        return icts

    def get_correctors(self):
        #print("Reading correctors' strengths...")
        self.log("Reading correctors' strengths...")
        bdes = np.zeros(len(self.corrs))
        for i,corrector in enumerate(self.corrs):
            if corrector[:2] == "ZH" or corrector[:2] == "ZX":
                bdes[i] = (self.lattice[corrector].get_strength()[0]*10)  # gauss*m
            elif corrector[:2] == "ZV":
                bdes[i] = (self.lattice[corrector].get_strength()[1]*10)  # gauss*m
        correctors = { "names": self.corrs, "bdes": bdes, "bact": bdes }
        return correctors
    
    def get_bpms(self):
        #print('Reading bpms...')
        self.log('Reading bpms...')
        self.__ensure_tracked()
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
    
    def get_bpms_S(self):
        S = []
        for name in self.bpms:
            S.append(self.lattice[name].get_S())
        return np.array(S, dtype=float)

    def get_element_S(self, name: str) -> float:
        return float(self.lattice[name].get_S())


    def get_screens(self):
        #print('Reading screens...')
        self.log('Reading screens...')
        nscreens = len(self.screens)
        hpixel = np.ones(nscreens) * 0.1 # mm, horizonatl size of a pixel
        vpixel = np.ones(nscreens) * 0.1 # mm, vertical size of a pixel
        images = []
        hedges_all = []
        vedges_all = []
        for i,s in enumerate(self.lattice.get_screens()):
            m = s.get_bunch().get_phase_space('%x %y')
            nx = np.ptp(m[:,0]) / hpixel[i]
            ny = np.ptp(m[:,1]) / vpixel[i]
            image, hedges, vedges = np.histogram2d(m[:,0], m[:,1], bins=(nx,ny))
            images.append(image)
            hedges_all.append(hedges)
            vedges_all.append(vedges)
        screens = { "names": self.screens,
                    "hpixel": hpixel,
                    "vpixel": vpixel,
                    "hedges" : hedges_all,
                    "vedges" : vedges_all,
                    "images": images }
        return screens

    def push(self, names, corr_vals):
        if not isinstance(names, list):
            names = [names]
        if np.isscalar(corr_vals):
            corr_vals = [corr_vals] * len(names)
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].set_strength(val* self.kl_per_A[corr]*1000, 0.0)  # A to T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].set_strength(0.0, val* self.kl_per_A[corr]*1000)  # A to T*mm
        self.__track_bunch()
    
    def vary_correctors(self, names, corr_vals):
        if not isinstance(names, list):
            names = [names]
        if np.isscalar(corr_vals):
            corr_vals = [corr_vals] * len(names)
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].vary_strength(val* self.kl_per_A[corr]*1000, 0.0)  # A to T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].vary_strength(0.0, val* self.kl_per_A[corr]*1000)  # A to T*mm
        # self.__track_bunch()
        self._needs_tracking = True

    def align_everything(self):
        self.lattice.align_elements()
        self.__track_bunch()

    def misalign_quadrupoles(self,sigma_x=0.100,sigma_y=0.100):
        self.lattice.scatter_elements('quadrupole', sigma_x, sigma_y, 0, 0, 0, 0, 'center')
        self.__track_bunch()

    def misalign_bpms(self,sigma_x=0.100,sigma_y=0.100):
        self.lattice.scatter_elements('bpm', sigma_x, sigma_y, 0, 0, 0, 0, 'center')
        self.__track_bunch()

    def measure_dispersion(self):
        print("Measuring dispersion (RF-Track)...")

        # Nominal energy
        self.__setup_beam0()
        self.__track_bunch()
        bpms0 = self.get_bpms()
        x0 = np.mean(bpms0["x"], axis=0)
        y0 = np.mean(bpms0["y"], axis=0)

        # Reduced energy
        self.__setup_beam1()
        self.__track_bunch()
        bpms1 = self.get_bpms()
        x1 = np.mean(bpms1["x"], axis=0)
        y1 = np.mean(bpms1["y"], axis=0)

        # Restore nominal
        self.__setup_beam0()
        self.__track_bunch()

        delta = -0.02  # Pref -> 0.98 * Pref
        eta_x = (x1 - x0) / delta
        eta_y = (y1 - y0) / delta

        return {"eta_x": eta_x, "eta_y": eta_y}

    def apply_qmag_current(self, name, dA):
        dk1l = self.kl_per_A[name]*dA
        print(f"Applying {name} current: dA = {dA}")
        elems = self.lattice[name]
        # 複数要素を同じだけ変える
        
        for elem in elems:
            k1l = elem.get_K1L(self.Pref)
            elem.set_K1L(self.Pref, k1l + dk1l/len(elems))
        # self.__track_bunch()
        self._needs_tracking = True

    def apply_qmag_offsets(self, name, dx, dy, dr, add = True):
        elems = self.lattice[name] + self.lattice[name + "MULT"] 
        bpms = self.lattice["M" + name]
        if not isinstance(bpms, (list,tuple)):
            bpms = [bpms]

        for elem in elems:
            if add == True:
                x = elem.get_offsets()[0][0] #mm
                y = elem.get_offsets()[0][1] #mm
                z = elem.get_offsets()[0][2] #mm
                r = elem.get_offsets()[0][3] #rad
                elem.set_offsets(x*1e-3 + dx *1e-6, y*1e-3 + dy*1e-6, z*1e-3, r + dr*1e-6, 0 , 0)
            else:
                elem.set_offsets(dx *1e-6, dy*1e-6, 0, dr*1e-6, 0 , 0)

        for bpm in bpms:
            if add == True:
                x = bpm.get_offsets()[0][0] #mm
                y = bpm.get_offsets()[0][1] #mm
                z = bpm.get_offsets()[0][2] #mm
                bpm.set_offsets(x*1e-3 + dx *1e-6, y*1e-3 + dy*1e-6, z*1e-3)
            else:
                bpm.set_offsets(dx *1e-6, dy*1e-6, 0)


    def apply_sum_knob(self, I):
        """
        SUM knob: QS1X +k, QS2X +k
        """
        print(f"Applying SUM knob: k = {I}")
        self.apply_qmag_current("QS1X",I)
        self.apply_qmag_current("QS2X",I)

        # self.__track_bunch()
        self._needs_tracking = True

    def apply_random_misalignment(
        self,
        seed: int,
        sigma_dx_um: float,
        sigma_dy_um: float,
        sigma_dtheta_urad: float,
        sigma_dk_rel: float,):

        print(
            f"Applying random misalignment (custom): seed={seed}, "
            f"sigma_dx={sigma_dx_um}um, sigma_dy={sigma_dy_um}um, "
            f"sigma_dtheta={sigma_dtheta_urad}urad, sigma_dk_rel={sigma_dk_rel}"
        )

        rng = np.random.default_rng(seed)

        Qnames = self.Qmagnames  

        for name in Qnames:
            dx = rng.normal(0.0, sigma_dx_um)
            dy = rng.normal(0.0, sigma_dy_um)
            dtheta = rng.normal(0.0, sigma_dtheta_urad)
            dk_rel = rng.normal(0.0, sigma_dk_rel)

            print(f"{name}: dx={dx:.1f}um  dy={dy:.1f}um  dθ={dtheta:.2f}urad  dk_rel={dk_rel:.3e}")

            self.apply_qmag_offsets(name, dx, dy, dtheta, add= False)
            elems = self.lattice[name]
            for elem in elems:
                k1l = elem.get_K1L(self.Pref)
                elem.set_K1L(self.Pref, k1l * (1 + dk_rel))

        self._needs_tracking = True

    def reset_lattice(self):
        self._build_lattice()

    def get_ipbsm_state(self):
        self.__track_bunch()
        B1_IP = self.lattice['IP'].get_bunch()

        ps = B1_IP.get_phase_space('%x %xp %y %yp %dt %P')
        y_positions = ps[:, 2] * 1e-3  # mm→m

        degMode, ModIPBSM, SigIPBSM = ipbsm_calc.FuncIPBSMbeamsize(y_positions)

        return {
            "modulation": ModIPBSM,
            "angle_deg": degMode,
            "sigma_y_m": SigIPBSM,
        }
    
    # ----------------------------
    # Knobs (linear / nonlinear)
    # ----------------------------
    def get_linear_knob_names(self):
        return list(self.knobs.linear_matrix.keys())

    def get_nonlinear_knob_names(self):
        return list(self.knobs.nonlinear_matrix.keys())

    def set_linear_knob(self, knob_name: str, value: float):
        self.knobs.set_linear_knob(knob_name, float(value))
        self.knobs.apply()
        self._needs_tracking = True

    def set_nonlinear_knob(self, knob_name: str, value: float):
        self.knobs.set_nonlinear_knob(knob_name, float(value))
        self.knobs.apply()
        self._needs_tracking = True

    def reset_knobs(self):
        self.knobs.reset_knobs()
        self._needs_tracking = True