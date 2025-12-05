import os
import time
import numpy as np
import RF_Track as rft
import ipbsm_calc
from knobs import KnobSystem



class InterfaceATF2_Ext_RFTrack:

    def get_name(self):
        return "ATF2_Ext_RFT"

    def __init__(self, population=2e10, jitter=0.0, bpm_resolution=0.0, nsamples=1):
        self.population = population
        self.jitter = jitter
        self.bpm_resolution = bpm_resolution
        self.nsamples = nsamples

        here = os.path.dirname(__file__)
        self.twiss_path = os.path.join(here, "Ext_ATF2", "ATF2_EXT_FF_v5.2.twiss")

        self.Pref = 1.2999999e3  # 1.3 GeV/c

        self._build_lattice()

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
        """Twiss から格子・BPM・コレクタ情報を構築し、ビームをセットアップしてトラッキング。"""
        self.lattice = rft.Lattice(self.twiss_path)
        Scr = rft.Screen()
        self.lattice['IP'].replace_with(Scr)
        self.lattice.set_bpm_resolution(self.bpm_resolution)

        self.sequence = [e.get_name() for e in self.lattice["*"]]
        self.bpms = [e.get_name() for e in self.lattice.get_bpms()]
        self.corrs = [e.get_name() for e in self.lattice.get_correctors()]

        self.__setup_beam0()
        self.__track_bunch()

    def __setup_beam0(self):
        """Nominal エネルギー・電荷でのビーム。"""
        T = rft.Bunch6d_twiss()
        T.emitt_x = 5.2  # mm.mrad normalised emittance
        T.emitt_y = 0.03  # mm.mrad
        T.beta_x = 6.848560987  # m
        T.beta_y = 2.935758992  # m
        T.alpha_x = 1.108024744
        T.alpha_y = -1.907222942
        T.sigma_t = 8  # mm/c
        T.sigma_pt = 0.8  # permille
        Nparticles = 10000  # number of macroparticles
        self.B0 = rft.Bunch6d_QR(
            rft.electronmass, self.population, -1, self.Pref, T, Nparticles
        )

    def __setup_beam1(self):
        """Reduced energy (DFS 用)。Pref を 0.98 倍。"""
        Pref = 0.98 * self.Pref  # 98% of nominal energy
        T = rft.Bunch6d_twiss()
        T.emitt_x = 2e-3  # mm.mrad normalised emittance
        T.emitt_y = 1.179228346e-5  # mm.mrad
        T.beta_x = 6.848560987  # m
        T.beta_y = 2.935758992  # m
        T.alpha_x = 1.108024744
        T.alpha_y = -1.907222942
        T.sigma_t = 8  # mm/c
        T.sigma_pt = 0.8  # permille
        Nparticles = 10000  # number of macroparticles
        self.B0 = rft.Bunch6d_QR(
            rft.electronmass, self.population, -1, Pref, T, Nparticles
        )

    def __setup_beam2(self):
        """Reduced bunch charge (WFS 用)。"""
        population = 0.90 * self.population  # 90% of nominal charge
        T = rft.Bunch6d_twiss()
        T.emitt_x = 2e-3  # mm.mrad normalised emittance
        T.emitt_y = 1.179228346e-5  # mm.mrad
        T.beta_x = 6.848560987  # m
        T.beta_y = 2.935758992  # m
        T.alpha_x = 1.108024744
        T.alpha_y = -1.907222942
        T.sigma_t = 8  # mm/c
        T.sigma_pt = 0.8  # permille
        Nparticles = 10000  # number of macroparticles
        self.B0 = rft.Bunch6d_QR(
            rft.electronmass, population, -1, self.Pref, T, Nparticles
        )

    def __track_bunch(self):
        """ジッター付きビームで格子を 1 pass トラッキング。"""
        I0 = self.B0.get_info()
        dx = self.jitter * I0.sigma_x
        dy = self.jitter * I0.sigma_y
        dz, roll = 0.0, 0.0
        pitch = self.jitter * I0.sigma_py
        yaw = self.jitter * I0.sigma_px
        B0_offset = self.B0.displaced(dx, dy, dz, roll, pitch, yaw)
        self.lattice.track(B0_offset)

    def __ensure_tracked(self):
        if getattr(self, "_needs_tracking", False):
            self.__track_bunch()
            self._needs_tracking = False


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
        return [s for s in self.corrs if s.lower().startswith("zh")]

    def get_vcorrectors_names(self):
        return [s for s in self.corrs if s.lower().startswith("zv")]

    def get_elements_position(self, names):
        return [index for index, string in enumerate(self.sequence) if string in names]

    def get_bpms_S(self):
        S = []
        for name in self.bpms:
            S.append(self.lattice[name].get_S())
        return np.array(S, dtype=float)

    def get_element_S(self, name: str) -> float:
        return float(self.lattice[name].get_S())

    def get_icts(self):
        print("Reading ict's...")
        charge = [bpm.get_total_charge() for bpm in self.lattice.get_bpms()]
        icts = {"names": self.bpms, "charge": charge}
        return icts

    def get_correctors(self):
        print("Reading correctors' strengths...")
        bdes = np.zeros(len(self.corrs))
        for i, corrector in enumerate(self.corrs):
            # RF-Track 内部は Tmm で保持している
            if corrector[:2] == "ZH":
                tmm = self.lattice[corrector].get_strength()[0]   # Tmm
            elif corrector[:2] == "ZV":
                tmm = self.lattice[corrector].get_strength()[1]   # Tmm
            else:
                tmm = 0.0

            if corrector in self.kl_per_A:
                if self.kl_per_A[corrector] == 0.0:
                    bdes[i] = 0.0
                else:
                    bdes[i] = tmm / self.kl_per_A[corrector]   # I = Tmm / (Tmm/A)
            else:
                bdes[i] = 0.0    # 係数未設定なら 0A として返す

        correctors = {"names": self.corrs, "bdes": bdes, "bact": bdes}
        return correctors

    def get_bpms(self):
        print("Reading bpms...")
        self.__ensure_tracked()
        x = np.zeros((self.nsamples, len(self.bpms)))
        y = np.zeros_like(x)
        tmit = np.zeros_like(x)
        for i in range(self.nsamples):
            for j, bpm in enumerate(self.bpms):
                b = self.lattice[bpm]
                reading = b.get_reading()
                x[i, j] = reading[0]
                y[i, j] = reading[1]
                tmit[i, j] = b.get_total_charge()
        bpms = {"names": self.bpms, "x": x, "y": y, "tmit": tmit}
        return bpms

    def push(self, names, corr_vals):
        """絶対値 set."""
        if not isinstance(names, list):
            names = [names]
        if np.isscalar(corr_vals):
            corr_vals = [corr_vals] * len(names)
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].set_strength(val* self.kl_per_A[corr], 0.0)  # A to T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].set_strength(0.0, val* self.kl_per_A[corr])  # A to T*mm
        self.__track_bunch()

    def vary_correctors(self, names, corr_vals):
        if not isinstance(names, list):
            names = [names]
        if np.isscalar(corr_vals):
            corr_vals = [corr_vals] * len(names)
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].vary_strength(val* self.kl_per_A[corr], 0.0)  # A to T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].vary_strength(0.0, val* self.kl_per_A[corr])  # A to T*mm
        # self.__track_bunch()
        self._needs_tracking = True

    # ------------------------------------------------------------------
    # 新 API: Dispersion / knobs / misalignment / reset
    # ------------------------------------------------------------------
    def measure_dispersion(self):
        """
        RF-Track 上で dispersion を測定。
        - Nominal energy
        - Reduced energy (Pref = 0.98 * Pref)
        の 2 つのビームを流して、delta = -0.02 と仮定して ηx, ηy を返す。
        戻り値: {"eta_x": np.ndarray, "eta_y": np.ndarray}
        """
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
        elems = self.lattice[name] + self.lattice[name + "MULT"] + self.lattice["M" + name]

        for elem in elems:
            if add == True:
                x = elem.get_offsets()[0][0] #m
                y = elem.get_offsets()[0][1] #m
                z = elem.get_offsets()[0][2] #m
                r = elem.get_offsets()[0][3] #rad
                elem.set_offsets(x + dx *1e-6, y + dy*1e-6, z, r + dr*1e-6, 0 , 0)
            else:
                elem.set_offsets(dx *1e-6, dy*1e-6, 0, dr*1e-6, 0 , 0)


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
        """
        ミスアライメントを含めて格子を Twiss から再ロードしてリセット。
        GUI の RESET ボタンから呼ばれる。
        """
        print("Resetting lattice from Twiss file (RF-Track)...")
        self._build_lattice()

    def get_ipbsm_state(self):
        """
        IPBSM 状態を取得:
        - modulation M
        - angle mode [deg]
        - sigma_y [m] （ipbsm_calc.sigmay_from_modulation により計算）
        """
        # トラッキング
        self.__track_bunch()
        # IPの bunch 取得
        B1_IP = self.lattice['IP'].get_bunch()

        # フェーズスペース取り出し
        ps = B1_IP.get_phase_space('%x %xp %y %yp %dt %P')
        y_positions = ps[:, 2] * 1e-3  # mm→m

        # IPBSM計算
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

