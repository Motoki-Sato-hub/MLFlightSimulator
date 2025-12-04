import os
import time
import numpy as np
import RF_Track as rft
import ipbsm_calc

from misalignment_generator import RandomMisalignment


class InterfaceATF2_Ext_RFTrack:


    def get_name(self):
        return "ATF2_Ext_RFT"

    def __init__(self, population=2e10, jitter=0.0, bpm_resolution=0.0, nsamples=1):
        self.population = population
        self.jitter = jitter
        self.bpm_resolution = bpm_resolution
        self.nsamples = nsamples

        # IPBSM 状態（modulation, angle, sigma_y）
        self.ipbsm_modulation = 0.5  # dimensionless
        self.ipbsm_angle_deg = 2.0   # deg (mode)
        # sigma_y は必要時に ipbsm_calc で計算

        # Twiss ファイルへの絶対パス
        here = os.path.dirname(__file__)
        self.twiss_path = os.path.join(here, "Ext_ATF2", "ATF2_EXT_FF_v5.2.twiss")

        # ビーム基準運動量 [MeV/c]
        self.Pref = 1.2999999e3  # 1.3 GeV/c

        # 格子とビームを構築
        self._build_lattice()

    # ------------------------------------------------------------------
    # 内部セットアップ
    # ------------------------------------------------------------------
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
        """必要なら __track_bunch() を 1 回だけ実行する。"""
        if getattr(self, "_needs_tracking", False):
            self.__track_bunch()
            self._needs_tracking = False

    # ------------------------------------------------------------------
    # 既存 API (SysID 用)
    # ------------------------------------------------------------------
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
        """各 BPM の S [m] を self.bpms の順に返す。"""
        S = []
        for name in self.bpms:
            S.append(self.lattice[name].get_S())
        return np.array(S, dtype=float)

    def get_element_S(self, name: str) -> float:
        """任意の要素名の S [m] を返す。"""
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
            if corrector[:2] == "ZH":
                bdes[i] = self.lattice[corrector].get_strength()[0] * 10.0  # gauss*m
            elif corrector[:2] == "ZV":
                bdes[i] = self.lattice[corrector].get_strength()[1] * 10.0  # gauss*m
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
                self.lattice[corr].set_strength(val / 10.0, 0.0)  # T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].set_strength(0.0, val / 10.0)  # T*mm
        self.__track_bunch()

    def vary_correctors(self, names, corr_vals):
        """相対的に加える。"""
        if not isinstance(names, list):
            names = [names]
        if np.isscalar(corr_vals):
            corr_vals = [corr_vals] * len(names)
        for corr, val in zip(names, corr_vals):
            if corr[:2] == "ZH":
                self.lattice[corr].vary_strength(val / 10.0, 0.0)  # T*mm
            elif corr[:2] == "ZV":
                self.lattice[corr].vary_strength(0.0, val / 10.0)  # T*mm
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

    def apply_qf6x(self, dk1):
        """
        QF6X の k1 に dk1 を加える knob。
        - QF6X は 2 要素 (lattice["QF6X"][0], [1]) として存在すると仮定。
        GUI からは interface.apply_qf6x(dk1) として呼ばれる。
        """
        print(f"Applying QF6X knob: dk1 = {dk1}")
        elems = self.lattice["QF6X"]
        # 複数要素を同じだけ変える
        for elem in elems:
            k = elem.get_strength()
            elem.set_strength(k+dk1)
        # self.__track_bunch()
        self._needs_tracking = True

    def apply_sum_knob(self, k):
        """
        SUM knob: QS1X +k, QS2X +k
        - それぞれ 2 要素 (lattice["QS1X"][0/1], ["QS2X"][0/1]) を想定。
        GUI からは interface.apply_sum_knob(k) として呼ばれる。
        """
        print(f"Applying SUM knob: k = {k}")
        qs1_elems = self.lattice["QS1X"]
        qs2_elems = self.lattice["QS2X"]

        # QS1X: +k
        for e in qs1_elems:
            k2 = e.get_strength()
            e.set_strength(k2 + k)

        # QS2X: +k
        for e in qs2_elems:
            k2 = e.get_strength()
            e.set_strength(k2 + k)

        # self.__track_bunch()
        self._needs_tracking = True

    def apply_random_misalignment(
        self,
        seed: int,
        sigma_dx_um: float,
        sigma_dy_um: float,
        sigma_dtheta_urad: float,
        sigma_dk_rel: float,
    ):
        """
        RF-Track の RandomMisalignment を使ってランダムなミスアライメントを付与。
        GUI からは interface.apply_random_misalignment(...) として呼ばれる。
        """
        print(
            f"Applying random misalignment: seed={seed}, "
            f"sigma_dx={sigma_dx_um} um, sigma_dy={sigma_dy_um} um, "
            f"sigma_dtheta={sigma_dtheta_urad} urad, sigma_dk={sigma_dk_rel}"
        )

        # misalignment_generator 側のデフォルト値に対するスケーリング
        BASE_DX = 100.0   # μm
        BASE_DY = 100.0   # μm
        BASE_DTH = 200.0  # μrad
        BASE_DK = 0.001   # relative

        scale_dx = sigma_dx_um / BASE_DX if BASE_DX > 0 else 1.0
        scale_dy = sigma_dy_um / BASE_DY if BASE_DY > 0 else 1.0
        scale_dth = sigma_dtheta_urad / BASE_DTH if BASE_DTH > 0 else 1.0
        scale_dk = sigma_dk_rel / BASE_DK if BASE_DK > 0 else 1.0

        M = RandomMisalignment(self.lattice, seed=seed)
        # 基本的なランダム誤差を生成
        M.apply_random_errors()

        # 生成された誤差をスケーリング
        for key, params in M.errors.items():
            params["dx"] *= scale_dx
            params["dy"] *= scale_dy
            params["dtheta"] *= scale_dth
            params["dk"] *= scale_dk
            M.errors[key] = params

        # 一旦 JSON に保存してから再適用（RandomMisalignment の API に合わせる）
        tmpfile = os.path.join(os.getcwd(), "tmp_misalign.json")
        M.save_errors(tmpfile)
        M.load_errors_and_apply(tmpfile)

        # 再トラッキング
        # self.__track_bunch()
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

