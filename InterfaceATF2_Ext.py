import time
import numpy as np

from epics import PV, ca


class InterfaceATF2_Ext:
    """
    実機 ATF2 Ext インターフェース。
    - 既存の SysID 用 API (get_bpms, get_correctors, push, vary_correctors, ...)
    - シミュレーションと同じシグネチャの新 API
      (measure_dispersion, apply_qf6x, apply_sum_knob, apply_random_misalignment, reset_lattice)
      を用意するが、実装は後で PV 名を埋めていく想定。
    """

    def get_name(self):
        return "ATF2_Ext"

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
        monitors_from_sequence = [s for s in sequence if not s.lower().startswith("z")]
        bpm_ok = all(bpm in monitors for bpm in monitors_from_sequence)
        if not bpm_ok:
            bpms_unknown = [bpm for bpm in monitors_from_sequence if bpm not in monitors]
            print(f"Unknown bpms {bpms_unknown} removed from list")
        # Only retain BPMs in config file which are known by Epics
        sequence_filtered = [
            elem for elem in sequence
            if (elem in monitors) or elem.lower().startswith("z")
        ]
        # Subset of BPMs and correctors from the config file
        self.sequence = sequence_filtered
        self.bpms = [s for s in self.sequence if not s.lower().startswith("z")]
        self.corrs = [s for s in self.sequence if s.lower().startswith("z")]
        # Index of the selected BPMs in the Epics PV ATF2:monitors
        self.bpm_indexes = [i for i, s in enumerate(monitors) if s in self.bpms]
        # Bunch current monitors
        self.ict_names = [
            "gun:GUNcharge", "l0:L0charge", "linacbt:LNEcharge", "linacbt:BTMcharge",
            "ext:EXTcharge", "linacbt:BTEcharge", "BIM:DR:nparticles", "BIM:IP:nparticles"
        ]

    # ---------------------------------------------------------------
    # 既存 API
    # ---------------------------------------------------------------
    def change_energy(self, *args):
        pass

    def reset_energy(self, *args):
        pass

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
        return [i for i, s in enumerate(self.sequence) if s in names]

    def get_bpms_S(self):
        """
        実機では S[m] をここで設計値から与える想定。
        いまは未実装のため、例外を投げる。
        """
        raise NotImplementedError(
            "get_bpms_S is not implemented yet for InterfaceATF2_Ext."
        )

    def get_element_S(self, name: str) -> float:
        """
        任意の要素名の S[m] を返す想定。
        いまは未実装のため、例外を投げる。
        """
        raise NotImplementedError(
            "get_element_S is not implemented yet for InterfaceATF2_Ext."
        )

    def get_icts(self):
        print("Reading ict's...")
        charge = []
        for ict in self.ict_names:
            pv = PV(f"{ict}")
            charge.append(pv.get())
        names = [self.ict_names] if isinstance(self.ict_names, str) else self.ict_names
        charge = np.array(charge)
        icts = {"names": names, "charge": charge}
        return icts

    def get_correctors(self):
        print("Reading correctors' strengths...")
        bdes, bact = [], []
        for corrector in self.corrs:
            pv_des = PV(f"{corrector}:currentWrite")
            pv_act = PV(f"{corrector}:currentRead")
            bdes.append(pv_des.get())
            bact.append(pv_act.get())
        names = [self.corrs] if isinstance(self.corrs, str) else self.corrs
        bdes = np.array(bdes)
        bact = np.array(bact)
        correctors = {"names": names, "bdes": bdes, "bact": bact}
        return correctors

    def get_bpms(self):
        print("Reading bpms...")
        p = PV("LINAC:monitors")
        x, y, tmit = [], [], []
        for sample in range(self.nsamples):
            print(f"Sample = {sample}")
            a = p.get().reshape((-1, 20))
            status = a[self.bpm_indexes, 0]
            status[status != 1] = 0
            x.append(a[self.bpm_indexes, 1])
            y.append(a[self.bpm_indexes, 2])
            tmit.append(status * a[self.bpm_indexes, 3])
            time.sleep(1)
        names = [self.bpms] if isinstance(self.bpms, str) else self.bpms
        x = np.vstack(x) / 1e3  # mm
        y = np.vstack(y) / 1e3  # mm
        tmit = np.vstack(tmit)
        bpms = {"names": names, "x": x, "y": y, "tmit": tmit}
        return bpms

    def push(self, names, corr_vals):
        if isinstance(corr_vals, float):
            corr_vals = np.array([corr_vals])
        if isinstance(names, str):
            names = [names]
        if len(names) != corr_vals.size:
            print("Error: len(names) != len(corr_vals) in push(names, corr_vals)")
        for corrector, corr_val in zip(names, corr_vals):
            pv_des = PV(f"{corrector}:currentWrite")
            pv_des.put(corr_val)
        time.sleep(1)

    def vary_correctors(self, names, corr_vals):
        if isinstance(corr_vals, float):
            corr_vals = np.array([corr_vals])
        if isinstance(names, str):
            names = [names]
        if len(names) != corr_vals.size:
            print(
                "Error: len(names) != len(corr_vals) in vary_correctors(names, corr_vals)"
            )
        for corrector, corr_val in zip(names, corr_vals):
            pv_des = PV(f"{corrector}:currentWrite")
            curr_val = pv_des.get()
            pv_des.put(curr_val + corr_val)
        time.sleep(1)

    # ---------------------------------------------------------------
    # 新 API: RF-Track と同じシグネチャを持つが、実装は今後追加
    # ---------------------------------------------------------------
    def measure_dispersion(self):
        """
        実機での dispersion 測定用プレースホルダ。
        GUI 側からは interface.measure_dispersion() として呼ばれる。
        将来的には RF 位相や DR RF などの PV を叩いてエネルギーを変えた測定に対応。
        """
        raise NotImplementedError(
            "measure_dispersion is not implemented yet for InterfaceATF2_Ext."
        )

    def apply_qf6x(self, dk1):
        """
        QF6X knob 用プレースホルダ。
        """
        raise NotImplementedError(
            "apply_qf6x is not implemented yet for InterfaceATF2_Ext."
        )

    def apply_sum_knob(self, k):
        """
        SUM knob (QS1X +k, QS2X -k) 用プレースホルダ。
        """
        raise NotImplementedError(
            "apply_sum_knob is not implemented yet for InterfaceATF2_Ext."
        )

    def get_ipbsm_state(self):
        """
        IPBSM 状態取得プレースホルダ。
        将来、modulation, angle, sigma_y などを PV から読み出して返す想定。
        戻り値のフォーマットは RF-Track 側と合わせて:
            {
                "modulation": float,   # dimensionless
                "angle_deg": float,    # deg
                "sigma_y_m": float,    # m
            }
        """
        raise NotImplementedError(
            "get_ipbsm_state is not implemented yet for InterfaceATF2_Ext."
        )

    def set_ipbsm_state(self, modulation: float, angle_deg: float):
        """
        IPBSM 状態設定プレースホルダ。
        将来、modulation, angle mode を PV に書き込む実装に差し替える。
        """
        raise NotImplementedError(
            "set_ipbsm_state is not implemented yet for InterfaceATF2_Ext."
        )


    def apply_random_misalignment(
        self,
        seed: int,
        sigma_dx_um: float,
        sigma_dy_um: float,
        sigma_dtheta_urad: float,
        sigma_dk_rel: float,
    ):
        """
        実機ではランダム misalignment は行わないので、プレースホルダ。
        """
        raise NotImplementedError(
            "apply_random_misalignment is not supported for real machine."
        )

    def reset_lattice(self):
        """
        実機では格子の reset は意味を持たないので、プレースホルダ。
        """
        raise NotImplementedError(
            "reset_lattice is not supported for real machine."
        )
