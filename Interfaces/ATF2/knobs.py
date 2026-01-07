# knobs.py
# Author: Motoki Sato
# Date: 2025-09-10

import numpy as np
import RF_Track as rft

class KnobSystem:
    def __init__(self, lattice, p_ref=-1300):
        self.lat = lattice
        self.p_ref = p_ref 

    
        self.base_magnets = [
            "SD0FF", "SF1FF", "SD4FF", "SF5FF", "SF6FF",
            "SK1FF", "SK2FF", "SK3FF", "SK4FF"
        ]

        self.segments = {name: {"sext": [], "multi": []} for name in self.base_magnets}
        for name in self.base_magnets:
            # sext
            sext_len = self._safe_len(name)
            for i in range(sext_len):
                self.segments[name]["sext"].append((name, i))
            # multi
            mname = name + "MULT"
            multi_len = self._safe_len(mname)
            for i in range(multi_len):
                self.segments[name]["multi"].append((mname, i))

        self.offsets0 = {}
        self.knl0 = {}
        self.length = {}
        for name in self.base_magnets:
            for group in ("sext", "multi"):
                for lat_name, idx in self.segments[name][group]:
                    elem = self.lat[lat_name][idx]
                    off = np.array(elem.get_offsets()[0], dtype=float)  # [x,y,z,1,...]
                    self.offsets0[(lat_name, idx)] = (off[0], off[1], off[2] if len(off) > 2 else 0.0)
                    self.knl0[(lat_name, idx)] = np.array(elem.get_KnL(self.p_ref)).flatten()
                    try:
                        self.length[(lat_name, idx)] = float(elem.get_length())
                    except Exception:
                        self.length[(lat_name, idx)] = 0.0

        
        self.linear_knobs = {}    
        self.nonlinear_knobs = {}  

        # (dx,dy) in [μm], from LinearKnob_20240617.dat
        self.linear_matrix = {
            "A_x": {"SD0FF": (-116.5, 0), "SF1FF": (-35.2, 0), "SD4FF": (-37.8, 0), "SF5FF": (0, 0), "SF6FF": (-623.4, 0)},  
            "E_x": {"SD0FF": (-813.0, 0), "SF1FF": (793.5, 0), "SD4FF": (-137.9, 0), "SF5FF": (0, 0), "SF6FF": (1492.9, 0)},
            "A_y": {"SD0FF": (-98.9, 0), "SF1FF": (-9.6, 0), "SD4FF": (-246.7, 0), "SF5FF": (0, 0), "SF6FF": (120.0, 0)},
            "E_y": {"SD0FF": (0, 374.1), "SF1FF": (0, -120.0), "SD4FF": (0, -451.3), "SF5FF": (0, 0), "SF6FF": (0, -64.3)},
            "Coup1": {"SD0FF": (0, -100), "SF1FF": (0, 100), "SD4FF": (0, 0), "SF5FF": (0, 0), "SF6FF": (0, 0)},
            "Coup2": {"SD0FF": (0, 107.8), "SF1FF": (0, 4.2), "SD4FF": (0, 152.6), "SF5FF": (0, 0), "SF6FF": (0, 90.4)},
            "Spare1": {"SD0FF": (-676.0, 0), "SF1FF": (-316.0, 0), "SD4FF": (252.0, 0), "SF5FF": (587.0, 0), "SF6FF": (593.0, 0)},
            "Spare2": {"SD0FF": (0, 78.0), "SF1FF": (0, 188.0), "SD4FF": (0, -55.0), "SF5FF": (-179.0, 0), "SF6FF": (0, 38.0)},
            "Spare3": {"SD0FF": (0, 0), "SF1FF": (0, 0), "SD4FF": (0, 1), "SF5FF": (0, 0), "SF6FF": (0, 0)}
        }

        # coeff × knob_value = ΔK2L, from multiknob_itit_param_250527.dat
        self.nonlinear_matrix = {
            "Y24": {"SK1FF": 0.0, "SK2FF": 0.0, "SK3FF": 0.0, "SK4FF": 0.0, "SD0FF": 0.119, "SF1FF": -0.013, "SD4FF": -0.554, "SF5FF": -0.083, "SF6FF":  -0.175},
            "Y46": {"SK1FF": 0.0, "SK2FF": 0.0, "SK3FF": 0.0, "SK4FF": 0.0, "SD0FF": 0.259, "SF1FF": -0.057, "SD4FF": 1.049, "SF5FF": -0.106,"SF6FF":  -0.056},
            "Y22": {"SK1FF": -1.629, "SK2FF": 0.174, "SK3FF": 1.024, "SK4FF": 2.435, "SD0FF": 0.0, "SF1FF": 0.0, "SD4FF": 0.0, "SF5FF": 0.0, "SF6FF":  0.0},
            "Y26":{"SK1FF": 1.763, "SK2FF": -0.126, "SK3FF": 0.463, "SK4FF": -0.701, "SD0FF": 0.0, "SF1FF": 0.0, "SD4FF": 0.0, "SF5FF": 0.0, "SF6FF":  0.0},
            "Y66": {"SK1FF": 5.571, "SK2FF": -0.207, "SK3FF": -4.668, "SK4FF": -6.673, "SD0FF": 0.0, "SF1FF": 0.0, "SD4FF": 0.0, "SF5FF": 0.0, "SF6FF":  0.0},
            "Y44": {"SK1FF": 0.037, "SK2FF": 1.614, "SK3FF": -0.458, "SK4FF": -0.186, "SD0FF": 0.0, "SF1FF": 0.0, "SD4FF": 0.0, "SF5FF": 0.0, "SF6FF":  0.0},
            "Spare": {"SK1FF": 0.0, "SK2FF": 0.0, "SK3FF": 0.0, "SK4FF": 0.0, "SD0FF": 0.0, "SF1FF": 0.0, "SD4FF": 0.0, "SF5FF": 0.0, "SF6FF": 0.0}
        }

        self.kIconverter = {
            "SK1FF": 0.0,
            "SK2FF": 0.0,
            "SK3FF": 0.0,
            "SK4FF": 0.0,
            "SD0FF": -0.6803372,
            "SF1FF": 0.6865404,
            "SD4FF": -0.7802854,
            "SF5FF": 0.5505292,
            "SF6FF": 0.8329751 
        }

    def set_linear_knob(self, knob_name, value: float):
            self.linear_knobs[knob_name] = float(value)

        
    def set_nonlinear_knob(self, knob_name, value: float):
        self.nonlinear_knobs[knob_name] = float(value)

    def apply(self):
        for base in self.base_magnets:
            dx_total, dy_total = 0.0, 0.0
            for knob, mapping in self.linear_matrix.items():
                if knob in self.linear_knobs and base in mapping:
                    dx_c, dy_c = mapping[base]
                    v = self.linear_knobs.get(knob, 0.0)
                    dx_total += dx_c * v * 1e-6
                    dy_total += dy_c * v * 1e-6

            for group in ("sext", "multi"):
                for lat_name, idx in self.segments[base][group]:
                    x0, y0, z0 = self.offsets0[(lat_name, idx)]
                    elem = self.lat[lat_name][idx]
                    elem.set_offsets(x0 + dx_total, y0 + dy_total, z0)
        dK_by_mag_order_comp = {}  # {(base, order, comp): dK_total}
        for knob, mapping in self.nonlinear_matrix.items():
            if knob not in self.nonlinear_knobs:
                continue
            v = self.nonlinear_knobs[knob]
            for base, val in mapping.items():
                # normalize mapping values
                if isinstance(val, (int,float)):
                    coeff_dK, order, comp = float(val), 2, 'normal'
                elif isinstance(val, (list,tuple)) and len(val)>=1:
                    coeff_dK = float(val[0]); order = int(val[1]) if len(val)>1 else 2; comp = str(val[2]) if len(val)>2 else 'normal'
                elif isinstance(val, dict):
                    coeff_dK = float(val.get('coeff', 0.0)); order = int(val.get('order', 2)); comp = str(val.get('comp','normal'))
                else:
                    raise TypeError(f'Unsupported nonlinear mapping value for {base}: {type(val)}')
                key = (base, order, comp)
                dK_by_mag_order_comp[key] = dK_by_mag_order_comp.get(key, 0.0) + coeff_dK * v

        for (base, order, comp), dK in dK_by_mag_order_comp.items():
            for lat_name, idx in self.segments[base]["sext"]:
                elem = self.lat[lat_name][idx]
                L = self.length[(lat_name, idx)]
                dKnL = dK * L
                KnL = np.array(self.knl0[(lat_name, idx)], dtype=complex)

                if comp == "skew":
                    # skew
                    KnL[order] = complex(KnL[order].real, KnL[order].imag + dKnL)
                else:
                    # normal
                    KnL[order] = complex(KnL[order].real + dKnL, KnL[order].imag)

                elem.set_KnL(self.p_ref, KnL)

    def reset_knobs(self):
        self.linear_knobs = {k: 0.0 for k in self.linear_matrix.keys()}
        self.nonlinear_knobs = {k: 0.0 for k in self.nonlinear_matrix.keys()}
        self.apply()

    def recapture_baseline(self):
        for base in self.base_magnets:
            for group in ("sext", "multi"):
                for lat_name, idx in self.segments[base][group]:
                    elem = self.lat[lat_name][idx]
                    off = np.array(elem.get_offsets()[0], dtype=float)
                    self.offsets0[(lat_name, idx)] = (off[0], off[1], off[2] if len(off) > 2 else 0.0)
                    self.knl0[(lat_name, idx)] = np.array(elem.get_KnL(self.p_ref)).flatten()
                    try:
                        self.length[(lat_name, idx)] = float(elem.get_length())
                    except Exception:
                        self.length[(lat_name, idx)] = 0.0

    def show_offsets(self, base):
        # SEXT
        if self.segments[base]["sext"]:
            for (lat_name, idx) in self.segments[base]["sext"]:
                print(f"{lat_name}[{idx}].offsets =", self.lat[lat_name][idx].get_offsets())
        # MULTI
        if self.segments[base]["multi"]:
            for (lat_name, idx) in self.segments[base]["multi"]:
                print(f"{lat_name}[{idx}].offsets =", self.lat[lat_name][idx].get_offsets())

    def show_KnL(self, base):
        if self.segments[base]["sext"]:
            for (lat_name, idx) in self.segments[base]["sext"]:
                print(f"{lat_name}[{idx}].KnL =", self.lat[lat_name][idx].get_KnL(self.p_ref))
        if self.segments[base]["multi"]:
            for (lat_name, idx) in self.segments[base]["multi"]:
                print(f"{lat_name}[{idx}].KnL =", self.lat[lat_name][idx].get_KnL(self.p_ref))


    def _safe_len(self, name):
        try:
            return len(self.lat[name])
        except Exception:
            return 0