# -*- coding: utf-8 -*-
"""
Opt_IPBSM.py
Core logic for Gaussian-estimation / BO / LBO optimization with:
- Synthetic (test) controller
- EPICS (machine) controller stub
- Logging, stopping, and plotting hooks

Dependencies: numpy, matplotlib
(Only standard library + numpy + matplotlib required.)
"""

from __future__ import annotations

import os
import json
import time
import math
import csv
import datetime as _dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import time, threading
import numpy as np
from epics import PV
from epics import caget, caput

from collections import defaultdict

# ----------------------------
# Utilities
# ----------------------------
import struct

def decode_ipbsm_dat(raw: bytes):
    offset = 0

    save_amplitude, = struct.unpack_from("<d", raw, offset); offset += 8
    save_eamplitude, = struct.unpack_from("<d", raw, offset); offset += 8
    save_beamsize, = struct.unpack_from("<d", raw, offset); offset += 8
    save_ebeamsize, = struct.unpack_from("<d", raw, offset); offset += 8
    save_average, = struct.unpack_from("<d", raw, offset); offset += 8
    save_phase, = struct.unpack_from("<d", raw, offset); offset += 8

    save_filename = (
        raw[offset:offset+256]
        .split(b"\x00", 1)[0]
        .decode(errors="ignore")
    )
    offset += 256

    save_ict_average, = struct.unpack_from("<d", raw, offset)

    return {
        "modulation": save_amplitude,
        "error": save_eamplitude,
        "beamsize": save_beamsize,
        "ebeamsize": save_ebeamsize,
        "average": save_average,
        "phase": save_phase,
        "filename": save_filename,
        "ict_average": save_ict_average,
    }


def now_tag() -> str:
    # Asia/Tokyo not enforced here; GUI passes local time anyway.
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def normal_pdf(z: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

def normal_cdf(z: np.ndarray) -> np.ndarray:
    # Using erf; no scipy dependency
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))

# ----------------------------
# Controllers (interfaces)
# ----------------------------

class BaseIPBSMController:
    """
    Abstract controller. Real machine version should talk to EPICS PVs.
    """

    def get_ipbsm(self) -> Tuple[float, float]:
        """Return (modulation, modulation_error)."""
        raise NotImplementedError

    def set_magnet_current(self, name: str, values: float) -> None:
        raise NotImplementedError

    def set_magnet_position(self, name: str, values: float) -> None:
        raise NotImplementedError

    def apply_knobs(self, knob_valuess: Dict[str, float]) -> None:
        """
        Default: map knob_valuess to currents with set_magnet_current.
        Override if your machine mapping differs.
        """
        for k, v in knob_valuess.items():
            self.set_magnet_current(k, v)


class IPBSMInterface:
    def __init__(self):
        self.error_history = []

        self.pv_trigger = PV("TEST:ai")
        self.pv_end = PV("TEST:ENDai")

        self._ipbsm_lock = threading.Lock()

        self.datafile = "/atf/data/ipbsm/knob/knob_fringe_result_v2.dat"
                # (dx,dy) in [μm], from LinearKnob_20240617.dat
        self.linear_matrix = {
            "Ax": {"SD0FF": (-116.5, 0), "SF1FF": (-35.2, 0), "SD4FF": (-37.8, 0), "SF5FF": (0, 0), "SF6FF": (-623.4, 0)},  
            "Ex": {"SD0FF": (-813.0, 0), "SF1FF": (793.5, 0), "SD4FF": (-137.9, 0), "SF5FF": (0, 0), "SF6FF": (1492.9, 0)},
            "Ay": {"SD0FF": (-98.9, 0), "SF1FF": (-9.6, 0), "SD4FF": (-246.7, 0), "SF5FF": (0, 0), "SF6FF": (120.0, 0)},
            "Ey": {"SD0FF": (0, 374.1), "SF1FF": (0, -120.0), "SD4FF": (0, -451.3), "SF5FF": (0, 0), "SF6FF": (0, -64.3)},
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
        

    def get_ipbsm(self, timeout=300, file_wait=330.0, poll=0.1, trig_pulse=0.05):
        with self._ipbsm_lock:
            # A) baseline mtime BEFORE trigger (これが重要)
            try:
                mtime0 = os.path.getmtime(self.datafile)
            except FileNotFoundError:
                mtime0 = 0.0

            # B) ENDai が 1 のまま張り付いてたら 0 に落としておく（即break防止）

            # 1) trigger (パルス推奨)
            self.pv_trigger.put(1)

            # 2) wait ENDai == 1
            t_deadline = time.time() + float(timeout)
            while True:
                v = self.pv_end.get()
                if v is not None and int(v) == 1:
                    break
                if time.time() >= t_deadline:
                    raise TimeoutError("IPBSM measurement timeout (ENDai never became 1)")
                time.sleep(poll)

            # 3) reset ENDai（効くなら）
            try:
                self.pv_end.put(0)
            except Exception:
                pass

            os.scandir('/atf/data/ipbsm/knob')

            # 4) wait for datafile updated AFTER this trigger
            t_deadline = mtime0 + float(file_wait)
            last_print = 0.0
            while True:
                try:
                    mtime = os.path.getmtime(self.datafile)
                except FileNotFoundError:
                    mtime = 0.0

                if mtime > mtime0:
                    break

                if time.time() >= t_deadline:
                    raise TimeoutError(f"Datafile not updated: mtime={mtime} baseline={mtime0}")

                # デバッグ表示（1秒に1回だけ）
                if time.time() - last_print > 1.0:
                    print(f"[IPBSM] waiting file update: mtime={mtime} baseline={mtime0}", flush=True)
                    last_print = time.time()

                time.sleep(poll)

            # 5) read dat
            with open(self.datafile, "rb") as f:
                raw = f.read()

            res = decode_ipbsm_dat(raw)
            modulation = float(res["modulation"])
            error = abs(float(res["error"]))
            return modulation, error


        



    def vary_magnet_position(self, name: str, dx: float, dy: float,
                            wait: bool = True,
                            max_attempts: int = 5,
                            attempt_timeout: float = 30.0,
                            settle_dt: float = 0.5,
                            tol: float = 0.01,
                            use_dotrim: bool = True):
        """
        Move magnet/mover by relative offset (dx, dy).
        - Writes DES:X, DES:Y with current + delta.
        - Optionally triggers :DOTRIM (if exists).
        - Waits until readback PVs reach target within tol.

        tol unit must match your readback PV units (e.g. mm or um).
        """

        pv_des_x = f"{name}:DES:X"
        pv_des_y = f"{name}:DES:Y"

        pv_rb_x = f"{name}:X"       
        pv_rb_y = f"{name}:Y"     
        pv_dotrim = f"{name}:TRIM" 

        # read current desired and build targets
        x0 = caget(pv_des_x)
        y0 = caget(pv_des_y)
        if x0 is None or y0 is None:
            raise RuntimeError(f"Failed to read DES PVs: {pv_des_x}, {pv_des_y}")

        x_target = float(x0) + float(dx)
        y_target = float(y0) + float(dy)

        # write targets
        caput(pv_des_x, x_target)
        caput(pv_des_y, y_target)

        if not wait:
            return

        time.sleep(settle_dt)

        attempt = 0
        last_xm = last_ym = None
        any_timeout = False

        while attempt < max_attempts:
            attempt += 1

            # trigger trim if available
            if use_dotrim:
                try:
                    caput(pv_dotrim, 1)
                    time.sleep(settle_dt)
                except Exception:
                    pass

                # wait until DOTRIM becomes 0 (done) or timeout
                t_deadline = time.time() + float(attempt_timeout)
                timed_out_this_attempt = False

                while True:
                    state = 0
                    try:
                        state = int(caget(pv_dotrim) or 0)
                    except Exception:
                        state = 0

                    if state == 0:
                        break

                    if time.time() >= t_deadline:
                        timed_out_this_attempt = True
                        any_timeout = True
                        try:
                            caput(pv_dotrim, 0)
                        except Exception:
                            pass
                        time.sleep(1.0)
                        break

                    time.sleep(settle_dt)

            # readbacks
            try:
                xm = float(caget(pv_rb_x) or 0.0)
                ym = float(caget(pv_rb_y) or 0.0)
            except Exception:
                xm = ym = 0.0

            last_xm, last_ym = xm, ym

            # success
            if abs(xm - x_target) <= tol and abs(ym - y_target) <= tol:
                return  # SUCCESS

        # max_attempts まで行っても成功しなかった
        raise TimeoutError(
            f"{name} did not reach target. last=({last_xm},{last_ym}) "
            f"target=({x_target},{y_target}) tol={tol} any_timeout={any_timeout}"
        )

    def _build_linear_deltas(self, knob_values: dict):
        dpos = defaultdict(lambda: [0.0, 0.0])
        for knob, k in knob_values.items():
            for mag, (ax, ay) in self.linear_matrix[knob].items():
                dpos[mag][0] = ax * k
                dpos[mag][1] = ay * k
        return {m: (v[0], v[1]) for m, v in dpos.items()}

    def _build_nonlinear_deltas(self, knob_values: dict):
        dcur = defaultdict(float)
        for knob, k in knob_values.items():
            for mag, a in self.nonlinear_matrix[knob].items():
                dcur[mag] = a * k
        return dict(dcur)
    
    
    def apply_linear_knobs(self, knob_values: dict,
                        tol=15, timeout=30.0, poll=0.05,
                        settle_dt=0.5, use_trim=True):
        dpos = self._build_linear_deltas(knob_values)
        self._apply_positions_batch(dpos, tol=tol, timeout=timeout,
                                    poll=poll, settle_dt=settle_dt, use_trim=use_trim)

    def apply_nonlinear_knobs(self, knob_values: dict,
                            tol=0.01, timeout=15.0, poll=0.05):
        dcur = self._build_nonlinear_deltas(knob_values)
        self._apply_currents_batch(dcur, tol=tol, timeout=timeout, poll=poll)


    def _apply_positions_batch(self, name_to_dxy: dict, *,
                            tol, timeout, poll, settle_dt, use_trim):
        target = {}
        for mag, (dx, dy) in name_to_dxy.items():
            pv_des_x = f"{mag}:MAG:DES:X"
            pv_des_y = f"{mag}:MAG:DES:Y"
            x0 = caget(pv_des_x); y0 = caget(pv_des_y)
            if x0 is None or y0 is None:
                raise RuntimeError(f"DES read failed: {pv_des_x} / {pv_des_y}")

            xt = float(x0) + float(dx)
            yt = float(y0) + float(dy)
            target[mag] = (xt, yt)

            caput(pv_des_x, xt)
            caput(pv_des_y, yt)

        if not target:
            return

        time.sleep(settle_dt)

        if use_trim:
            for mag in target.keys():
                caput(f"{mag}:TRIM", 1)
            time.sleep(settle_dt)

        deadline = time.time() + float(timeout)
        pending = set(target.keys())

        while pending:
            done = []
            for mag in list(pending):
                if use_trim:
                    st = caget(f"{mag}:TRIM")
                    if st is None or int(st) != 0:
                        continue

                xm = caget(f"{mag}:MAG:X")
                ym = caget(f"{mag}:MAG:Y")
                if xm is None or ym is None:
                    continue

                xt, yt = target[mag]
                if abs(float(xm) - xt) <= tol and abs(float(ym) - yt) <= tol:
                    done.append(mag)

            for mag in done:
                pending.remove(mag)

            if not pending:
                return

            if time.time() >= deadline:
                raise TimeoutError(f"Position timeout. pending={sorted(pending)}")

            time.sleep(poll)


    def _apply_currents_batch(self, name_to_dI: dict, *,
                            tol, timeout, poll):
        target = {}
        for mag, dI in name_to_dI.items():
            pv_set = f"{mag}:currentWrite"
            cur0 = caget(pv_set)
            if cur0 is None:
                raise RuntimeError(f"Current setpoint read failed: {pv_set}")

            It = float(cur0) + float(dI)
            target[mag] = It
            caput(pv_set, It)

        if not target:
            return

        deadline = time.time() + float(timeout)
        pending = set(target.keys())

        while pending:
            done = []
            for mag in list(pending):
                rb = caget(f"{mag}:current")
                if rb is None:
                    continue
                if abs(float(rb) - target[mag]) <= tol:
                    done.append(mag)

            for mag in done:
                pending.remove(mag)

            if not pending:
                return

            if time.time() >= deadline:
                raise TimeoutError(f"Current timeout. pending={sorted(pending)}")

            time.sleep(poll)

class EPICSIPBSMController:
    """
    Controller wrapper for real machine.
    Optimizer expects: apply_knobs(dict)->None and get_ipbsm()->(y,yerr)
    """
    def __init__(self, interface: IPBSMInterface, mode_name: str = "linear"):
        self.interface = interface
        self.mode_name = str(mode_name).lower()

    def apply_knobs(self, knob_values: Dict[str, float]) -> None:
        # mode_name は GUI のモード選択（linear / nonlinear2 / nonlinear4 など）に合わせる
        if self.mode_name.startswith("linear"):
            # knob_values: {"Ay":..., "Ey":..., "Coup2":...} など
            self.interface.apply_linear_knobs(knob_values)
        else:
            # knob_values: {"Y24":..., "Y46":...} or 4D etc
            self.interface.apply_nonlinear_knobs(knob_values)

    def get_ipbsm(self) -> Tuple[float, float]:
        return self.interface.get_ipbsm()



# ----------------------------

@dataclass
class GaussianFitResult:
    ok: bool
    mu: List[float]
    cov: List[List[float]]
    amp: float
    ln_amp: float
    ridge: float
    mode: str  # "diag" or "full"
    residual_rms: float
    n_points: int


def _design_matrix_quadratic(X: np.ndarray, mode: str) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int,int]]]]:
    """
    Build design matrix for quadratic regression of ln(y):
      ln y ≈ c + sum_i b_i x_i + sum_i d_i x_i^2 + sum_{i<j} e_{ij} x_i x_j (full only)
    Returns (Phi, term_meta)

    term_meta maps to reconstruct quadratic form.
    """
    n, d = X.shape
    cols = []
    meta = []

    # constant
    cols.append(np.ones((n, 1)))
    meta.append(("c", (-1, -1)))

    # linear
    cols.append(X)
    for i in range(d):
        meta.append(("b", (i, i)))

    # squared
    cols.append(X * X)
    for i in range(d):
        meta.append(("d", (i, i)))

    if mode == "full":
        cross_terms = []
        for i in range(d):
            for j in range(i+1, d):
                cross_terms.append((X[:, i] * X[:, j]).reshape(n, 1))
                meta.append(("e", (i, j)))
        if cross_terms:
            cols.append(np.hstack(cross_terms))

    Phi = np.hstack(cols)
    return Phi, meta

def fit_gaussian_from_samples(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "diag",
    ridge: float = 1e-6,
    eps_y: float = 1e-6,
    y_cap: Optional[float] = None,
    weighted: bool = True,
) -> GaussianFitResult:
    """
    Fit Gaussian peak (single-mode) by quadratic regression on ln(y).
    - mode="diag": no cross terms, covariance forced diagonal.
    - mode="full": includes cross terms, covariance full (weak correlation possible).

    Returns mu, cov, amplitude estimate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    if n < max(6, d + 2):
        return GaussianFitResult(
            ok=False,
            mu=[0.0]*d,
            cov=np.eye(d).tolist(),
            amp=float(np.max(y)) if n else 0.0,
            ln_amp=float(np.log(max(float(np.max(y)) if n else 1e-6, 1e-6))),
            ridge=ridge,
            mode=mode,
            residual_rms=float("inf"),
            n_points=n,
        )

    if y_cap is None:
        y_clip = np.maximum(y, eps_y)
    else:
        y_clip = np.clip(y, eps_y, float(y_cap))
    yln = np.log(y_clip)

    Phi, meta = _design_matrix_quadratic(X, mode=mode)

    # Weighted ridge regression on ln(y): down-weight tiny y (near floor)
    if weighted:
        wgt = np.sqrt(y_clip / max(float(np.max(y_clip)), eps_y))
        W = wgt.reshape(-1, 1)
        Phi_w = Phi * W
        yln_w = yln * wgt
        A = Phi_w.T @ Phi_w
        A += ridge * np.eye(A.shape[0])
        w = np.linalg.solve(A, Phi_w.T @ yln_w)
    else:
        A = Phi.T @ Phi
        A += ridge * np.eye(A.shape[0])
        w = np.linalg.solve(A, Phi.T @ yln)

    yln_hat = Phi @ w
    residual = yln - yln_hat
    residual_rms = float(np.sqrt(np.mean(residual**2)))

    # Reconstruct quadratic form:
    # ln y = c + b^T x - 0.5 x^T Q x  + (linear in mu absorbed)
    # Our paramization:
    # ln y = c + sum b_i x_i + sum d_i x_i^2 + sum e_ij x_i x_j
    # Quadratic coefficient matrix M such that x^T M x equals:
    #   sum d_i x_i^2 + sum_{i<j} e_ij x_i x_j
    # Then Q = -2 M (must be PD)
    idx = 0
    c = float(w[idx]); idx += 1
    b = np.array(w[idx:idx+d], dtype=float); idx += d
    dcoef = np.array(w[idx:idx+d], dtype=float); idx += d

    M = np.zeros((d, d), dtype=float)
    for i in range(d):
        M[i, i] = dcoef[i]
    if mode == "full":
        # cross terms correspond to x_i x_j with coefficient e_ij
        for i in range(d):
            for j in range(i+1, d):
                if idx >= len(w):
                    break
                eij = float(w[idx]); idx += 1
                # x^T M x has 2*M_ij x_i x_j
                M[i, j] = 0.5 * eij
                M[j, i] = 0.5 * eij

    Q = -2.0 * M  # want Q positive definite
    # Symmetrize
    Q = 0.5 * (Q + Q.T)

    # Force PD by eigenvalues clamp
    try:
        eigvals, eigvecs = np.linalg.eigh(Q)
        ev_min = 1e-6
        eigvals = np.maximum(eigvals, ev_min)
        Q_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    except Exception:
        Q_pd = np.eye(d)

    # For diag mode, zero off-diagonals
    if mode == "diag":
        Q_pd = np.diag(np.diag(Q_pd))

    try:
        cov = np.linalg.inv(Q_pd)
        mu = cov @ b
        ln_amp = c + 0.5 * float(mu.T @ Q_pd @ mu)
        # avoid overflow / non-physical amplitudes
        ln_amp = max(min(ln_amp, 10.0), -50.0)
        amp = float(np.exp(ln_amp))
    except Exception:
        cov = np.eye(d)
        mu = np.zeros(d)
        ln_amp = float(c)
        amp = float(np.exp(ln_amp))

    return GaussianFitResult(
        ok=True,
        mu=mu.tolist(),
        cov=cov.tolist(),
        amp=amp,
        ln_amp=ln_amp,
        ridge=ridge,
        mode=mode,
        residual_rms=residual_rms,
        n_points=n,
    )

def bootstrap_fit(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    ridge: float,
    n_boot: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Bootstrap Gaussian fit to estimate uncertainty of mu and cov diagonals.
    Returns dict with:
      mu_mean, mu_std, cov_mean, cov_diag_mean, cov_diag_std
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    rng = rng or np.random.default_rng()

    # filter pathological fits (log-quad fit can blow up if all y are tiny)
    mu_abs_max = 10.0
    cov_diag_min = 1e-8
    cov_diag_max = 100.0

    mus = []
    covs = []
    ok_count = 0
    for _ in range(max(0, n_boot)):
        idx = rng.integers(0, n, size=n)
        fr = fit_gaussian_from_samples(X[idx], y[idx], mode=mode, ridge=ridge)
        if fr.ok:
            mu_v = np.array(fr.mu, float)
            cov_m = np.array(fr.cov, float)
            diag = np.diag(cov_m) if cov_m.ndim == 2 else np.array([])
            if np.any(~np.isfinite(mu_v)) or np.any(~np.isfinite(cov_m)):
                continue
            if np.any(np.abs(mu_v) > mu_abs_max):
                continue
            if diag.size and (np.any(diag < cov_diag_min) or np.any(diag > cov_diag_max)):
                continue
            mus.append(mu_v)
            covs.append(cov_m)
            ok_count += 1

    if ok_count == 0:
        mu_mean = np.zeros(d)
        mu_std = np.full(d, np.inf)
        cov_mean = np.eye(d)
    else:
        mu_stack = np.vstack(mus)
        cov_stack = np.stack(covs, axis=0)
        mu_mean = np.mean(mu_stack, axis=0)
        mu_std = np.std(mu_stack, axis=0, ddof=1) if ok_count > 1 else np.zeros(d)
        cov_mean = np.mean(cov_stack, axis=0)

    cov_diag = np.diag(cov_mean)
    if ok_count > 1:
        cov_diag_std = np.std(np.stack([np.diag(c) for c in covs], axis=0), axis=0, ddof=1)
    else:
        cov_diag_std = np.zeros(d)

    return {
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "cov_mean": cov_mean,
        "cov_diag_mean": cov_diag,
        "cov_diag_std": cov_diag_std,
    }

# ----------------------------
# Simple GP regression (no sklearn)
# ----------------------------

@dataclass
class GPParams:
    length_scale: float = 1.0
    signal_var: float = 1.0
    noise_var: float = 1e-4

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray:
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    # squared Euclidean distances
    # ||x - x'||^2 = x^2 + x'^2 - 2 x x'
    x1_sq = np.sum(X1*X1, axis=1, keepdims=True)
    x2_sq = np.sum(X2*X2, axis=1, keepdims=True).T
    d2 = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
    return signal_var * np.exp(-0.5 * d2 / (length_scale**2 + 1e-12))

class SimpleGP:
    def __init__(self, params: GPParams):
        self.params = params
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        n = X.shape[0]
        K = rbf_kernel(X, X, self.params.length_scale, self.params.signal_var)
        K += (self.params.noise_var + 1e-12) * np.eye(n)
        # Cholesky
        self.L = np.linalg.cholesky(K)
        # Solve for alpha: K^-1 y = L^-T L^-1 y
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        self.X = X
        self.y = y

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = np.asarray(Xs, float)
        if self.X is None:
            mu = np.zeros(Xs.shape[0])
            std = np.ones(Xs.shape[0])
            return mu, std
        Ks = rbf_kernel(self.X, Xs, self.params.length_scale, self.params.signal_var)
        mu = (Ks.T @ self.alpha).reshape(-1)
        # variance: k(x,x) - v^T v where v = L^-1 Ks
        v = np.linalg.solve(self.L, Ks)
        kss = np.diag(rbf_kernel(Xs, Xs, self.params.length_scale, self.params.signal_var))
        var = np.maximum(kss - np.sum(v*v, axis=0), 1e-12)
        std = np.sqrt(var)
        return mu, std

# ----------------------------
# Acquisition functions
# ----------------------------

def acq_ucb(mu: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
    return mu + beta * std

def acq_ei(mu: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
    # Expected improvement for maximization
    imp = mu - y_best - xi
    z = imp / np.maximum(std, 1e-12)
    return imp * normal_cdf(z) + std * normal_pdf(z)

# ----------------------------
# Optimizer
# ----------------------------

@dataclass
class OptimizerConfig:
    mode_name: str  # "linear", "nonlinear2", "nonlinear4"
    method: str     # "GF", "BO", "LBO"
    acquisition: str  # "UCB" or "EI" (for BO/LBO)
    params: List[str]
    bounds: Dict[str, Tuple[float, float]]
    init_sigma: Dict[str, float]
    meas_sigma: float = 0.01
    expected_y_max: Optional[float] = None  # e.g. 0.8 for synthetic test
    stop_modulation: Optional[float] = 0.65  # stop immediately if y >= this (GF/BO/LBO)
    knob_step: float = 0.01              # quantization step for ALL knob params (linear/nonlinear)
    gf_weight_peak: float = 1.0          # GF policy weight: peak-seeking (use fitted mu)
    gf_weight_refine: float = 1.0        # GF policy weight: localization / precision improvement
    gf_jitter_frac: float = 0.25         # GF peak-seeking jitter as fraction of sigma
    max_steps: int = 60
    stop_mu_sigma: float = 0.02       # stop when bootstrap std(mu_i) < this for all dims
    stop_y_sigma: float = 0.01        # stop when predicted peak y std small (not strict)
    seed: int = 123
    n_init_random: int = 8
    n_candidates: int = 6000
    n_bootstrap: int = 60
    ridge_fit: float = 1e-4
    gp_length_scale: float = 1.2
    gp_signal_var: float = 0.15
    gp_noise_var: float = 1e-4
    ucb_beta: float = 2.0
    ei_xi: float = 0.0
    probe_scale: float = 1.0          # for GF probing around peak
    init_strategy: str = "structured"  # "structured" or "random"
    lbo_line_points: int = 200
    lbo_dir_strategy: str = "cov_maxvar"  # "cov_maxvar" or "random"

@dataclass
class StepRecord:
    step: int
    t_iso: str
    x: Dict[str, float]
    y: float
    y_err: float
    chosen_by: str

class StopFlag:
    def __init__(self):
        self._stop = False
    def request_stop(self):
        self._stop = True
    def is_stopped(self) -> bool:
        return self._stop

class Optimizer:
    def _structured_init_points(self) -> List[np.ndarray]:
        """
        Deterministic-ish initialization around the origin (current knob baseline):
          x0 = 0
          x0 +/- sigma0_i along each axis
        This helps avoid the "all points far from peak => unstable fit" failure.
        """
        d = len(self.cfg.params)
        x0 = np.zeros(d, dtype=float)
        sig0 = np.array([max(1e-6, float(self.cfg.init_sigma.get(p, 0.5))) for p in self.cfg.params], float)

        pts = [x0]
        for i in range(d):
            xp = x0.copy(); xp[i] += sig0[i]
            xm = x0.copy(); xm[i] -= sig0[i]
            pts.append(xp); pts.append(xm)

        lo, hi = self._bounds_arrays()
        pts = [clamp(p, lo, hi) for p in pts]
        return pts

    def __init__(
        self,
        controller: BaseIPBSMController,
        config: OptimizerConfig,
        out_dir: Path,
        progress_cb: Optional[Callable[[int, Dict], None]] = None,
        stop_flag: Optional[StopFlag] = None,
    ):
        self.controller = controller
        self.cfg = config
        self.out_dir = ensure_dir(out_dir)
        self.progress_cb = progress_cb
        self.stop_flag = stop_flag or StopFlag()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.X = []
        self.y = []
        self.yerr = []
        self.records: List[StepRecord] = []

        self._save_config()

    def _save_config(self):
        cfg_path = self.out_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2, ensure_ascii=False)

    def _quantize_knob(self, v: float) -> float:
        step = float(self.cfg.knob_step)
        return round(v / step) * step

    def _log_step(self, rec: StepRecord):
        self.records.append(rec)
        csv_path = self.out_dir / "measurements.csv"
        is_new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["step", "t_iso"] + self.cfg.params + ["modulation", "mod_err", "chosen_by"])

            x_q = [self._quantize_knob(rec.x[p]) for p in self.cfg.params]

            w.writerow(
                [rec.step, rec.t_iso]
                + x_q
                + [rec.y, rec.y_err, rec.chosen_by]
            )


    def _emit(self, step: int, info: Dict):
        if self.progress_cb:
            self.progress_cb(step, info)

    def _bounds_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.cfg.bounds[p][0] for p in self.cfg.params], float)
        hi = np.array([self.cfg.bounds[p][1] for p in self.cfg.params], float)
        return lo, hi

    def _x_dict(self, x_vec: np.ndarray) -> Dict[str, float]:
        return {p: float(x_vec[i]) for i, p in enumerate(self.cfg.params)}

    def _measure_at(self, x_vec: np.ndarray, chosen_by: str) -> Tuple[float, float]:
        # Quantize knobs to hardware-like step and clamp
        lo, hi = self._bounds_arrays()
        knob_step = max(1e-12, float(getattr(self.cfg, "knob_step", 0.01)))

        xq = np.round(np.asarray(x_vec, float) / knob_step) * knob_step
        xq = clamp(xq, lo, hi)

        meas_idx = len(self.X) + 1  # <- これが「今何回目の測定か」

        self.controller.apply_knobs(self._x_dict(xq))
        y, yerr = self.controller.get_ipbsm()

        best = max([float(y)] + [float(v) for v in self.y]) if self.y else float(y)
        print(f"[MEAS] i={meas_idx} by={chosen_by}   y={float(y):.6f} ± {float(yerr):.6f}  best={best:.6f}")

        return float(y), float(yerr)

    def _random_point(self) -> np.ndarray:
        lo, hi = self._bounds_arrays()
        return lo + (hi - lo) * self.rng.random(lo.shape[0])

    def _candidate_points(self, n: int) -> np.ndarray:
        lo, hi = self._bounds_arrays()
        return lo + (hi - lo) * self.rng.random((n, lo.shape[0]))

    def _fit_and_bootstrap(self) -> Dict:
        mode = "diag" if self.cfg.mode_name == "linear" else "full"
        X = np.asarray(self.X, float)
        y = np.asarray(self.y, float)

        fit = fit_gaussian_from_samples(X, y, mode=mode, ridge=self.cfg.ridge_fit, y_cap=self.cfg.expected_y_max)

        # Stabilize: clamp mu to bounds and covariance diagonal to reasonable range
        lo, hi = self._bounds_arrays()
        mu = np.array(fit.mu, float)
        mu = clamp(mu, lo, hi)
        cov = np.array(fit.cov, float)
        cov = 0.5 * (cov + cov.T)

        sig0 = np.array([max(1e-6, float(self.cfg.init_sigma.get(p, 0.5))) for p in self.cfg.params], float)
        min_sig = np.maximum(0.05, 0.2 * sig0)
        max_sig = np.minimum(5.0, 3.0 * sig0 + 1.0)

        diag = np.diag(cov)
        diag = np.clip(diag, min_sig**2, max_sig**2)
        cov = cov.copy()
        for i in range(len(diag)):
            cov[i, i] = diag[i]

        fit.mu = mu.tolist()
        fit.cov = cov.tolist()

        boot = bootstrap_fit(X, y, mode=mode, ridge=self.cfg.ridge_fit, n_boot=self.cfg.n_bootstrap, rng=self.rng)

        out = {
            "fit": fit,
            "boot": boot,
        }
        # Save summary
        summary = {
            "fit": asdict(fit),
            "boot": {
                "mu_mean": boot["mu_mean"].tolist(),
                "mu_std": boot["mu_std"].tolist(),
                "cov_diag_mean": boot["cov_diag_mean"].tolist(),
                "cov_diag_std": boot["cov_diag_std"].tolist(),
            }
        }
        with open(self.out_dir / "fit_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return out

    def _stop_by_precision(self, boot: Dict) -> bool:

        mu_std = np.asarray(boot["mu_std"], float)
        if not np.all(np.isfinite(mu_std)):
            return False
        return bool(np.all(mu_std < self.cfg.stop_mu_sigma))

    def _stop_by_modulation(self, y: float) -> bool:
        thr = getattr(self.cfg, 'stop_modulation', None)
        if thr is None:
            return False
        try:
            return bool(np.isfinite(y) and float(y) >= float(thr))
        except Exception:
            return False

    def _propose_next_GF(self, fit: GaussianFitResult, boot: Dict) -> np.ndarray:
        lo, hi = self._bounds_arrays()
        d = len(self.cfg.params)

        # Peak estimate (parametric): use fitted/bootstrapped mu
        mu = np.asarray(boot.get("mu_mean", fit.mu), float)

        # Covariance estimate
        if "cov_mean" in boot:
            cov = np.asarray(boot["cov_mean"], float)
        else:
            cov = np.asarray(fit.cov, float)
        cov = 0.5 * (cov + cov.T)

        sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))

        # --- choose GF policy by weights ---
        w_peak = float(getattr(self.cfg, "gf_weight_peak", 1.0))
        w_ref  = float(getattr(self.cfg, "gf_weight_refine", 1.0))
        w_peak = max(0.0, w_peak)
        w_ref  = max(0.0, w_ref)
        w_sum = w_peak + w_ref
        p_peak = (w_peak / w_sum) if w_sum > 0 else 0.0

        if self.rng.random() < p_peak:
            # Policy A: peak-seeking (target mu, with small jitter to avoid duplicates)
            jitter_frac = float(getattr(self.cfg, "gf_jitter_frac", 0.25))
            jitter_frac = max(0.0, jitter_frac)
            x = mu.copy()
            x = x + self.rng.normal(size=d) * (jitter_frac * sig)
            x = clamp(x, lo, hi)
            return x

        # Policy B: localization / precision improvement
        # Prefer probing along the principal axis of covariance (most uncertain direction).
        try:
            ev, evec = np.linalg.eigh(cov)
            axis = int(np.argmax(ev))
            v = evec[:, axis]
            v = v / (np.linalg.norm(v) + 1e-12)
            scale = float(np.sqrt(max(ev[axis], 1e-12)))
            direction = 1.0 if (len(self.X) % 2 == 0) else -1.0
            step = float(self.cfg.probe_scale) * scale
            x = mu + direction * step * v
        except Exception:
            # Fallback: coordinate axis with largest bootstrap mu uncertainty if available
            mu_std = np.asarray(boot.get("mu_std", sig), float)
            axis = int(np.argmax(mu_std))
            direction = 1.0 if (len(self.X) % 2 == 0) else -1.0
            step = float(self.cfg.probe_scale) * float(sig[axis])
            x = mu.copy()
            x[axis] = x[axis] + direction * step

        x = clamp(x, lo, hi)

        # If duplicates, random point
        if len(self.X) > 0:
            X = np.asarray(self.X, float)
            if np.min(np.linalg.norm(X - x.reshape(1, -1), axis=1)) < 1e-6:
                x = self._random_point()
        return x

    def _propose_next_BO(self) -> np.ndarray:
        lo, hi = self._bounds_arrays()
        X = np.asarray(self.X, float)
        y = np.asarray(self.y, float)

        gp = SimpleGP(GPParams(
            length_scale=self.cfg.gp_length_scale,
            signal_var=self.cfg.gp_signal_var,
            noise_var=self.cfg.gp_noise_var,
        ))
        gp.fit(X, y)

        cand = self._candidate_points(self.cfg.n_candidates)
        mu, std = gp.predict(cand)

        if self.cfg.acquisition.upper() == "EI":
            a = acq_ei(mu, std, y_best=float(np.max(y)), xi=self.cfg.ei_xi)
        else:
            a = acq_ucb(mu, std, beta=self.cfg.ucb_beta)

        x_next = cand[int(np.argmax(a))]
        return clamp(x_next, lo, hi)

    def _propose_next_LBO(self, boot: Dict) -> np.ndarray:
        lo, hi = self._bounds_arrays()
        X = np.asarray(self.X, float)
        y = np.asarray(self.y, float)

        # Build GP for acquisition on a line
        gp = SimpleGP(GPParams(
            length_scale=self.cfg.gp_length_scale,
            signal_var=self.cfg.gp_signal_var,
            noise_var=self.cfg.gp_noise_var,
        ))
        gp.fit(X, y)

        # Center at current best
        x0 = X[int(np.argmax(y))].copy()

        # Direction
        d = X.shape[1]
        if self.cfg.lbo_dir_strategy == "random" or len(X) < 8:
            v = self.rng.normal(size=d)
            v /= (np.linalg.norm(v) + 1e-12)
        else:
            cov = np.asarray(boot["cov_mean"], float)
            cov = 0.5 * (cov + cov.T)
            ev, evec = np.linalg.eigh(cov)
            v = evec[:, int(np.argmax(ev))]
            v = v / (np.linalg.norm(v) + 1e-12)

        # Determine feasible t range so that x0 + t v within bounds
        t_lo = -1e9
        t_hi =  1e9
        for i in range(d):
            if abs(v[i]) < 1e-12:
                continue
            t1 = (lo[i] - x0[i]) / v[i]
            t2 = (hi[i] - x0[i]) / v[i]
            tmin, tmax = (min(t1, t2), max(t1, t2))
            t_lo = max(t_lo, tmin)
            t_hi = min(t_hi, tmax)

        if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
            return self._random_point()

        ts = np.linspace(t_lo, t_hi, int(self.cfg.lbo_line_points))
        line = x0.reshape(1, -1) + ts.reshape(-1, 1) * v.reshape(1, -1)

        mu, std = gp.predict(line)
        if self.cfg.acquisition.upper() == "EI":
            a = acq_ei(mu, std, y_best=float(np.max(y)), xi=self.cfg.ei_xi)
        else:
            a = acq_ucb(mu, std, beta=self.cfg.ucb_beta)

        x_next = line[int(np.argmax(a))]
        return clamp(x_next, lo, hi)

    def run(self) -> Dict:
        lo, hi = self._bounds_arrays()
        d = len(self.cfg.params)

        # Initialize (structured first, then random fill)
        init_pts = []
        if self.cfg.init_strategy == "structured":
            init_pts = self._structured_init_points()

        # measure structured points up to n_init_random
        for x in init_pts:
            if len(self.X) >= self.cfg.n_init_random or len(self.X) >= self.cfg.max_steps:
                break
            if self.stop_flag.is_stopped():
                break
            y, yerr = self._measure_at(x, chosen_by="init_structured")
            self.X.append(x)
            self.y.append(y)
            self.yerr.append(yerr)
            rec = StepRecord(
                step=len(self.X),
                t_iso=_dt.datetime.now().isoformat(timespec="seconds"),
                x=self._x_dict(x),
                y=y,
                y_err=yerr,
                chosen_by="init_structured",
            )
            self._log_step(rec)
            self._emit(len(self.X), {
                "phase": "init",
                "x": rec.x,
                "y": y,
                "y_err": yerr,
            })
            if self._stop_by_modulation(y):
                self.stop_flag.request_stop()
                self._emit(len(self.X), {"phase": "stop", "reason": "modulation_threshold_hit", "y": y})
                break

        # random fill
        while len(self.X) < self.cfg.n_init_random and len(self.X) < self.cfg.max_steps:
            if self.stop_flag.is_stopped():
                break
            x = self._random_point()
            y, yerr = self._measure_at(x, chosen_by="init_random")
            self.X.append(x)
            self.y.append(y)
            self.yerr.append(yerr)
            rec = StepRecord(
                step=len(self.X),
                t_iso=_dt.datetime.now().isoformat(timespec="seconds"),
                x=self._x_dict(x),
                y=y,
                y_err=yerr,
                chosen_by="init_random",
            )
            self._log_step(rec)
            self._emit(len(self.X), {
                "phase": "init",
                "x": rec.x,
                "y": y,
                "y_err": yerr,
            })
            if self._stop_by_modulation(y):
                self.stop_flag.request_stop()
                self._emit(len(self.X), {"phase": "stop", "reason": "modulation_threshold_hit", "y": y})
                break

        # Main loop# Main loop
        while len(self.X) < self.cfg.max_steps and (not self.stop_flag.is_stopped()):
            X_np = np.asarray(self.X, float)
            y_np = np.asarray(self.y, float)

            fit_pack = self._fit_and_bootstrap()
            fit: GaussianFitResult = fit_pack["fit"]
            boot = fit_pack["boot"]

            # Stop check by bootstrap mu precision (GF only)
            if self.cfg.method.upper() == "GF":
                if self._stop_by_precision(boot):
                    mu_std = np.asarray(boot.get("mu_std", np.full(len(self.cfg.params), np.nan)), float)
                    print(
                        f"[STOPCHK] n={len(self.X)} "
                        f"mu_std={ {p: float(s) for p, s in zip(self.cfg.params, mu_std)} } "
                        f"stop_mu_sigma={self.cfg.stop_mu_sigma}"
                    )
                    self.stop_flag.request_stop()
                    self._emit(len(self.X), {"phase": "stop", "reason": "mu_precision_reached"})
                    break

            # Propose next
            if self.cfg.method.upper() == "GF":
                x_next = self._propose_next_GF(fit, boot)
                chosen_by = "GF"
            elif self.cfg.method.upper() == "LBO":
                x_next = self._propose_next_LBO(boot)
                chosen_by = "LBO"
            else:
                x_next = self._propose_next_BO()
                chosen_by = "BO"

            # Avoid duplicates
            if len(self.X) > 0:
                if np.min(np.linalg.norm(X_np - x_next.reshape(1, -1), axis=1)) < 1e-6:
                    x_next = self._random_point()
                    chosen_by += "+jitter"

            # Measure
            y, yerr = self._measure_at(x_next, chosen_by=chosen_by)

            self.X.append(x_next)
            self.y.append(y)
            self.yerr.append(yerr)

            rec = StepRecord(
                step=len(self.X),
                t_iso=_dt.datetime.now().isoformat(timespec="seconds"),
                x=self._x_dict(x_next),
                y=y,
                y_err=yerr,
                chosen_by=chosen_by,
            )
            self._log_step(rec)

            self._emit(len(self.X), {
                "phase": "loop",
                "chosen_by": chosen_by,
                "x": rec.x,
                "y": y,
                "y_err": yerr,
                "best_y": float(np.max(self.y)),
            })
            if self._stop_by_modulation(y):
                self.stop_flag.request_stop()
                self._emit(len(self.X), {"phase": "stop", "reason": "modulation_threshold_hit", "y": y})
                break

        # Final fit
        final_pack = self._fit_and_bootstrap()
        final_fit: GaussianFitResult = final_pack["fit"]
        final_boot = final_pack["boot"]

        best_idx = int(np.argmax(np.asarray(self.y, float))) if self.y else -1
        best_x = self._x_dict(np.asarray(self.X[best_idx], float)) if best_idx >= 0 else {}
        best_y = float(self.y[best_idx]) if best_idx >= 0 else float("nan")

        out = {
            "out_dir": str(self.out_dir),
            "n_steps": len(self.X),
            "best_x": best_x,
            "best_y": best_y,
            "fit_mu": final_fit.mu,
            "fit_cov": final_fit.cov,
            "boot_mu_mean": final_boot["mu_mean"].tolist(),
            "boot_mu_std": final_boot["mu_std"].tolist(),
            "fit_mu_err": final_boot["mu_std"][0],
            "fit_sigma_err": 0.5 * final_boot["cov_diag_std"][0] / max(1e-12, np.sqrt(final_boot["cov_diag_mean"][0])),}

        with open(self.out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        return out

# ----------------------------
# Plotting
# ----------------------------

def plot_results(
    cfg: OptimizerConfig,
    out_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    fit: GaussianFitResult,
    boot: Optional[Dict] = None,
    save_prefix: str = "",
) -> List[Path]:
    """
    Save 1D and 2D slice plots based on fitted Gaussian.
    - 1D: for each parameter
    - 2D: for each pair (or subset depending on dims)
    Returns list of saved png paths.
    """
    import matplotlib.pyplot as plt

    out_dir = ensure_dir(out_dir)
    params = cfg.params
    d = len(params)
    lo = np.array([cfg.bounds[p][0] for p in params], float)
    hi = np.array([cfg.bounds[p][1] for p in params], float)

    mu = np.array(fit.mu, float)
    cov = np.array(fit.cov, float)
    cov = 0.5*(cov+cov.T)
    amp = float(fit.amp)
    if cfg.expected_y_max is not None:
        amp = min(amp, float(cfg.expected_y_max))

    # Helper to evaluate fitted gaussian slice
    try:
        Q = np.linalg.inv(cov)
    except Exception:
        Q = np.eye(d)

    def gaussian(x_vec: np.ndarray) -> float:
        dx = (x_vec - mu).reshape(-1, 1)
        val = amp * float(np.exp(-0.5 * (dx.T @ Q @ dx)))
        return val

    saved = []

    # 1D plots
    for i, p in enumerate(params):
        xs = np.linspace(lo[i], hi[i], 250)
        ys = []
        for x1 in xs:
            x_vec = mu.copy()
            x_vec[i] = x1
            ys.append(gaussian(x_vec))
        ys = np.array(ys, float)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(X[:, i], y, yerr=yerr, fmt="o", capsize=2)
        ax.plot(xs, ys)
        ax.set_xlabel(p)
        ax.set_ylabel("IPBSM modulation")
        ax.set_title(f"1D slice fit: {p}")

        # --- Fit parameter text (with uncertainties, 1D case) ---
        sigma_i = math.sqrt(max(cov[i, i], 1e-12))

        # --- uncertainties from bootstrap ---
        mu_err = None
        sigma_err = None

        if boot is not None:
            # μ uncertainty (most important)
            mu_err = float(boot["mu_std"][i])

            # σ uncertainty (from variance uncertainty, approximate)
            var_mean = float(boot["cov_diag_mean"][i])
            var_std  = float(boot["cov_diag_std"][i])
            sigma_err = 0.5 * var_std / max(1e-12, math.sqrt(var_mean))

        # --- text formatting ---
        txt_lines = [
            f"mu[{p}] = {mu[i]:+.4f}" + (f" ± {mu_err:.4f}" if mu_err is not None else ""),
            f"sigma[{p}] = {sigma_i:.4f}" + (f" ± {sigma_err:.4f}" if sigma_err is not None else ""),
            f"amp = {amp:.4f}",
            f"resid_rms(ln) = {fit.residual_rms:.4f}",
            f"n = {fit.n_points}",
        ]

        txt = "\n".join(txt_lines)

        ax.text(
            0.98, 0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="right",
        )

        name = f"{save_prefix}1D_{p}.png" if save_prefix else f"1D_{p}.png"
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # 2D plots (all pairs)
    # For d=2 -> 1 plot, d=3 -> 3 plots, d=4 -> 6 plots
    pairs = []
    for i in range(d):
        for j in range(i+1, d):
            pairs.append((i, j))

    for (i, j) in pairs:
        nx = 120
        ny = 120
        xs = np.linspace(lo[i], hi[i], nx)
        ys = np.linspace(lo[j], hi[j], ny)
        Z = np.zeros((ny, nx), float)
        for iy, yj in enumerate(ys):
            for ix, xi in enumerate(xs):
                x_vec = mu.copy()
                x_vec[i] = xi
                x_vec[j] = yj
                Z[iy, ix] = gaussian(x_vec)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        vmin = 0.0
        vmax = float(cfg.expected_y_max) if cfg.expected_y_max is not None else float(np.max(Z) if Z.size else 1.0)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[lo[i], hi[i], lo[j], hi[j]],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(X[:, i], X[:, j], s=12)
        ax.set_xlabel(params[i])
        ax.set_ylabel(params[j])
        ax.set_title(f"2D slice heatmap: {params[i]} vs {params[j]}")
        fig.colorbar(im, ax=ax, label="predicted modulation")
        name = f"{save_prefix}2D_{params[i]}_{params[j]}.png" if save_prefix else f"2D_{params[i]}_{params[j]}.png"
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Save fit text as a file too
    txt_path = out_dir / (f"{save_prefix}fit_params.txt" if save_prefix else "fit_params.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Gaussian fit parameters (from quadratic ln fit)\n")
        f.write(json.dumps({
            "params": params,
            "mu": mu.tolist(),
            "cov": cov.tolist(),
            "amp": amp,
            "residual_rms_ln": fit.residual_rms,
            "n_points": fit.n_points,
            "mode": fit.mode,
        }, indent=2, ensure_ascii=False))
    saved.append(txt_path)

    return saved
