# -*- coding: utf-8 -*-
"""
Opt_IPBSM_GUI.py
PyQt6 GUI for Gaussian estimation / BO / LBO multi-dimensional optimization.

Layout (requested):
- Top: mode/controller/method + run/stop + status
- Middle: settings (left, scroll) + PNG gallery (right)
- Bottom: result + log (always visible)
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QMessageBox, QScrollArea, QFrame
)

from Opt_IPBSM import (
    Optimizer, OptimizerConfig, StopFlag,
    fit_gaussian_from_samples, plot_results, now_tag
)
from Synthetic_IPBSM_Controller import make_random_spec, SyntheticGaussianIPBSMController
from Opt_IPBSM import IPBSMInterface

from collections import defaultdict




class OptimizerWorker(QThread):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, optimizer: Optimizer):
        super().__init__()
        self.optimizer = optimizer

    def run(self):
        try:
            def cb(step, info):
                self.progress.emit({"step": step, "info": info})
            self.optimizer.progress_cb = cb
            out = self.optimizer.run()
            self.finished.emit(out)
        except Exception as e:
            self.failed.emit(str(e) + "\n" + traceback.format_exc())


PARAMSETS = {
    "linear": ["Ay", "Ey", "Coup2"],
    "nonlinear2": ["Y24", "Y46"],
    "nonlinear4": ["Y22", "Y26", "Y66", "Y44"],
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Opt_IPBSM (GF / BO / LBO)")

        self.stop_flag = StopFlag()
        self.worker: Optional[OptimizerWorker] = None
        self.last_out_dir: Optional[Path] = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # =========================
        # Top: mode/controller/method
        # =========================
        top = QHBoxLayout()
        outer.addLayout(top)

        top.addWidget(QLabel("Mode:"))
        self.mode_box = QComboBox()
        self.mode_box.addItems(["linear", "nonlinear2", "nonlinear4"])
        top.addWidget(self.mode_box)

        top.addWidget(QLabel("Controller:"))
        self.ctrl_box = QComboBox()
        self.ctrl_box.addItems(["test", "machine"])
        top.addWidget(self.ctrl_box)

        top.addWidget(QLabel("Method:"))
        self.method_box = QComboBox()
        self.method_box.addItems(["GF", "BO", "LBO"])
        top.addWidget(self.method_box)

        self.top_acq_lbl = QLabel("Acq (BO/LBO):")
        top.addWidget(self.top_acq_lbl)
        self.acq_box = QComboBox()
        self.acq_box.addItems(["UCB", "EI"])
        top.addWidget(self.acq_box)

        top.addStretch(1)

        # =========================
        # Run controls row
        # =========================
        runrow = QHBoxLayout()
        outer.addLayout(runrow)

        self.run_btn = QPushButton("Run Optimization")
        self.stop_btn = QPushButton("Stop (after current point)")
        self.stop_btn.setEnabled(False)

        self.get_btn = QPushButton("Get IPBSM (machine only)")
        self.get_btn.setEnabled(False)

        runrow.addWidget(self.run_btn)
        runrow.addWidget(self.stop_btn)
        runrow.addWidget(self.get_btn)

        self.status_lbl = QLabel("Status: idle")
        runrow.addWidget(self.status_lbl, stretch=1)

        # =========================
        # Middle: settings (left) + gallery (right)
        # =========================
        middle = QHBoxLayout()
        outer.addLayout(middle, stretch=1)

        # ---- Left scroll: settings ----
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        middle.addWidget(self.left_scroll, stretch=2)

        left_widget = QWidget()
        self.left_scroll.setWidget(left_widget)
        left = QVBoxLayout(left_widget)

        cfg_group = QGroupBox("Optimization Settings")
        left.addWidget(cfg_group)
        cfg_form = QFormLayout(cfg_group)
        self.cfg_form = cfg_form

        self.max_steps = QSpinBox(); self.max_steps.setRange(1, 9999); self.max_steps.setValue(60)
        self.n_init = QSpinBox(); self.n_init.setRange(1, 999); self.n_init.setValue(8)

        self.bounds_range = QDoubleSpinBox(); self.bounds_range.setRange(0.1, 100.0); self.bounds_range.setDecimals(3); self.bounds_range.setValue(2.0)

        # IMPORTANT: meas_sigma drives both scatter (test controller) and plotted error bars.
        self.meas_sigma = QDoubleSpinBox()
        self.meas_sigma.setRange(1e-4, 0.2)
        self.meas_sigma.setDecimals(4)
        self.meas_sigma.setValue(0.01)

        self.stop_mu_sigma = QDoubleSpinBox(); self.stop_mu_sigma.setRange(1e-6, 1.0); self.stop_mu_sigma.setDecimals(4); self.stop_mu_sigma.setValue(0.02)
        self.stop_modulation = QDoubleSpinBox(); self.stop_modulation.setRange(0.0, 1.0); self.stop_modulation.setDecimals(4); self.stop_modulation.setValue(0.65)
        self.gf_w_peak = QDoubleSpinBox(); self.gf_w_peak.setRange(0.0, 100.0); self.gf_w_peak.setDecimals(2); self.gf_w_peak.setValue(1.0)
        self.gf_w_refine = QDoubleSpinBox(); self.gf_w_refine.setRange(0.0, 100.0); self.gf_w_refine.setDecimals(2); self.gf_w_refine.setValue(1.0)
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31-1); self.seed.setValue(123)
        self.n_boot = QSpinBox(); self.n_boot.setRange(0, 500); self.n_boot.setValue(60)
        self.n_cand = QSpinBox(); self.n_cand.setRange(100, 50000); self.n_cand.setValue(6000)

        self.ucb_beta = QDoubleSpinBox(); self.ucb_beta.setRange(0.0, 50.0); self.ucb_beta.setDecimals(3); self.ucb_beta.setValue(2.0)
        self.ei_xi = QDoubleSpinBox(); self.ei_xi.setRange(0.0, 1.0); self.ei_xi.setDecimals(4); self.ei_xi.setValue(0.0)

        self.ridge_fit = QDoubleSpinBox(); self.ridge_fit.setRange(1e-10, 1e-1); self.ridge_fit.setDecimals(10); self.ridge_fit.setValue(1e-4)
        self.probe_scale = QDoubleSpinBox(); self.probe_scale.setRange(0.2, 3.0); self.probe_scale.setDecimals(3); self.probe_scale.setValue(1.0)

        self.gp_len = QDoubleSpinBox(); self.gp_len.setRange(1e-3, 50.0); self.gp_len.setDecimals(3); self.gp_len.setValue(1.2)
        self.gp_sig = QDoubleSpinBox(); self.gp_sig.setRange(1e-6, 10.0); self.gp_sig.setDecimals(4); self.gp_sig.setValue(0.15)
        self.gp_noise = QDoubleSpinBox(); self.gp_noise.setRange(1e-12, 1e-1); self.gp_noise.setDecimals(10); self.gp_noise.setValue(1e-4)

        cfg_form.addRow("max_steps", self.max_steps)
        cfg_form.addRow("n_init_random", self.n_init)
        cfg_form.addRow("bounds +/-", self.bounds_range)
        cfg_form.addRow("meas_sigma (scatter & errorbar)", self.meas_sigma)
        cfg_form.addRow("stop_mu_sigma", self.stop_mu_sigma)
        cfg_form.addRow("stop_modulation (>= to stop)", self.stop_modulation)
        cfg_form.addRow("GF weight: peak", self.gf_w_peak)
        cfg_form.addRow("GF weight: refine", self.gf_w_refine)
        cfg_form.addRow("seed", self.seed)
        cfg_form.addRow("n_bootstrap", self.n_boot)
        cfg_form.addRow("n_candidates", self.n_cand)
        cfg_form.addRow("UCB beta", self.ucb_beta)
        cfg_form.addRow("EI xi", self.ei_xi)
        cfg_form.addRow("ridge_fit", self.ridge_fit)
        cfg_form.addRow("GF probe_scale (~1σ)", self.probe_scale)
        cfg_form.addRow("GP length_scale", self.gp_len)
        cfg_form.addRow("GP signal_var", self.gp_sig)
        cfg_form.addRow("GP noise_var", self.gp_noise)

        self.sigma_group = QGroupBox("Initial σ guess (search heuristic only)")
        left.addWidget(self.sigma_group)
        self.sigma_form = QFormLayout(self.sigma_group)
        self.sigma_boxes: Dict[str, QDoubleSpinBox] = {}
        self._rebuild_sigma_inputs()

        cfg_row = QHBoxLayout()
        left.addLayout(cfg_row)
        self.save_cfg_btn = QPushButton("Save config")
        self.load_cfg_btn = QPushButton("Load config")
        cfg_row.addWidget(self.save_cfg_btn)
        cfg_row.addWidget(self.load_cfg_btn)
        cfg_row.addStretch(1)

        left.addStretch(1)

        # ---- Right: gallery ----
        gallery_group = QGroupBox("PNG gallery (saved after run)")
        middle.addWidget(gallery_group, stretch=3)
        g_layout = QVBoxLayout(gallery_group)

        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        g_layout.addWidget(self.gallery_scroll)

        self.gallery_widget = QWidget()
        self.gallery_grid = QGridLayout(self.gallery_widget)
        self.gallery_grid.setSpacing(8)
        self.gallery_scroll.setWidget(self.gallery_widget)
    
        self.interface = IPBSMInterface()


        # =========================
        # Bottom: result + log (always visible)
        # =========================
        bottom = QVBoxLayout()
        outer.addLayout(bottom)

        self.result_lbl = QLabel("Result: -")
        bottom.addWidget(self.result_lbl)

        self.log_lbl = QLabel("")
        self.log_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log_lbl.setMinimumHeight(28)
        bottom.addWidget(self.log_lbl)

        # =========================
        # Signals
        # =========================
        self.mode_box.currentTextChanged.connect(self._rebuild_sigma_inputs)
        self.method_box.currentTextChanged.connect(self._update_method_visibility)
        self._update_method_visibility()
        self.ctrl_box.currentTextChanged.connect(self._update_ctrl_buttons)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.get_btn.clicked.connect(self._on_get_ipbsm)
        self.save_cfg_btn.clicked.connect(self._on_save_config)
        self.load_cfg_btn.clicked.connect(self._on_load_config)

        self._update_ctrl_buttons()

    # ----------------------------
    # helpers
    # ----------------------------
    def _update_ctrl_buttons(self):
        self.get_btn.setEnabled(self.ctrl_box.currentText() == "machine")

    def _params(self) -> List[str]:
        return PARAMSETS[self.mode_box.currentText()]

    def _rebuild_sigma_inputs(self):
        while self.sigma_form.count():
            item = self.sigma_form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.sigma_boxes.clear()
        for p in self._params():
            box = QDoubleSpinBox()
            box.setRange(1e-6, 100.0)
            box.setDecimals(4)
            box.setValue(0.5)
            self.sigma_boxes[p] = box
            self.sigma_form.addRow(f"σ0[{p}]", box)

    def _set_form_row_visible(self, form: QFormLayout, field: QWidget, visible: bool) -> None:
        lab = form.labelForField(field)
        if lab is not None:
            lab.setVisible(visible)
        field.setVisible(visible)

    def _update_method_visibility(self):
        m = self.method_box.currentText().upper()

        # GF-only
        is_gf = (m == "GF")
        self._set_form_row_visible(self.cfg_form, self.stop_mu_sigma, is_gf)
        self._set_form_row_visible(self.cfg_form, self.probe_scale, is_gf)
        self._set_form_row_visible(self.cfg_form, self.gf_w_peak, is_gf)
        self._set_form_row_visible(self.cfg_form, self.gf_w_refine, is_gf)

        # BO/LBO-only
        is_bo = (m in ("BO", "LBO"))
        self.acq_box.setVisible(is_bo)
        # if acq label exists, hide it too
        if hasattr(self, "top_acq_lbl"):
            self.top_acq_lbl.setVisible(is_bo)

        self._set_form_row_visible(self.cfg_form, self.n_cand, is_bo)
        self._set_form_row_visible(self.cfg_form, self.ucb_beta, is_bo)
        self._set_form_row_visible(self.cfg_form, self.ei_xi, is_bo)
        self._set_form_row_visible(self.cfg_form, self.gp_len, is_bo)
        self._set_form_row_visible(self.cfg_form, self.gp_sig, is_bo)
        self._set_form_row_visible(self.cfg_form, self.gp_noise, is_bo)

    def _collect_config(self) -> OptimizerConfig:
        params = self._params()
        rng = float(self.bounds_range.value())
        bounds = {p: (-rng, +rng) for p in params}
        init_sigma = {p: float(self.sigma_boxes[p].value()) for p in params}

        expected_y_max = 0.8 if (self.ctrl_box.currentText() == "test") else None

        return OptimizerConfig(
            mode_name=self.mode_box.currentText(),
            method=self.method_box.currentText(),
            acquisition=self.acq_box.currentText(),
            params=params,
            bounds=bounds,
            init_sigma=init_sigma,
            meas_sigma=float(self.meas_sigma.value()),
            expected_y_max=expected_y_max,
            stop_modulation=float(self.stop_modulation.value()),
            knob_step=0.01,
            gf_weight_peak=float(self.gf_w_peak.value()),
            gf_weight_refine=float(self.gf_w_refine.value()),
            gf_jitter_frac=0.25,
            max_steps=int(self.max_steps.value()),
            stop_mu_sigma=float(self.stop_mu_sigma.value()),
            seed=int(self.seed.value()),
            n_init_random=int(self.n_init.value()),
            n_candidates=int(self.n_cand.value()),
            n_bootstrap=int(self.n_boot.value()),
            ridge_fit=float(self.ridge_fit.value()),
            gp_length_scale=float(self.gp_len.value()),
            gp_signal_var=float(self.gp_sig.value()),
            gp_noise_var=float(self.gp_noise.value()),
            ucb_beta=float(self.ucb_beta.value()),
            ei_xi=float(self.ei_xi.value()),
            probe_scale=float(self.probe_scale.value()),
        )

    def _set_config_to_ui(self, cfg: dict):
        self.mode_box.setCurrentText(cfg.get("mode_name", "linear"))
        self.method_box.setCurrentText(cfg.get("method", "GF"))
        self.acq_box.setCurrentText(cfg.get("acquisition", "UCB"))
        if "controller" in cfg:
            self.ctrl_box.setCurrentText(cfg["controller"])

        self.max_steps.setValue(int(cfg.get("max_steps", 60)))
        self.n_init.setValue(int(cfg.get("n_init_random", 8)))

        b = cfg.get("bounds", {})
        if b:
            any_p = list(b.keys())[0]
            self.bounds_range.setValue(float(abs(b[any_p][1])))

        self.meas_sigma.setValue(float(cfg.get("meas_sigma", 0.01)))
        self.stop_mu_sigma.setValue(float(cfg.get("stop_mu_sigma", 0.02)))
        self.seed.setValue(int(cfg.get("seed", 123)))
        self.n_boot.setValue(int(cfg.get("n_bootstrap", 60)))
        self.n_cand.setValue(int(cfg.get("n_candidates", 6000)))

        self.ucb_beta.setValue(float(cfg.get("ucb_beta", 2.0)))
        self.ei_xi.setValue(float(cfg.get("ei_xi", 0.0)))
        self.ridge_fit.setValue(float(cfg.get("ridge_fit", 1e-4)))
        self.probe_scale.setValue(float(cfg.get("probe_scale", 1.0)))
        self.gp_len.setValue(float(cfg.get("gp_length_scale", 1.2)))
        self.gp_sig.setValue(float(cfg.get("gp_signal_var", 0.15)))
        self.gp_noise.setValue(float(cfg.get("gp_noise_var", 1e-4)))

        self._rebuild_sigma_inputs()
        init_sigma = cfg.get("init_sigma", {})
        for p, v in init_sigma.items():
            if p in self.sigma_boxes:
                self.sigma_boxes[p].setValue(float(v))

    def _make_controller(self, cfg: OptimizerConfig):
        if self.ctrl_box.currentText() == "machine":
            from Opt_IPBSM import IPBSMInterface
            return IPBSMInterface()
        correlated = (cfg.mode_name != "linear")
        spec = make_random_spec(
            params=cfg.params,
            seed=cfg.seed,
            correlated=correlated,
            weak_corr=0.2,
            amp=0.8,
            meas_sigma=cfg.meas_sigma,
        )
        return SyntheticGaussianIPBSMController(spec=spec, seed=cfg.seed + 999)

    # ----------------------------
    # config save/load
    # ----------------------------
    def _on_save_config(self):
        cfg = self._collect_config()
        payload = asdict(cfg)
        payload["controller"] = self.ctrl_box.currentText()
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.status_lbl.setText(f"Status: config saved -> {path}")

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._set_config_to_ui(payload)
        self.status_lbl.setText(f"Status: config loaded -> {path}")

    # ----------------------------
    # actions
    # ----------------------------
    def _on_get_ipbsm(self):
        if self.ctrl_box.currentText() != "machine":
            return
        try:
            from Opt_IPBSM import IPBSMInterface
            ctrl = IPBSMInterface()
            y, yerr = ctrl.get_ipbsm()
            QMessageBox.information(self, "IPBSM", f"modulation={y:.6f}\nerr={yerr:.6f}")
        except Exception as e:
            QMessageBox.warning(self, "IPBSM", f"Failed: {e}")

    def _on_run(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Optimizer is running.")
            return

        cfg = self._collect_config()
        tag = now_tag()
        out_dir = Path("Data") / f"{tag}-{cfg.mode_name}-{cfg.method}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.last_out_dir = out_dir

        self.stop_flag = StopFlag()

        try:
            ctrl = self._make_controller(cfg)
        except Exception as e:
            QMessageBox.critical(self, "Controller error", str(e))
            return

        opt = Optimizer(controller=ctrl, config=cfg, out_dir=out_dir, stop_flag=self.stop_flag)

        self.worker = OptimizerWorker(opt)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText(f"Status: running -> {out_dir}")
        self.log_lbl.setText("")
        self.result_lbl.setText("Result: -")
        self._clear_gallery()

        self.worker.start()

    def _on_stop(self):
        self.stop_flag.request_stop()
        self.status_lbl.setText("Status: stop requested (after current point)")

    def _on_progress(self, payload: dict):
        step = payload.get("step", 0)
        info = payload.get("info", {})
        phase = info.get("phase", "")
        y = info.get("y", None)
        best = info.get("best_y", None)

        line = f"step={step} phase={phase}"
        if y is not None:
            line += f" y={float(y):.5f}"
        if best is not None:
            line += f" best={float(best):.5f}"
        if "chosen_by" in info:
            line += f" by={info['chosen_by']}"
        self.log_lbl.setText(line)

    def _on_failed(self, msg: str):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("Status: failed")
        QMessageBox.critical(self, "Failed", msg)

    def _on_finished(self, out: dict):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        out_dir = Path(out.get("out_dir", "")) if out.get("out_dir") else self.last_out_dir
        if out_dir is None:
            self.status_lbl.setText("Status: finished (no output dir)")
            return

        # Read measurements.csv
        X, y, yerr = [], [], []
        csv_path = out_dir / "measurements.csv"
        params = self._params()
        if csv_path.exists():
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r)
                for row in r:
                    X.append([float(row[2+i]) for i in range(len(params))])
                    y.append(float(row[2+len(params)]))
                    yerr.append(float(row[3+len(params)]))

        X = np.array(X, float) if X else np.zeros((0, len(params)), float)
        y = np.array(y, float) if y else np.zeros((0,), float)
        yerr = np.array(yerr, float) if yerr else np.zeros((0,), float)

        cfg = self._collect_config()
        mode_fit = "diag" if cfg.mode_name == "linear" else "full"
        fit = fit_gaussian_from_samples(X, y, mode=mode_fit, ridge=cfg.ridge_fit, y_cap=cfg.expected_y_max)
        # ---- load bootstrap info from result.json ----
        boot = None
        result_path = out_dir / "result.json"
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                res = json.load(f)

            # plot_results が期待する形式に合わせる
            if "boot_mu_std" in res:
                boot = {
                    "mu_std": res.get("boot_mu_std"),
                    "cov_diag_mean": [
                        res["fit_cov"][i][i] for i in range(len(res["fit_cov"]))
                    ] if "fit_cov" in res else None,
                    "cov_diag_std": [
                        2.0 * res.get("fit_sigma_err", 0.0)
                        * np.sqrt(res["fit_cov"][i][i])
                        for i in range(len(res["fit_cov"]))
                    ] if "fit_cov" in res else None,
                }

        saved = plot_results(
            cfg=cfg,
            out_dir=out_dir,
            X=X,
            y=y,
            yerr=yerr,
            fit=fit,     # ← さっき計算した fit
            boot=boot,   # ← result.json から読んだもの
        )
        self._populate_gallery([p for p in saved if str(p).lower().endswith(".png")])

        best_x = out.get("best_x", {})
        best_y = out.get("best_y", float("nan"))
        mu_mean = out.get("boot_mu_mean", None)
        mu_std = out.get("boot_mu_std", None)

        msg = f"finished: n={out.get('n_steps', '?')}  best_y={best_y:.5f}\n" \
              f"best_x={best_x}\n"
        if mu_mean and mu_std:
            msg += f"mu_est={mu_mean}\nmu_std={mu_std}\n"
        msg += f"saved -> {out_dir}"

        self.result_lbl.setText(msg)
        self.status_lbl.setText(f"Status: finished -> {out_dir}")

    # ----------------------------
    # gallery
    # ----------------------------
    def _clear_gallery(self):
        while self.gallery_grid.count():
            item = self.gallery_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _populate_gallery(self, png_paths: List[Path]):
        self._clear_gallery()
        if not png_paths:
            return

        png_paths = sorted(png_paths, key=lambda p: str(p))
        cols = 2  # bigger thumbnails
        r = 0
        c = 0
        for path in png_paths:
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            v = QVBoxLayout(frame)

            lbl = QLabel(path.name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(lbl)

            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(str(path))
            if not pix.isNull():
                pix = pix.scaledToWidth(520, Qt.TransformationMode.SmoothTransformation)
                img.setPixmap(pix)
            else:
                img.setText("(failed to load)")

            v.addWidget(img)
            self.gallery_grid.addWidget(frame, r, c)

            c += 1
            if c >= cols:
                c = 0
                r += 1

    def on_get_ipbsm_clicked(self):
        self.get_btn.setEnabled(False)
        self.log_lbl.setText("IPBSM: measuring...")

        self.w_ipbsm = GetIPBSMWorker(self.interface, timeout=600)
        self.w_ipbsm.done.connect(self._on_ipbsm_ok)
        self.w_ipbsm.fail.connect(self._on_ipbsm_fail)
        self.w_ipbsm.finished.connect(lambda: self.get_btn.setEnabled(True))
        self.w_ipbsm.start()

    def _on_ipbsm_ok(self, m, e):
        print(f"IPBSM modulation={m:.6f}  err={e:.6f}")
        self.log_lbl.setText(f"IPBSM: {m:.6f} ± {e:.6f}")

    def _on_ipbsm_fail(self, msg):
        print("IPBSM failed:", msg)
        self.log_lbl.setText(f"IPBSM failed: {msg}")


class GetIPBSMWorker(QThread):
    done = pyqtSignal(float, float)   # modulation, error
    fail = pyqtSignal(str)

    def __init__(self, iface: IPBSMInterface, timeout=600):
        super().__init__()
        self.iface = iface
        self.timeout = timeout

    def run(self):
        try:
            m, e = self.iface.get_ipbsm(timeout=self.timeout)
            self.done.emit(m, e)
        except Exception as ex:
            self.fail.emit(str(ex))


def main():
    app = QApplication([])
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
