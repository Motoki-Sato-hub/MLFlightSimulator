import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QVBoxLayout,QDialog, QLabel)
from PyQt6.QtWidgets import QSizePolicy
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ChiSquaredWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("χ² = w₁·O + w₂·D + w₃·W")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumSize(520, 320)
        self.resize(700, 420)
        self.setSizeGripEnabled(True)
        self.setSizeGripEnabled(True)

        self.w1 = self.w2 = self.w3 = 1.0

        self._O = []
        self._D = []
        self._W = []

        self._O0=None
        self._D0=None
        self._W0=None

        layout = QVBoxLayout(self)
        self.info = QLabel(f"w1={self.w1}, w2={self.w2}, w3={self.w3}")
        layout.addWidget(self.info,alignment=Qt.AlignmentFlag.AlignTop.AlignCenter)

        self.info.setVisible(False)
        self.info.setMaximumHeight(0)
        self._title=None

        self.fig = Figure(figsize=(5.0, 2.4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

        layout.setStretchFactor(self.info, 0)
        layout.setStretchFactor(self.canvas, 1)

        self._title = self.fig.suptitle(self.info.text(), y=0.98)
        self.fig.subplots_adjust(top=0.88)


        self.ax_orbit=self.fig.add_subplot(311)
        self.ax_d = self.fig.add_subplot(312)
        self.ax_w = self.fig.add_subplot(313)

        (self.line_O,) = self.ax_orbit.plot([], [], marker="o", label="O (orbit)")

        (self.line_D,) = self.ax_d.plot([], [], marker="o", label="D (dispersion)")
        (self.line_W,) = self.ax_w.plot([], [], marker="o", label="W (wakefield)")

        self.ax_orbit.set_xlabel("Iteration"), self.ax_d.set_xlabel("Iteration"), self.ax_w.set_xlabel("Iteration")
        self.ax_orbit.set_ylabel("Value"),self.ax_d.set_ylabel("Value"),self.ax_w.set_ylabel("Value")
        self.ax_orbit.grid(True, alpha=0.3),self.ax_d.grid(True, alpha=0.3), self.ax_w.grid(True, alpha=0.3)
        self.ax_orbit.legend(loc="best"),self.ax_d.legend(loc="best"), self.ax_w.legend(loc="best")

    def set_weights(self, w1, w2, w3):
        self.w1, self.w2, self.w3 = float(w1), float(w2), float(w3)
        txt=f"w1={self.w1:g}, w2={self.w2:g}, w3={self.w3:g}"
        self.info.setText(txt)
        if self._title is not None:
            self._title.set_text(txt)
            self.canvas.draw_idle()


    def clear(self):
        self._O.clear(); self._D.clear(); self._W.clear()
        self._O0=self._D0=self._W0=None
        self._set_lines([], [], [], [])

    def append_point(self, orbit_rms, disp_rms, wake_rms):

        O_beg=float(orbit_rms) if orbit_rms is not None else np.nan
        D_beg=float(disp_rms) if disp_rms is not None else np.nan
        W_beg=float(wake_rms) if wake_rms is not None else np.nan

        if self._O0 is None:
        #self._O0,self._D0,self._W0=O_beg,D_beg,W_beg
            eps=np.finfo(float).eps

            self._O0=O_beg if np.isfinite(O_beg) and O_beg !=0 else eps
            self._D0=D_beg if np.isfinite(D_beg) and D_beg !=0 else eps
            self._W0=W_beg if np.isfinite(W_beg) and W_beg !=0 else eps

        O = (O_beg/self._O0)**2 if np.isfinite(O_beg) else np.nan
        D = (D_beg/self._D0)**2 if np.isfinite(D_beg) else np.nan
        W = (W_beg/self._W0)**2 if np.isfinite(W_beg) else np.nan

        # O=O_beg
        # D=D_beg
        # W=W_beg
        self._O.append(O); self._D.append(D); self._W.append(W)
        self._redraw()

    def _redraw(self):
        x = list(range(1, len(self._O) + 1))
        self._set_lines(x, self._O, self._D, self._W)

    def _set_lines(self, x, O, D, W):
        self.line_O.set_data(x, O)
        self.line_D.set_data(x, D)
        self.line_W.set_data(x, W)

        self.ax_orbit.relim(),self.ax_d.relim(), self.ax_w.relim()
        self.ax_orbit.autoscale_view(),self.ax_d.autoscale_view(), self.ax_w.autoscale_view()
        self.canvas.draw_idle()

    def closeEvent(self, e):
        p = self.parent()
        if p is not None and hasattr(p, "_chi_dlg"):
            p._chi_dlg = None
        super().closeEvent(e)

    def calculating_chi(self, O, D, W):
        if not O:
            return

        if self._O0 is None:
            self._O0=next((value for value in O if value), 1)
            self._D0=next((value for value in D if value), 1)
            self._W0=next((value for value in W if value),1)

        self._O = [(value/self._O0)**2 for value in O]
        self._D = [(value/self._D0)**2 for value in D]
        self._W = [(value/self._W0)**2 for value in W]
        self._redraw()
