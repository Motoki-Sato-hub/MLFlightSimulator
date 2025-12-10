import sys, os, pickle, re, matplotlib, glob, time,json
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QSizePolicy, QMainWindow, QFileDialog, QListWidget, QMessageBox,QProgressDialog, QVBoxLayout, QPushButton, QDialog, QLabel)
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from Emitt_Meas_Simulation_ATF2 import Emitt_Meas_Simulation
import RF_Track as rft

class EmittMeasGUI(QMainWindow,Emitt_Meas_Simulation):
    def __init__(self):
        super().__init__()
        here = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(here, "EmittMeas_GUI.ui")
        uic.loadUi(ui_path, self)
        self.simulation_atf2 = Emitt_Meas_Simulation()
        self.screensListWidget.addItems(["OTR0X", "OTR1X", "OTR2X", "OTR3X"])
        self.cwd = os.getcwd()
        self.loadTwissButton.clicked.connect(self._pick_and_load_lattice_data)
        self.measureButton.clicked.connect(self._on_measure_click)
        self._hist_sigma_x = []
        self._hist_sigma_y = []
        self._hist_y_phase_space = []
        self._hist_y_phase_space = []
        self._otr_s=[] # position of otrs
        self._otr_names = []
        self._lattice_s=[]
        self._lattice_sigma_x=[]
        self._lattice_sigma_y=[]
        self.L=

        self._setup_canvases()
        #self.chosen_interface = chosen_interface

    def _loading_func(self, filename="", loading_name="Load file",*, use_dialog=True, base_dir=None):
        default_dir = base_dir or os.path.join(self.cwd, "Ext_ATF2")
        os.makedirs(default_dir, exist_ok=True)

        if use_dialog:
            start_path = os.path.join(default_dir, filename) if filename else default_dir
            fn, _ = QFileDialog.getOpenFileName(self,loading_name,start_path,"All files (*)")
            if not fn:
                return
        else:
            if not filename:
                return
            fn = filename
            if not os.path.isabs(fn):
                fn = os.path.join(default_dir, fn)

        if not os.path.isfile(fn):
            QMessageBox.warning(self, "Load data", f"File not found:\n{fn}")
            return

        with open(fn, "r") as f:
            selected = [ln.strip() for ln in f]

        QMessageBox.information(self, "Data file selected", f"Loaded:\n{fn}")
        self.twissFileLineEdit.setText(fn)

    def _pick_and_load_lattice_data(self):
        self._loading_func(loading_name="Load MAD-X file")

    def _on_measure_click(self):
        madx_file_path = self.twissFileLineEdit.text().strip() #deletes chars from beginning and the end of a string
        if not madx_file_path:
            QMessageBox.warning(self,"No data file selected", "No data file selected")
        madx_file_path=os.path.expanduser(madx_file_path)
        if not os.path.isfile(madx_file_path):
            QMessageBox.warning(self,"No data file", "No data file")
        self.simulation_atf2.filename= madx_file_path
        self._start_measuring()
        self._draw_plots()

    def _setup_canvases(self):
        if FigureCanvas is None:
            self.y_phase_space_canvas = self.sigma_y_canvas = self.x_phase_space_canvas = self.sigma_x_canvas = None
            return

        def install(host):
            fig = Figure(figsize=(5, 2.4), tight_layout=True)
            canvas = FigureCanvas(fig)
            layout = host.layout()
            if layout is None:
                from PyQt6.QtWidgets import QVBoxLayout
                layout = QVBoxLayout(host)
                layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)
            return fig, canvas

        self.y_phase_space_fig, self.y_phase_space_canvas = install(self.plotFrame11)
        self.sigma_y_fig, self.sigma_y_canvas = install(self.plotFrame12)
        self.x_phase_space_fig, self.x_phase_space_canvas = install(self.plotFrame21)
        self.sigma_x_fig, self.sigma_x_canvas = install(self.plotFrame22)

    def _plot_series(self, canvas, fig, x,y, title, ylabel,xlabel):
        if canvas is None:
            return
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(x,y, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel=xlabel, fontsize=8)
        ax.set_ylabel(ylabel=ylabel, fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.grid(True, alpha=0.3)

        highlight_filter=['OTR0X','OTR1X','OTR2X','OTR3X']
        highlight_positions=[]
        highlight_names=[]
        # Reconstruct lattice positions
        s = 0.0
        lattice = self.simulation_atf2.lattice
        start=lattice[-1].get_name()
        end=lattice[0].get_name()
        names = list(self.simulation_atf2.lattice.keys())
        start_index = names.index(start)
        end_index = names.index(end) + 1
        names_in_lattice = names[start_index:end_index]

        for name in names_in_lattice:
            elem = element_descriptions[name]
            L = elem['L']
            elem_type = elem['element_type']

            if elem_type == 'Screen' and name in highlight_filter:
                # Compute center of element (if L > 0), or use current s (if L == 0)
                center = s + L / 2 if L > 0 else s
                highlight_positions.append(center)
                highlight_names.append(name)

            s += L

        print("Screens in lattice:")
        for name in names_in_lattice:
            elem = element_descriptions[name]
            if elem['element_type'] == 'Screen':
                print(name, "->", name.split('.')[-1], "at", elem['s_start'])

        # Plot the screen markers
        # error propagation
        # monte carlo
        y_min,y_max=ax.get_ylim()
        for name, pos in zip(highlight_names, highlight_positions):
            ax.axvline(x=pos, color='k', linestyle='--', alpha=0.3)
            ax.text(pos, 0, name, rotation=90, verticalalignment='bottom', fontsize=8)
        canvas.draw_idle()

    def _plot_phase_space(self, fig, plane_for_M, ylabel, xlabel):
        emitt_meas_simulation = self.simulation_atf2
        chosen_otr = self.screensListWidget.currentItem().text()
        B = emitt_meas_simulation.get_bunch_at_otr(chosen_otr)
        fig.clear()
        ax = fig.add_subplot(111)
        M = B.get_phase_space(plane_for_M)
        ax.plot(M[:, 0], M[:, 1], '.', markersize=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plane_for_M =='%x %xp':
            ax.set_title(f"Horizontal phase space at {chosen_otr}")
            self.x_phase_space_canvas.draw_idle()
        else:
            ax.set_title(f"Vertical phase space at {chosen_otr}")
            self.y_phase_space_canvas.draw_idle()
        ax.grid(True)

    def _start_measuring(self):
        #if self.chosen_interface == "ATF2":
        emitt_meas_simulation = self.simulation_atf2
        filename = emitt_meas_simulation.filename
        lattice = self.simulation_atf2.lattice
        B0 = emitt_meas_simulation.setup_beam0()
        lattice.track(B0)
        T = lattice.get_transport_table('%S %sigma_x %sigma_y')
        self._lattice_s=list(T[:,0])
        self._lattice_sigma_x=list(T[:,1])
        self._lattice_sigma_y=list(T[:,2])

        entrance_name, entrance, otrs = emitt_meas_simulation.get_data_from_twiss_file()
        Mx, My = emitt_meas_simulation.compute_transport_matrix()
        Sigma_xy_beam = emitt_meas_simulation.compute_beam_matrix()
        sigma_x_i, sigma_y_i, B_screens, inserted_screens = emitt_meas_simulation.measure_sigmas(return_bunches=True)
        emittance_x, emittance_y, beta_x, beta_y, alpha_x, alpha_y, gamma_x, gamma_y = emitt_meas_simulation.solve_least_squares(Mx, My, sigma_x_i, sigma_y_i)

        self._hist_sigma_x = list(sigma_x_i)
        self._hist_sigma_y = list(sigma_y_i)
        self._otr_names = ['OTR0X', 'OTR1X', 'OTR2X', 'OTR3X']

        mass = rft.electronmass
        Pref = emitt_meas_simulation.Pref
        beta_gamma = Pref / mass
        print(f"Beta gamma is: {beta_gamma}")
        print(f"emittance_x (normalised) = {emittance_x} mm mrad")
        print(f"emittance_y (normalised) = {emittance_y} mm mrad")

    def _draw_plots(self):
        self._plot_series(canvas=self.sigma_y_canvas, fig=self.sigma_y_fig, x=self._lattice_s,y=self._lattice_sigma_y, title="Vertical Beam Size", ylabel=r"$\sigma_y$ [mm]",xlabel="S [m]")
        self._plot_series(canvas=self.sigma_x_canvas, fig= self.sigma_x_fig, x=self._lattice_s,y=self._lattice_sigma_x, title="Horizontal Beam Size", ylabel=r"$\sigma_y$ [mm]",xlabel="S [m]")
        self._plot_phase_space(plane_for_M='%x %xp',xlabel = "x [mm]", ylabel = "x' [mrad]",fig = self.x_phase_space_fig)
        self._plot_phase_space(plane_for_M='%y %yp',xlabel = "y [mm]", ylabel = "y' [mrad]",fig = self.y_phase_space_fig)

    # def _clear_graphs(self):
    #     self._cancel = True
    #     self._hist_orbit.clear()
    #     self._hist_disp.clear()
    #     self._hist_wake.clear()
    #     self._plot_series(self.traj_canvas, self.traj_fig, [], None, "[mm]")
    #     self._plot_series(self.disp_canvas, self.disp_fig, [], None, "[mm]")
    #     self._plot_series(self.wake_canvas, self.wake_fig, [], None, "[mm]")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = EmittMeasGUI()
    w.show()
    sys.exit(app.exec())
