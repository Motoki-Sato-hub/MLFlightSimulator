from State import State
from datetime import datetime
from functools import partial
from collections import deque
import numpy as np
import threading
import signal
import time
import sys
import os
from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QIcon, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget, QMessageBox, QProgressBar
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal, pyqtSlot
from enum import Enum
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Mode(Enum):
    Orbit = "Orbit Correction"
    Dispersion = "Changed energy"
    Wakefield = "Changed intensity"
    All = "All modes at once"

class Machine(Enum):
    ATF2_DR = "ATF2_DR"
    ATF2_EXT = "ATF2_Ext"
    ATF2_LINAC = "ATF2_Linac"
    ATF2_DR_RFT = "ATF2_DR_RFT"
    ATF2_EXT_RFT = "ATF2_Ext_RFT"

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None, title='', orbit=None):
        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

class Worker(QObject):
    plot_data = pyqtSignal(dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str)
    progress=pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, interface, state, correctors, bpms, hkicks,vkicks, max_osc_h, max_osc_v, max_curr_h, max_curr_v, Niter,output_dir):
        super().__init__()
        self.output_dir=output_dir
        self.interface = interface
        self.S = state
        self.correctors = correctors
        self.hcorrs = self.interface.get_hcorrectors_names()
        self.vcorrs = self.interface.get_vcorrectors_names()
        self.bpms = bpms
        self.hkicks = hkicks
        self.vkicks = vkicks
        self.max_osc_h = max_osc_h
        self.max_osc_v = max_osc_v
        self.max_curr_h = max_curr_h
        self.max_curr_v = max_curr_v
        self.Niter = Niter
        self.running = False
        self.progress_value=0

        if hasattr(self, "working_directory_dialog"):
            self.working_directory_dialog.clicked.connect(self._pick_and_load_data_dir)

    @pyqtSlot()
    def run(self):
        self.running = True

        total_steps=self.Niter*len(self.correctors)
        self.progress_value=0
        I = self.interface
        S = self.S
        vkicks = self.vkicks
        hkicks = self.hkicks

        def clamp(val, max_val):
            if max_val == 0.0:
                return val
            return max(-max_val, min(val, max_val))

        for iter in range(self.Niter):
            if self.running == False:
                break

            for icorr, corrector in enumerate(self.correctors):
                corr = S.get_correctors(corrector)
                if self.running == False:
                    break

                if corrector in S.get_hcorrectors_names():
                    max_curr=self.max_curr_h
                    kick=hkicks[icorr]
                elif corrector in S.get_vcorrectors_names():
                    max_curr=self.max_curr_v
                    kick=vkicks[icorr]

                if not self.running:
                    break

                corr_changed = False

                print(f"Corrector {corrector} '+' excitation...")
                filename_p=f'DATA_{corrector}_p{iter:04d}.pkl'
                if not os.path.isfile(filename_p):
                    print('corr[bds] =', corr['bdes'], ' also kick = ', kick) 
                    curr_p = corr['bdes'] + kick
                    if corrector in S.get_hcorrectors_names():
                        curr_p = clamp(curr_p, self.max_curr_h)
                    else:
                        curr_p = clamp(curr_p, self.max_curr_v)
                    self.interface.push(corrector, curr_p)
                    corr_changed = True

                    if not self.running:
                        break

                    S.pull(I)
                    S.save(filename=filename_p)
                else:
                    S.load(filename_p)
                Op = S.get_orbit(self.bpms)

                print(f"Corrector {corrector} '-' excitation...")
                filename_m=f'DATA_{corrector}_m{iter:04d}.pkl'
                if not os.path.isfile(filename_m):
                    curr_m = corr['bdes'] - kick
                    if corrector in S.get_hcorrectors_names():
                        curr_m = clamp(curr_m, self.max_curr_h)
                    else:
                        curr_m = clamp(curr_m, self.max_curr_v)
                    self.interface.push(corrector, curr_m)
                    corr_changed = True

                    if not self.running:
                        break

                    S.pull(I)
                    S.save(filename=f'DATA_{corrector}_m{iter:04d}.pkl')
                else:
                    S.load(filename_m)
                Om = S.get_orbit(self.bpms)

                if corr_changed:
                    self.interface.push(corrector, corr['bdes'])

                Diff_x = (Op['x'] - Om['x']) / 2.0
                Diff_y = (Op['y'] - Om['y']) / 2.0
                nsamples = Op['stdx'].size
                Err_x = np.sqrt(np.square(Op['stdx']) + np.square(Om['stdx'])) / np.sqrt(nsamples)
                Err_y = np.sqrt(np.square(Op['stdy']) + np.square(Om['stdy'])) / np.sqrt(nsamples)
                self.plot_data.emit(Op, Diff_x, Err_x, Diff_y, Err_y, corrector)

                self.progress_value=self.progress_value + 1
                percent = int(self.progress_value / total_steps * 100)
                self.progress.emit(percent)

                if corrector in S.get_hcorrectors_names():
                    Diff_x_clean = Diff_x[~np.isnan(Diff_x)]
                    if np.max(np.abs(Diff_x_clean)) != 0.0:
                        hkicks[icorr] *= self.max_osc_h / np.max(np.abs(Diff_x_clean))
                    hkicks[icorr] = 0.8 * hkicks[icorr] + 0.2 * kick
                else:
                    Diff_y_clean = Diff_y[~np.isnan(Diff_y)]
                    if np.max(np.abs(Diff_y_clean)) != 0.0:
                        vkicks[icorr] *= self.max_osc_v / np.max(np.abs(Diff_y_clean))
                    vkicks[icorr] = 0.8 * vkicks[icorr] + 0.2 * kick

                with open(os.path.join(self.output_dir,'kicks.txt'), 'w') as f:
                    for i, c in enumerate(self.correctors):
                        f.write(f'{c} {hkicks[i]} {vkicks[i]}\n')

                time.sleep(1)

        self.running = False
        self.finished.emit()

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __set_status_in_title(self, status):
        self.setWindowTitle("SYSID - " + self.interface.__class__.__name__ + " " + status)

    @pyqtSlot(int)
    def _update_progress(self,value):
        self.progressBar.setValue(value)

    def __init__(self, interface, dir_name):
        super().__init__()

        # SysID
        self.worker = None
        self.thread = None
        self._activate_mode=None

        self.cwd = os.getcwd()
        self.interface = interface
        bpms_list = interface.get_bpms()['names']
        correctors = self.interface.get_correctors()
        correctors_list = correctors['names']

        if correctors_list is not None:
            hcorrs = self.interface.get_hcorrectors_names()
            vcorrs = self.interface.get_vcorrectors_names()
            hcorr_indexes = np.array([index for index, string in enumerate(correctors_list) if string in hcorrs])
            vcorr_indexes = np.array([index for index, string in enumerate(correctors_list) if string in vcorrs])
            def clean_array(a):
                a = np.array([0 if x is None else x for x in a], dtype=float)
                a[np.isnan(a)] = 0
                return a
            max_curr_h = 1.15 * np.max(np.abs(clean_array(np.array(correctors['bdes'])[hcorr_indexes])))
            max_curr_v = 1.15 * np.max(np.abs(clean_array(np.array(correctors['bdes'])[vcorr_indexes])))

        # Load the interface
        uic.loadUi("SysID_GUI.ui", self)

        # Replace the placeholder with your real widget
        self.right_layout.removeWidget(self.plot_widget)
        self.plot_widget.deleteLater()
        self.plot_widget = MatplotlibWidget(self)
        self.right_layout.addWidget(self.plot_widget)

        # Setting up the interface
        self.save_correctors_button.clicked.connect(self.__save_correctors_button_clicked)
        self.load_correctors_button.clicked.connect(self.__load_correctors_button_clicked)
        self.clear_correctors_button.clicked.connect(self.__clear_correctors_button_clicked)
        self.save_bpms_button.clicked.connect(self.__save_bpms_button_clicked)
        self.load_bpms_button.clicked.connect(self.__load_bpms_button_clicked)
        self.clear_bpms_button.clicked.connect(self.__clear_bpms_button_clicked)
        self.start_button.clicked.connect(self.__start_button_clicked)
        self.stop_button.clicked.connect(self.__stop_button_clicked)

        self.choose_mode.setCurrentText(Mode.Orbit.value)
        self.mode=Mode.Orbit
        self.choose_mode.currentTextChanged.connect(self._choose_the_correction_mode)
        self.initial_hkick_settings.setText("0.01")
        self.initial_vkick_settings.setText("0.01")

        self.correctors_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.correctors_list.insertItems(0, correctors_list)
        self.bpms_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.bpms_list.insertItems(0, bpms_list)
        self.working_directory_input.setText(dir_name)
        self.max_horizontal_current_spinbox.setValue(max_curr_h)
        self.max_horizontal_current_spinbox.setSingleStep(0.01)
        self.max_vertical_current_spinbox.setValue(max_curr_v)
        self.max_vertical_current_spinbox.setSingleStep(0.01)
        self.horizontal_excursion_spinbox.setValue(0.5)
        self.horizontal_excursion_spinbox.setSingleStep(0.1)
        self.vertical_excursion_spinbox.setValue(0.5)
        self.vertical_excursion_spinbox.setSingleStep(0.1)

        if hasattr(self, "working_directory_dialog"):
            self.working_directory_dialog.clicked.connect(self._pick_and_load_data_dir)
        self.__set_status_in_title("[Idle]")
        interface_name=interface.get_name()
        machine = Machine(interface_name)

        self.modes_to_do=[]
        self.counter=0
        self.current_mode=None

    def _current_measuring_mode(self):
        if self.mode == Mode.All:
            self.modes_to_do=[Mode.Orbit,Mode.Dispersion,Mode.Wakefield]
        else:
            self.modes_to_do=[self.mode]
        self.counter=0

    def _start_next_mode(self):
        initial_hkick=self._read_initial_kicks()
        #selected_correctors = self.interface.get_correctors()['names']
        #kicks=initial_hkick*np.ones(len(self.selected_correctors),dtype=float)
        if self.counter>=len(self.modes_to_do):
            self.__set_status_in_title("[Idle]")
            self.progressBar.setValue(100)
            return
        mode=self.modes_to_do[self.counter]
        self.current_mode=mode
        print(f"Currently at mode: {mode.name}")
        self.__set_status_in_title(f"[Running {mode.name} mode]")
        self.S.load('machine_status')
        self.S.push(self.interface)

        if mode==Mode.Dispersion:
            self.interface.change_energy()
            print("Energy changed")
        elif mode==Mode.Wakefield:
            self.interface.change_intensity()
            print("Intensity changed")
        self.progressBar.setValue(0)

    def _read_all_parameters(self,text):
        text = text.strip()
        params = {}
        for p in text.split(","):
            p = p.strip()
            if not p:
                continue
            k,v = p.split("=",1)
            k = k.strip()
            v = v.strip()
            try:
                params[k] = float(v)
            except ValueError:
                raise ValueError(f"Not a number encountered in {p}")
        return params

    def _choose_the_correction_mode(self):
        data_mode=self.choose_mode.currentText()
        self.mode=Mode(data_mode)

    def __save_correctors_button_clicked(self):
        dir_name = self.working_directory_input.text()
        os.makedirs (dir_name, exist_ok=True)
        os.chdir (dir_name)
        selected_correctors = self.correctors_list.selectedItems()
        dir_name = self.working_directory_input.text() + '/correctors.txt'
        filename, _ = QFileDialog.getSaveFileName(None, "Save File", dir_name, "Text Files (*.txt)")
        if filename:
            with open(filename, 'w') as f:
                for item in selected_correctors:
                    f.write(f"{item.text()}\n")

    def __load_correctors_button_clicked(self):
        dir_name = self.working_directory_input.text() + '/correctors.txt'
        filename, _ = QFileDialog.getOpenFileName(None, "Open File", dir_name, "Text Files (*.txt)")
        if filename:
            with open(filename, 'r') as f:
                selected_correctors = [line.strip() for line in f]
        else:
            selected_correctors = self.interface.get_correctors()['names']

        self.correctors_list.clearSelection()
        for item in selected_correctors:
            items = self.correctors_list.findItems(item, Qt.MatchFlag.MatchExactly)
            for item in items:
                item.setSelected(True)

    def __clear_correctors_button_clicked(self):
        self.correctors_list.clearSelection()

    def __save_bpms_button_clicked(self):
        dir_name = self.working_directory_input.text()
        os.makedirs (dir_name, exist_ok=True)
        os.chdir (dir_name)
        selected_bpms = self.bpms_list.selectedItems()
        dir_name = self.working_directory_input.text() + '/bpms.txt'
        filename, _ = QFileDialog.getSaveFileName(None, "Save File", dir_name, "Text Files (*.txt)")
        if filename:
            with open(filename, 'w') as f:
                for item in selected_bpms:
                    f.write(f"{item.text()}\n")

    def __load_bpms_button_clicked(self):
        dir_name = self.working_directory_input.text() + '/bpms.txt'
        filename, _ = QFileDialog.getOpenFileName(None, "Open File", dir_name, "Text Files (*.txt)")
        if filename:
            with open(filename, 'r') as f:
                selected_bpms = [line.strip() for line in f]
        else:
            selected_bpms = self.interface.get_bpms()['names']

        self.bpms_list.clearSelection()
        for item in selected_bpms:
            items = self.bpms_list.findItems(item, Qt.MatchFlag.MatchExactly)
            for item in items:
                item.setSelected(True)

    def __clear_bpms_button_clicked(self):
        self.bpms_list.clearSelection()

    def _read_initial_kicks(self):
        text=self.initial_hkick_settings.text().strip()
        if not text:
            return 0.1
        try:
            return float(text)
        except ValueError as e:
            print(e)
            return 0.1

    def __start_button_clicked(self):
        dir_name = self.working_directory_input.text()
        os.makedirs (dir_name, exist_ok=True)
        os.chdir (dir_name)
        self.S = State(interface=self.interface)
        self.S.save(basename='machine_status')

        self.progressBar.setValue(0)
        if self.thread and self.thread.isRunning():
            return  # already running

        selected_correctors = [item.text() for item in self.correctors_list.selectedItems()]
        self.selected_correctors = selected_correctors
        if not selected_correctors:
            for i in range(self.correctors_list.count()):
                self.correctors_list.item(i).setSelected(True)
            selected_correctors = self.interface.get_correctors()['names']

        selected_bpms = [item.text() for item in self.bpms_list.selectedItems()]
        self.selected_bpms = selected_bpms
        if not selected_bpms:
            for i in range(self.bpms_list.count()):
                self.bpms_list.item(i).setSelected(True)
            selected_bpms = self.interface.get_bpms()['names']

        self._current_measuring_mode()
        self._start_next_mode()

        # kicks = 0.1 * np.ones(len(selected_correctors), dtype=float)
        initial_hkick=float(self.initial_hkick_settings.text())
        initial_vkick=float(self.initial_vkick_settings.text())
        hkicks=initial_hkick*np.ones(len(selected_correctors), dtype=float)
        vkicks=initial_vkick*np.ones(len(selected_correctors), dtype=float)

        max_osc_h = self.horizontal_excursion_spinbox.value()
        max_osc_v = self.vertical_excursion_spinbox.value()
        max_curr_h = self.max_horizontal_current_spinbox.value()
        max_curr_v = self.max_vertical_current_spinbox.value()
        Niter = int(self.niter_number.text())
        print(f"Niter: {Niter}")

        self.thread = QThread()
        self.worker = Worker(self.interface, self.S, selected_correctors, selected_bpms, hkicks,vkicks ,max_osc_h, max_osc_v, max_curr_h, max_curr_v, Niter,dir_name)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Cleanup after thread is done
        def clear_thread():
            try:
                if self.current_mode==Mode.Orbit:
                    print("Orbit mode active.")
                elif self.current_mode==Mode.Dispersion:
                    self.interface.reset_energy()
                elif self.current_mode==Mode.Wakefield:
                    self.interface.reset_intensity()
            except Exception as e:
                print(e)
            print("Restoring initial correctors' settings...")
            self.S.load('machine_status')
            self.S.push(self.interface)
            self.progressBar.setValue(100)
            self.thread = None
            self.worker = None
            self.counter+=1
            if self.counter< len(self.modes_to_do):
                self._start_next_mode()
                project_name = self.interface.get_name()
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_name = f"~/flight-simulator-data/{project_name}_{time_str}"
                dir_name = os.path.expanduser(os.path.expandvars(dir_name))
                os.makedirs(dir_name, exist_ok=True)
                self.working_directory_input.setText(dir_name)

                initial_hkick = self._read_initial_kicks()
                hkicks = initial_hkick * np.ones(len(selected_correctors), dtype=float)
                vkicks = initial_vkick * np.ones(len(selected_correctors), dtype=float)
                max_osc_h = self.horizontal_excursion_spinbox.value()
                max_osc_v = self.vertical_excursion_spinbox.value()
                max_curr_h = self.max_horizontal_current_spinbox.value()
                max_curr_v = self.max_vertical_current_spinbox.value()
                Niter = int(self.niter_number.text())
                print(f"Niter: {Niter}")
                self.thread = QThread()
                self.worker = Worker(self.interface, self.S, selected_correctors, selected_bpms, hkicks, vkicks,max_osc_h,max_osc_v, max_curr_h, max_curr_v, Niter,dir_name)
                self.worker.moveToThread(self.thread)

                self.thread.started.connect(self.worker.run)
                self.worker.finished.connect(self.thread.quit)
                self.worker.finished.connect(self.worker.deleteLater)
                self.thread.finished.connect(self.thread.deleteLater)
                self.thread.finished.connect(clear_thread)
                self.worker.plot_data.connect(self.__update_plot)
                self.worker.progress.connect(self._update_progress)
                self.thread.start()
            else:
                self.__set_status_in_title("[Idle]")

        self.thread.finished.connect(clear_thread)
        self.worker.plot_data.connect(self.__update_plot)
        self.worker.progress.connect(self._update_progress)

        self.thread.start()

    def __stop_button_clicked(self):
        if self.worker:
            self.__set_status_in_title("[Stopping...]")
            self.worker.stop()
            self.progressBar.setValue(0)
        self.__set_status_in_title("[Idle]")
        print('SysID stopped.')

    def __update_plot(self, Op, Diff_x, Err_x, Diff_y, Err_y, corrector):
        self.plot_widget.axes.clear()
        selected_bpms = [item.text() for item in self.bpms_list.selectedItems()]
        #nbpms=Op['nbpms']
        l_bpms=len(selected_bpms)
        scale=np.arange(l_bpms)
        #bpms_names=Op['names']
        self.plot_widget.axes.errorbar(scale, Diff_x, yerr=Err_x, lw=2, capsize=5, capthick=2, label="X")
        self.plot_widget.axes.errorbar(scale, Diff_y, yerr=Err_y, lw=2, capsize=5, capthick=2, label="Y")
        self.plot_widget.axes.legend(loc='upper left')
        self.plot_widget.axes.set_xticks(scale)
        self.plot_widget.axes.set_xticklabels(selected_bpms,rotation=90,fontsize=8)
        self.plot_widget.axes.set_ylabel('Orbit [mm]')
        self.plot_widget.axes.set_title(f"Corrector '{corrector}'")
        self.plot_widget.axes.grid(color='#EEEEEE')
        self.plot_widget.draw()
        self.plot_widget.repaint()

    def _pick_and_load_data_dir(self):
        default_dir = os.path.join(self.cwd, "Data")
        os.makedirs(default_dir, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, "Select data directory", default_dir)
        if not folder:
            return
        self.working_directory_input.setText(folder)


## MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)

    ## Select interface
    #from SelectInterface import InterfaceSelectionDialog
    import SelectInterface
    #dialog = InterfaceSelectionDialog()
    dialog = SelectInterface.choose_acc_and_interface()
    if dialog is None:
        print("Selection cancelled.")
        sys.exit(1)

    # if dialog.exec():
    #     print(f"Selected interface: {dialog.selected_interface_name}")
    #     I = dialog.selected_interface
    # else:
    #     print("Selection cancelled.")
    #     sys.exit(1)
    I=dialog
    project_name = I.get_name()
    print(f"Selected interface: {project_name}")

    ## Prepare project space
    #project_name = dialog.selected_interface_name
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"~/flight-simulator-data/{project_name}_{time_str}"
    dir_name = os.path.expanduser(os.path.expandvars(dir_name))

    ## Main Window
    window = MainWindow(interface=I, dir_name=dir_name)
    window.show()
    sys.exit(app.exec())
