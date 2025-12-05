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
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget,QMessageBox
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal, pyqtSlot

import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None, title='', orbit=None):
        fig = Figure(tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)


class Worker(QObject):
    plot_data = pyqtSignal(dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str)
    finished = pyqtSignal()

    def __init__(self, interface, state, correctors, bpms, kicks, max_osc_h, max_osc_v, max_curr_h, max_curr_v, Niter):
        super().__init__()
        self.interface = interface
        self.S = state
        self.correctors = correctors
        self.bpms = bpms
        self.kicks = kicks
        self.max_osc_h = max_osc_h
        self.max_osc_v = max_osc_v
        self.max_curr_h = max_curr_h
        self.max_curr_v = max_curr_v
        self.Niter = Niter
        self.running = False

        #self.working_directory_dialog.clicked.connect()

        if hasattr(self, "working_directory_dialog"):
            self.working_directory_dialog.clicked.connect(self._pick_and_load_data_dir)

        # FOR THE BBA_GUI!!
        self.cond="nominal"
        self.scale_E=0.98
        self.scale_I=0.90

    @pyqtSlot()
    def run(self):
        self.running = True

        # FOR THE BBA_GUI!!

        if self.cond == "scale_E":
            self.interface.change_energy(self.scale_E)
        elif self.cond == "scale_I":
            self.interface.change_intensity(self.scale_I)

        I = self.interface
        S = self.S
        kicks = self.kicks

        def clamp(val, max_val):
            if max_val == 0.0:
                return val
            return max(-max_val, min(val, max_val))

        for iter in range(self.Niter):
            if self.running == False:
                break
            for icorr, corrector in enumerate(self.correctors):
                if self.running == False:
                    break

                corr = S.get_correctors(corrector)
                kick = kicks[icorr]

                print(f"Corrector {corrector} '+' excitation...")
                curr_p = corr['bdes'] + kick
                if corrector in S.get_hcorrectors_names():
                    curr_p = clamp(curr_p, self.max_curr_h)
                else:
                    curr_p = clamp(curr_p, self.max_curr_v)
                I.push(corrector, curr_p)
                S.pull(I)
                S.save(filename=f'DATA_{corrector}_p{iter:04d}.pkl')
                Op = S.get_orbit(self.bpms)

                print(f"Corrector {corrector} '-' excitation...")
                curr_m = corr['bdes'] - kick
                if corrector in S.get_hcorrectors_names():
                    curr_m = clamp(curr_m, self.max_curr_h)
                else:
                    curr_m = clamp(curr_m, self.max_curr_v)
                I.push(corrector, curr_m)
                S.pull(I)
                S.save(filename=f'DATA_{corrector}_m{iter:04d}.pkl')
                Om = S.get_orbit(self.bpms)

                I.push(corrector, corr['bdes'])

                Diff_x = (Op['x'] - Om['x']) / 2.0
                Diff_y = (Op['y'] - Om['y']) / 2.0
                nsamples = Op['stdx'].size
                Err_x = np.sqrt(np.square(Op['stdx']) + np.square(Om['stdx'])) / np.sqrt(nsamples)
                Err_y = np.sqrt(np.square(Op['stdy']) + np.square(Om['stdy'])) / np.sqrt(nsamples)
                self.plot_data.emit(Op, Diff_x, Err_x, Diff_y, Err_y, corrector)

                if corrector in S.get_hcorrectors_names():
                    Diff_x_clean = Diff_x[~np.isnan(Diff_x)]
                    if np.max(np.abs(Diff_x_clean)) != 0.0:
                        kicks[icorr] *= self.max_osc_h / np.max(np.abs(Diff_x_clean))
                else:
                    Diff_y_clean = Diff_y[~np.isnan(Diff_y)]
                    if np.max(np.abs(Diff_y_clean)) != 0.0:
                        kicks[icorr] *= self.max_osc_v / np.max(np.abs(Diff_y_clean))

                kicks[icorr] = 0.8 * kicks[icorr] + 0.2 * kick

                with open('kicks.txt', 'w') as f:
                    for c, k in zip(self.correctors, kicks):
                        f.write(f'{c} {k}\n')

                time.sleep(1)

#FOR THE BBA_GUI!!
        if self.cond=="scale_E":
            self.interface.reset_energy()
        elif self.cond=="scale_I":
            self.interface.reset_intensity()

        self.finished.emit()


    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __set_status_in_title(self, status):
        self.setWindowTitle("SYSID - " + self.interface.__class__.__name__ + " " + status)
    
    def __init__(self, interface, dir_name):
        super().__init__()

        # SysID
        self.worker = None
        self.thread = None

        self.cwd = os.getcwd()
        self.interface = interface
        bpms_list = interface.get_bpms()['names']
        correctors = I.get_correctors()
        correctors_list = correctors['names']

        if correctors_list is not None:
            hcorrs = I.get_hcorrectors_names()
            vcorrs = I.get_vcorrectors_names()
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

    def __start_button_clicked(self):
        if self.thread and self.thread.isRunning():
            return  # already running

        self.__set_status_in_title("[Running...]")

        dir_name = self.working_directory_input.text()
        os.makedirs(dir_name, exist_ok=True)
        os.chdir(dir_name)

        selected_correctors = [item.text() for item in self.correctors_list.selectedItems()]
        if not selected_correctors:
            for i in range(self.correctors_list.count()):
                self.correctors_list.item(i).setSelected(True)
            selected_correctors = self.interface.get_correctors()['names']

        selected_bpms = [item.text() for item in self.bpms_list.selectedItems()]
        if not selected_bpms:
            for i in range(self.bpms_list.count()):
                self.bpms_list.item(i).setSelected(True)
            selected_bpms = self.interface.get_bpms()['names']

        self.S = State(interface=self.interface)
        self.S.save(basename='machine_status')

        kicks = 0.1 * np.ones(len(selected_correctors), dtype=float)
        max_osc_h = self.horizontal_excursion_spinbox.value()
        max_osc_v = self.vertical_excursion_spinbox.value()
        max_curr_h = self.max_horizontal_current_spinbox.value()
        max_curr_v = self.max_vertical_current_spinbox.value()
        Niter = 3

        self.thread = QThread()
        self.worker = Worker(self.interface, self.S, selected_correctors, selected_bpms, kicks, max_osc_h, max_osc_v, max_curr_h, max_curr_v, Niter)
        self.worker.moveToThread(self.thread)

        #FOR THE BBA_GUI!
        self.worker.cond="nominal"
        #self.worker.scale_E=0.98
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Cleanup after thread is done
        def clear_thread():
            self.thread = None
            self.worker = None

        self.thread.finished.connect(clear_thread)
        self.worker.plot_data.connect(self.__update_plot)

        self.thread.start()

    def __stop_button_clicked(self):
        if self.worker:
            self.__set_status_in_title("[Stopping...]")
            self.worker.stop()
        self.__set_status_in_title("[Idle]")
        print('SysID stopped.')
        print("Restoring initial correctors' settings...")
        selected_correctors = [item.text() for item in self.correctors_list.selectedItems()]
        self.interface.push(selected_correctors, self.S.get_correctors(selected_correctors)['bdes'])
        print("Restored initial correctors' settings.")


    def __update_plot(self, Op, Diff_x, Err_x, Diff_y, Err_y, corrector):
        self.plot_widget.axes.clear()
        self.plot_widget.axes.errorbar(range(Op['nbpms']), Diff_x, yerr=Err_x, lw=2, capsize=5, capthick=2, label="X")
        self.plot_widget.axes.errorbar(range(Op['nbpms']), Diff_y, yerr=Err_y, lw=2, capsize=5, capthick=2, label="Y")
        self.plot_widget.axes.legend(loc='upper left')
        self.plot_widget.axes.set_xlabel('Bpm [#]')
        self.plot_widget.axes.set_ylabel('Orbit [mm]')
        self.plot_widget.axes.set_title(f"Corrector '{corrector}'")
        self.plot_widget.axes.grid()
        self.plot_widget.draw()
        self.plot_widget.repaint()

    def _pick_and_load_data_dir(self):
        default_dir = os.path.join(self.cwd, "Data")
        os.makedirs(default_dir, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, "Select data directory", default_dir)
        if not folder:
            return
        self.working_directory_input.setText(folder)
        #QMessageBox.information(self.working_directory_dialog, "Data directory selected", self.working_directory_dialog)

## MAIN
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
project_name=I.get_name()
print(f"Selected interface: {project_name}")

## Prepare project space
#project_name = dialog.selected_interface_name
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
dir_name = f"~/flight-simulator-data/{project_name}_{time_str}"
dir_name = os.path.expanduser(os.path.expandvars(dir_name))

## Main Window
window = MainWindow(I, dir_name)
window.show()
sys.exit(app.exec())
