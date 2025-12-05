from State import State
from Response import Response
from datetime import datetime

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import numpy as np
import glob
import sys
import os

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QAbstractItemView, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal

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

class MainWindow(QMainWindow):
    def __init__(self):
        
        super().__init__()

        # Use glob to get the list of DATA files
        self.cwd = os.getcwd()
        self.datafiles = glob.glob('DATA*.pkl')

        # Prepare for computation
        S = State (filename=self.datafiles[0])

        # Init
        self.sequence = S.get_sequence()
        self.correctors = S.get_correctors()['names']
        self.bpms = S.get_bpms()['names']

        self.setWindowTitle("Compute Response Matrix Tool")
        self.setGeometry(100, 100, 600, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Left side layout
        left_layout = QVBoxLayout()
        top_layout.addLayout(left_layout,1)

        # Correctors list
        correctors_layout = QHBoxLayout()
        left_layout.addLayout(correctors_layout)

        correctors_label = QLabel("Correctors:")
        correctors_layout.addWidget(correctors_label)

        self.correctors_list = QListWidget()
        self.correctors_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.correctors_list.insertItems(0, self.correctors)
        left_layout.addWidget(self.correctors_list)

        # Save / Load / Clear correctors buttons
        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout)

        self.save_correctors_button = QPushButton("Save As..")
        self.save_correctors_button.clicked.connect(self.__save_correctors_button_clicked)
        button_layout.addWidget(self.save_correctors_button)

        self.load_correctors_button = QPushButton("Load..")
        self.load_correctors_button.clicked.connect(self.__load_correctors_button_clicked)
        button_layout.addWidget(self.load_correctors_button)

        self.clear_correctors_button = QPushButton("Clear")
        self.clear_correctors_button.clicked.connect(self.__clear_correctors_button_clicked)
        button_layout.addWidget(self.clear_correctors_button)

        # Middle layout
        middle_layout = QVBoxLayout()
        top_layout.addLayout(middle_layout)

        # BPMs list
        bpms_layout = QHBoxLayout()
        left_layout.addLayout(bpms_layout)

        bpms_label = QLabel("BPMs:")
        bpms_layout.addWidget(bpms_label)

        self.bpms_list = QListWidget()
        self.bpms_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.bpms_list.insertItems(0, self.bpms)
        left_layout.addWidget(self.bpms_list)

        # Save / Load / Clear bpms buttons
        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout)

        self.save_bpms_button = QPushButton("Save As..")
        self.save_bpms_button.clicked.connect(self.__save_bpms_button_clicked)
        button_layout.addWidget(self.save_bpms_button)

        self.load_bpms_button = QPushButton("Load..")
        self.load_bpms_button.clicked.connect(self.__load_bpms_button_clicked)
        button_layout.addWidget(self.load_bpms_button)

        self.clear_bpms_button = QPushButton("Clear")
        self.clear_bpms_button.clicked.connect(self.__clear_bpms_button_clicked)
        button_layout.addWidget(self.clear_bpms_button)

        # Right side layout
        right_layout = QVBoxLayout()
        top_layout.addLayout(right_layout,2)

        # Force Triangular Matrix CheckBox
        triangular_checkbox_layout = QHBoxLayout()
        right_layout.addLayout(triangular_checkbox_layout)
        
        self.triangular_checkbox = QCheckBox("Force triangular matrix")
        self.triangular_checkbox.setChecked(False)
        triangular_checkbox_layout.addStretch()
        triangular_checkbox_layout.addWidget(self.triangular_checkbox)
        triangular_checkbox_layout.addStretch()
        
        # Plot
        self.plot = MatplotlibWidget(self)
        right_layout.addWidget(self.plot)

        # Compute response matrix
        
        operation_layout = QHBoxLayout()
        right_layout.addLayout(operation_layout)
        
        self.compute_button = QPushButton("Compute")
        self.compute_button.setStyleSheet("background-color: green; color: white;")
        self.compute_button.clicked.connect(self.__compute_button_clicked)
        operation_layout.addWidget(self.compute_button)

        self.save_as_button = QPushButton("Save As..")
        self.save_as_button.setStyleSheet("background-color: red; color: white;")
        self.save_as_button.clicked.connect(self.__save_as_button_clicked)
        operation_layout.addWidget(self.save_as_button)

    def __save_correctors_button_clicked(self):
        dir_name = self.cwd
        os.makedirs (dir_name, exist_ok=True)
        os.chdir (dir_name)
        selected_correctors = self.correctors_list.selectedItems()
        dir_name = self.cwd + '/correctors.txt'
        filename, _ = QFileDialog.getSaveFileName(None, "Save File", dir_name, "Text Files (*.txt)")
        if filename:
            with open(filename, 'w') as f:
                for item in selected_correctors:
                    f.write(f"{item.text()}\n")

    def __load_correctors_button_clicked(self):
        dir_name = self.cwd + '/correctors.txt'
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
        dir_name = self.cwd
        os.makedirs (dir_name, exist_ok=True)
        os.chdir (dir_name)
        selected_bpms = self.bpms_list.selectedItems()
        dir_name = self.cwd + '/bpms.txt'
        filename, _ = QFileDialog.getSaveFileName(None, "Save File", dir_name, "Text Files (*.txt)")
        if filename:
            with open(filename, 'w') as f:
                for item in selected_bpms:
                    f.write(f"{item.text()}\n")

    def __load_bpms_button_clicked(self):
        dir_name = self.cwd + '/bpms.txt'
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

    def __compute_button_clicked(self):
        
        S = State (filename=self.datafiles[0])

        correctors = [ item.text() for item in self.correctors_list.selectedItems() ]
        bpms = [ item.text() for item in self.bpms_list.selectedItems() ]
        
        if not correctors:
            for i in range(self.correctors_list.count()):
                self.correctors_list.item(i).setSelected(True)
            correctors = self.correctors

        if not bpms:
            for i in range(self.bpms_list.count()):
                self.bpms_list.item(i).setSelected(True)
            bpms = self.bpms


        hcorrs = [string for string in correctors if (string.lower().startswith('zh') or string.lower().startswith('zx'))]
        vcorrs = [string for string in correctors if string.lower().startswith('zv')]

        # Pick all correctors preceding the last bpm
        hcorrs = [ corr for corr in hcorrs if self.sequence.index(corr) < self.sequence.index(bpms[-1]) ]
        vcorrs = [ corr for corr in vcorrs if self.sequence.index(corr) < self.sequence.index(bpms[-1]) ]

        # Pick all bpms following the first corrector
        bpms = [ bpm for bpm in bpms if self.sequence.index(bpm) > self.sequence.index(hcorrs[0]) ]
        bpms = [ bpm for bpm in bpms if self.sequence.index(bpm) > self.sequence.index(vcorrs[0]) ]

        # Read all orbits
        Bx = np.empty((0,len(bpms)))
        By = np.empty((0,len(bpms)))
        Cx = np.empty((0,len(hcorrs)))
        Cy = np.empty((0,len(vcorrs)))
        datafiles_p = [f for f in self.datafiles if f[-9] == 'p']
        for datafile_p in datafiles_p:
            datafile_m = datafile_p[:-9] + 'm' + datafile_p[-8:]
            if os.path.exists(datafile_m):
                Sp = State(filename=datafile_p)
                Sm = State(filename=datafile_m)
                Op = Sp.get_orbit (bpms)
                Om = Sm.get_orbit (bpms)
                Cx_p = Sp.get_correctors(hcorrs)['bact']
                Cy_p = Sp.get_correctors(vcorrs)['bact']
                Cx_m = Sm.get_correctors(hcorrs)['bact']
                Cy_m = Sm.get_correctors(vcorrs)['bact']
                if 0:
                    O_x = Op['x'] - Om['x']
                    O_y = Op['y'] - Om['y']
                    C_x = Cx_p - Cx_m
                    C_y = Cy_p - Cy_m
                    Bx = np.vstack((Bx, O_x))
                    By = np.vstack((By, O_y))
                    Cx = np.vstack((Cx, C_x))
                    Cy = np.vstack((Cy, C_y))
                    print(Cx, Bx)
                else:
                    Bx = np.vstack((Bx, Op['x']))
                    Bx = np.vstack((Bx, Om['x']))
                    By = np.vstack((By, Op['y']))
                    By = np.vstack((By, Om['y']))
                    Cx = np.vstack((Cx, Cx_p))
                    Cx = np.vstack((Cx, Cx_m))
                    Cy = np.vstack((Cy, Cy_p))
                    Cy = np.vstack((Cy, Cy_m))
            else:
                print(f"Data file '{datafile_m}' does not exist, ignoring counterpart '{datafile_p}' for response matrix computation")

        # Compute the response matrices
        ones_column_x = np.ones((Cx.shape[0], 1))
        ones_column_y = np.ones((Cy.shape[0], 1))

        # Add the column of ones to the matrix
        Cx = np.hstack((Cx, ones_column_x))
        Cy = np.hstack((Cy, ones_column_y))

        Rxx = np.transpose(np.linalg.lstsq(Cx, Bx, rcond=None)[0])
        Rxy = np.transpose(np.linalg.lstsq(Cy, Bx, rcond=None)[0])
        Ryx = np.transpose(np.linalg.lstsq(Cx, By, rcond=None)[0])
        Ryy = np.transpose(np.linalg.lstsq(Cy, By, rcond=None)[0])

        # Reference trajectory
        '''
        Bx = Rxx[:,-1]
        By = Ryy[:,-1]
        '''

        Bx = np.mean(Bx,axis=0).reshape(-1,1)
        By = np.mean(By,axis=0).reshape(-1,1)

        # Response matrices
        Rxx = Rxx[:,:-1]
        Rxy = Rxy[:,:-1]
        Ryx = Ryx[:,:-1]
        Ryy = Ryy[:,:-1]

        # Zero the response of all bpms preceeding the correctors
        if self.triangular_checkbox.isChecked():
            for corr in hcorrs:
                 bpm_indexes = [ bpms.index(bpm) for bpm in bpms if self.sequence.index(bpm) < self.sequence.index(corr) ]
                 Rxx[bpm_indexes, hcorrs.index(corr)] = 0
                 Ryx[bpm_indexes, hcorrs.index(corr)] = 0

            for corr in vcorrs:
                 bpm_indexes = [ bpms.index(bpm) for bpm in bpms if self.sequence.index(bpm) < self.sequence.index(corr) ]
                 Rxy[bpm_indexes, vcorrs.index(corr)] = 0
                 Ryy[bpm_indexes, vcorrs.index(corr)] = 0

        # Save on disk
        R = Response()
        R.bpms = bpms
        R.hcorrs = hcorrs
        R.vcorrs = vcorrs
        R.Rxx = Rxx
        R.Rxy = Rxy
        R.Ryx = Ryx
        R.Ryy = Ryy
        R.Bx = Bx
        R.By = By
        self.R = R

        # Clear existing figure
        self.plot.figure.clf()

        # === 3D surface plots ===
        # If you want to show both sets (2D + 3D), you need two canvases.
        # But if you want to reuse the same canvas, just do this after the first draw:

        # Clear and re-plot the 3D figure
        self.plot.figure.clf()
        fig1 = self.plot.figure
        # fig1.set_size_inches(12, 10)

        ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
        ax3 = fig1.add_subplot(2, 2, 3, projection='3d')
        ax2 = fig1.add_subplot(2, 2, 2, projection='3d')
        ax4 = fig1.add_subplot(2, 2, 4, projection='3d')

        x = np.array(range(len(hcorrs)))
        y = np.array(range(len(bpms)))
        X, Y = np.meshgrid(x, y)

        ax1.plot_surface(X, Y, Rxx, cmap='viridis')
        ax1.set_title('$R_{xx}$')
        ax1.set_xlabel('Corrector [#]')
        ax1.set_ylabel('BPM [#]')

        ax3.plot_surface(X, Y, Ryx, cmap='viridis')
        ax3.set_title('$R_{yx}$')
        ax3.set_xlabel('Corrector [#]')
        ax3.set_ylabel('BPM [#]')

        x = np.array(range(len(vcorrs)))
        X, Y = np.meshgrid(x, y)

        ax2.plot_surface(X, Y, Rxy, cmap='viridis')
        ax2.set_title('$R_{xy}$')
        ax2.set_xlabel('Corrector [#]')
        ax2.set_ylabel('BPM [#]')

        ax4.plot_surface(X, Y, Ryy, cmap='viridis')
        ax4.set_title('$R_{yy}$')
        ax4.set_xlabel('Corrector [#]')
        ax4.set_ylabel('BPM [#]')

        fig1.tight_layout()

        self.plot.draw()
        self.plot.flush_events()
        self.plot.repaint()
        
    def __save_as_button_clicked(self):
        dir_name = self.cwd + '/response2.pkl'
        os.chdir (self.cwd)
        filename, _ = QFileDialog.getSaveFileName(None, "Save Response Matrix", dir_name, "Piclke Files (*.pkl)")
        if filename:
            self.R.save('response2.pkl')

## MAIN
app = QApplication(sys.argv)

## Inspect directory

## Main Window
window = MainWindow()
window.show()
sys.exit(app.exec())
