from PyQt6 import uic
from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QListWidget, QMessageBox, QProgressDialog, QVBoxLayout, QPushButton, QDialog, QLabel)
import sys, time,os
from pathlib import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cwd = os.getcwd()
        self._cancel = False

        ui_path = os.path.join(os.path.dirname(__file__), "MainPanel.ui")
        uic.loadUi(ui_path, self)
        self.application_choice=None

        #self.sysid_interface_button.clicked.connect(self.handle_sysid_click)
        self.compute_matrix_button.clicked.connect(self.handle_compute_response_matrix_click)
        self.bba_interface_button.clicked.connect(self.handle_bba_click)
        self.emittance_interface_button.clicked.connect(self.handle_emittance_click)
        self.knobs_interface_button.clicked.connect(self.handle_knobs_click)
        self.sysid_interface_orbit_button.clicked.connect(self.handle_sysid_orbit_click)
        self.sysid_interface_dispersion_button.clicked.connect(self.handle_sysid_dispersion_click)
        self.sysid_interface_wakefield_button.clicked.connect(self.handle_sysid_wakefield_click)

        self._procs=[]
        self.setWindowTitle("Choose the application")

    def handling(self, app_name,cwd=None, args=None):
        path = os.path.join(os.path.dirname(__file__), app_name)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(cwd or self.cwd)

        env = QProcessEnvironment.systemEnvironment()
        proc.setProcessEnvironment(env)

        argv = [path] + (args or [])
        proc.start(sys.executable, argv)

        proc.readyReadStandardOutput.connect(
            lambda p=proc: print(bytes(p.readAllStandardOutput()).decode(errors="ignore"), end="")
        )
        self._procs.append(proc)

    def _latest_response_file(self) -> str | None:
        root = f"~/flight-simulator-data/"
        root = os.path.expanduser(os.path.expandvars(root))
        if not root.exists():
            return None
        files = []
        for d in root.iterdir():
            if d.is_dir() and any(d.glob("DATA*.pkl")):
                files.append((d.stat().st_mtime, d))
        if not files:
            return None
        return str(max(files, key=lambda t: t[0])[1])

    def handle_sysid_click(self):
        self.handling("SysID_GUI.py")

    def handle_bba_click(self):
        self.handling("BBA_GUI.py")

    def handle_emittance_click(self):
        pass
    def handle_knobs_click(self):
        pass
    def handle_sysid_orbit_click(self):
        pass
    def handle_sysid_dispersion_click(self):
        pass
    def handle_sysid_wakefield_click(self):
        pass

    def handle_compute_response_matrix_click(self):
        latest=self._latest_response_file()
        if not latest:
            QMessageBox.warning(self, "No data available", "No data available")

        self.handling("ComputeResponseMatrix_GUI.py", cwd=latest, args=[latest])

if __name__ == "__main__":
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
