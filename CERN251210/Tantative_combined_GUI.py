# UnifiedGUI.py
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QSplitter
from PyQt6.QtCore import Qt
from datetime import datetime

from SysID_GUI import MainWindow as SysIDWindow
from BBA_GUI import MainWindow as BBAWindow
from SysGUI import MainWindow as SysGUISource


class EmittingStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        if text.strip():
            self.text_edit.append(text)

    def flush(self):
        pass

class UnifiedGUI(QMainWindow):

    def __init__(self, interface, dir_name):
        print("UnifiedGUI __init__ entered")
        super().__init__()

        self.interface = interface
        self.dir_name = dir_name

        self.setWindowTitle("Unified GUI")
        try:
            self.setWindowIcon(QIcon("SysID_GUI/CERN_logo.png"))
        except Exception:
            pass

        central = QWidget()
        layout = QVBoxLayout(central)

        self.splitter = QSplitter(Qt.Orientation.Vertical)

        self.tabs = QTabWidget()
        self.splitter.addWidget(self.tabs)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)
        self.splitter.addWidget(self.log_view)

        self.splitter.setStretchFactor(0, 9)
        self.splitter.setStretchFactor(1, 1)

        layout.addWidget(self.splitter)
        self.setCentralWidget(central)


        # ---------- SysID ----------
        self.sysid_window = SysIDWindow(interface, dir_name)
        self.tabs.addTab(self._wrap_qmainwindow(self.sysid_window), "SysID")

        # ---------- BBA ----------
        self.bba_window = BBAWindow(interface, dir_name)
        self.tabs.addTab(self._wrap_qmainwindow(self.bba_window), "BBA")

        # ---------- SysGUI (裏で1回だけ生成) ----------
        self.sysgui = SysGUISource(interface, dir_name)
        self.sysgui.hide()
        print("SysGUI tabs:", [self.sysgui.tabs.tabText(i) for i in range(self.sysgui.tabs.count())])
        for i in range(self.sysgui.tabs.count()-1, -1, -1):
            if self.sysgui.tabs.tabText(i) == "Response":
                self.sysgui.tabs.removeTab(i)


        # ---------- 抜き出し ----------
        self._extract_and_add_tab("Orbit")
        self._extract_and_add_tab("Dispersion")
        self._extract_and_add_tab("IPBSM")
        self._extract_and_add_tab("Misalignment")

        sys.stdout = EmittingStream(self.log_view)
        sys.stderr = EmittingStream(self.log_view)
        if hasattr(interface, "log_messages"):
            interface.log_messages(self.log_view.append)

    def _wrap_qmainwindow(self, window: QMainWindow):
        container = QWidget()
        layout = QVBoxLayout(container)
        cw = window.centralWidget()
        if cw is not None:
            layout.addWidget(cw)
        return container

    def _extract_and_add_tab(self, name: str):
        src_tabs = self.sysgui.tabs
        for i in range(src_tabs.count()):
            if src_tabs.tabText(i) == name:
                widget = src_tabs.widget(i)
                src_tabs.removeTab(i)       # SysGUI 側から取り外す
                self.tabs.addTab(widget, name)  # UnifiedGUI 側に移す
                return
        raise RuntimeError(f"[UnifiedGUI] SysGUI に '{name}' タブが見つかりません。")
    
    





# --------- 単体起動用 ---------
if __name__ == "__main__":
    # 既存の選択ダイアログがある前提
    from SelectInterface import InterfaceSelectionDialog

    app = QApplication(sys.argv)

    dlg = InterfaceSelectionDialog(selected_acc='ATF2', parent=None)
    if not dlg.exec():
        sys.exit(1)

    interface = dlg.selected_interface
    project = dlg.selected_interface_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"Data/{project}_{ts}"
    os.makedirs(dir_name, exist_ok=True)

    w = UnifiedGUI(interface, dir_name)
    w.show()
    sys.exit(app.exec())
