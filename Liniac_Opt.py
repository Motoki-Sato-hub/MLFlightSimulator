# -*- coding: utf-8 -*-
import sys
import time
import datetime
import csv
from pathlib import Path
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QLabel, QFileDialog, QTextEdit, QGroupBox,
    QMessageBox
)

try:
    from epics import PV
except ImportError:
    class PV:
        def __init__(self, name): self.name = name
        def get(self): return 1.0  # ダミー値
        def put(self, val): pass


try:
    from interfaces.ATF2.InterfaceATF2_Linac import InterfaceATF2_Linac
except ImportError:
    print("InterfaceATF2_Linac.py not found. Mocking class.")
    class InterfaceATF2_Linac:
        def __init__(self): pass

class OptimizationWorker(QThread):

    log_signal = pyqtSignal(str)       # ログ出力用
    status_signal = pyqtSignal(str)    # ステータス表示用
    finished_signal = pyqtSignal()     # 完了通知

    def __init__(self, config, save_dir):
        super().__init__()
        self.config = config
        self.save_dir = Path(save_dir)
        self.is_running = True
        self.interface = InterfaceATF2_Linac()
        
        # 透過率計算用のPV定義（★環境に合わせて変更してください）
        self.pv_ict_upstream = PV('ICT_AVE:AVERAGE:GUN_MEAN') 
        self.pv_ict_downstream = PV('ICT_AVE:AVERAGE:DR_MEAN')

        # ログ保存用のファイルパス生成
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.save_dir / f"LiniacOptimization_Log_{timestamp}.csv"
        self._init_csv()

    def _init_csv(self):
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Step', 'Device_PV', 'Set_Value', 'Transmission', 'Note'])

    def run(self):
        self.log_signal.emit("=== Optimization Started ===")
        
        try:
            # 1. Klystron Phase Tuning
            if self.config['klystron'] and self.is_running:
                self.log_signal.emit("--- Starting Klystron Phase Optimization ---")
                # 1番から8番まで順に最適化
                for i in range(1, 9):
                    if not self.is_running: break
                    pv_name = f'CM{i}L:phaseWrite'
                    current_val = PV(f'CM{i}L:phaseRead').get()
                    if current_val is None: current_val = 0
                    
                    scan_values = np.arange(current_val - 8, current_val + 8, 1)
                    self.perform_1d_scan(f"Klystron {i}", pv_name, scan_values)

            # 2. Quad Magnet Tuning
            if self.config['quad'] and self.is_running:
                self.log_signal.emit("--- Starting Quadrupole Magnet Optimization ---")
                quads = ['QA1L', 'QA2L', 'QA3L', 'QA4L', 'QA5L']
                # -10A ~ 10A, 0.5A刻み
                scan_values = np.arange(-10.0, 10.5, 0.5)
                
                for q_name in quads:
                    if not self.is_running: break
                    pv_name = f'{q_name}:currentWrite'
                    self.perform_1d_scan(q_name, pv_name, scan_values)

            # 3. BT Magnet Tuning
            if self.config['bt'] and self.is_running:
                self.log_signal.emit("--- Starting BT Magnet Optimization ---")
                bt_mags = ['BH10T', 'BV20T', 'BH30T']
                # -10A ~ 10A, 0.5A刻み
                scan_values = np.arange(-10.0, 10.5, 0.5)
                
                for m_name in bt_mags:
                    if not self.is_running: break
                    pv_name = f'{m_name}:currentWrite'
                    self.perform_1d_scan(m_name, pv_name, scan_values)

        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")
        finally:
            self.log_signal.emit("=== Optimization Finished ===")
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False
        self.log_signal.emit("!!! Stop Requested !!!")

    def get_transmission(self):
        """Downstream / Upstream"""
        try:
            time.sleep(0.5) 
            up = self.pv_ict_upstream.get()
            down = self.pv_ict_downstream.get()
            
            if up is None or down is None or up == 0:
                return 0.0
            
            return float(down) / float(up)
        except Exception:
            return 0.0

    def perform_1d_scan(self, device_label, pv_name, scan_values):
        """
        汎用1次元スキャンメソッド
        指定されたPVに対してscan_valuesを順に設定し、透過率が最大の場所に戻す。
        """
        self.log_signal.emit(f"Scanning {device_label} ({pv_name})...")
        pv_obj = PV(pv_name)
        
        best_trans = -1.0
        best_val = scan_values[0]
        history = []

        # 初期値の記録（オプション）
        initial_val = pv_obj.get() # currentReadなどがあればそちらを使うべきだが簡易化

        for val in scan_values:
            if not self.is_running: return

            # 値の設定
            pv_obj.put(val)
            # マグネット応答待ち（時定数などを考慮して調整してください）
            time.sleep(1.0) 
            
            # 透過率測定
            trans = self.get_transmission()
            
            # ログ記録
            self.log_csv(device_label, val, trans)
            msg = f"  Set {val:.2f} -> Trans: {trans:.4f}"
            self.log_signal.emit(msg)
            
            history.append((val, trans))

            # 最大値更新チェック
            if trans > best_trans:
                best_trans = trans
                best_val = val

        # 最適値への復帰
        if self.is_running:
            self.log_signal.emit(f"-> Best for {device_label}: {best_val:.2f} (Trans: {best_trans:.4f})")
            pv_obj.put(best_val)
            time.sleep(1.0)
            
            # 最終確認
            final_trans = self.get_transmission()
            self.log_csv(device_label, best_val, final_trans, note="OPTIMIZED_SET")

    def log_csv(self, device, value, trans, note=""):
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().strftime('%H:%M:%S'),
                    "Scan",
                    device,
                    value,
                    trans,
                    note
                ])
        except Exception as e:
            print(f"CSV Write Error: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linac Beam Auto-Tuning GUI")
        self.resize(600, 700)
        
        self.worker = None
        self.save_path = Path.cwd() / "Data"

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. Settings Area
        grp_settings = QGroupBox("Optimization Targets")
        layout_settings = QVBoxLayout()
        
        self.chk_klystron = QCheckBox("1. Klystron Phase (1-8)")
        self.chk_quad = QCheckBox("2. Quad Magnets (QA1L-QA5L)")
        self.chk_bt = QCheckBox("3. BT Magnets (BH10T, BV20T, BH30T)")
        
        # デフォルトで全てチェック
        self.chk_klystron.setChecked(True)
        self.chk_quad.setChecked(True)
        self.chk_bt.setChecked(True)

        layout_settings.addWidget(self.chk_klystron)
        layout_settings.addWidget(self.chk_quad)
        layout_settings.addWidget(self.chk_bt)
        grp_settings.setLayout(layout_settings)
        main_layout.addWidget(grp_settings)

        # 2. Path Selection
        grp_path = QGroupBox("Data Save Location")
        layout_path = QHBoxLayout()
        self.lbl_path = QLabel(str(self.save_path))
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_folder)
        
        layout_path.addWidget(self.lbl_path)
        layout_path.addWidget(btn_browse)
        grp_path.setLayout(layout_path)
        main_layout.addWidget(grp_path)

        # 3. Control Buttons
        layout_ctrl = QHBoxLayout()
        self.btn_start = QPushButton("START Optimization")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("font-weight: bold; background-color: #d0f0c0;")
        self.btn_start.clicked.connect(self.start_optimization)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setStyleSheet("font-weight: bold; background-color: #f0c0c0;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_optimization)

        layout_ctrl.addWidget(self.btn_start)
        layout_ctrl.addWidget(self.btn_stop)
        main_layout.addLayout(layout_ctrl)

        # 4. Log Display
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        main_layout.addWidget(QLabel("Operation Log:"))
        main_layout.addWidget(self.txt_log)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", str(self.save_path))
        if folder:
            self.save_path = Path(folder)
            self.lbl_path.setText(str(self.save_path))

    def start_optimization(self):
        # 設定の取得
        config = {
            'klystron': self.chk_klystron.isChecked(),
            'quad': self.chk_quad.isChecked(),
            'bt': self.chk_bt.isChecked()
        }

        if not any(config.values()):
            QMessageBox.warning(self, "Warning", "Please select at least one optimization target.")
            return

        # UI状態更新
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.chk_klystron.setEnabled(False)
        self.chk_quad.setEnabled(False)
        self.chk_bt.setEnabled(False)
        self.txt_log.clear()

        # スレッド起動
        self.worker = OptimizationWorker(config, self.save_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.optimization_finished)
        self.worker.start()

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False) # 二重押し防止

    def optimization_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.chk_klystron.setEnabled(True)
        self.chk_quad.setEnabled(True)
        self.chk_bt.setEnabled(True)
        QMessageBox.information(self, "Finished", "Optimization process finished.")

    def append_log(self, text):
        self.txt_log.append(text)
        # 常に最新行を表示
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())