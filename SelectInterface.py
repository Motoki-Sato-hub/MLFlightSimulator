import sys
import glob
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QDialogButtonBox,
    QRadioButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt

class InterfaceSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select an Interface")
        self.selected_interface = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose one of the following Interfaces:"))

        interfaces = [ 'InterfaceATF2_DR', 'InterfaceATF2_Ext', 'InterfaceATF2_Linac', 'InterfaceATF2_Ext_RFTrack' ]

        self.radio_buttons = []
        for f in interfaces:
            rb = QRadioButton(f)
            self.radio_buttons.append(rb)
            layout.addWidget(rb)

        if self.radio_buttons:
            self.radio_buttons[3].setChecked(True)

        buttons = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setDefault(True)
        layout.addWidget(self.button_box)

        self.installEventFilter(self)

    def accept(self):
        for rb in self.radio_buttons:
            if rb.isChecked():
                match rb.text():
                    case 'InterfaceATF2_DR':
                        from InterfaceATF2_DR import InterfaceATF2_DR
                        globals()['InterfaceATF2_DR'] = InterfaceATF2_DR
                        self.selected_interface = InterfaceATF2_DR(nsamples=3)

                    case 'InterfaceATF2_Ext':
                        from InterfaceATF2_Ext import InterfaceATF2_Ext
                        globals()['InterfaceATF2_Ext'] = InterfaceATF2_Ext
                        self.selected_interface = InterfaceATF2_Ext(nsamples=3)

                    case 'InterfaceATF2_Linac':
                        from InterfaceATF2_Linac import InterfaceATF2_Linac
                        globals()['InterfaceATF2_Linac'] = InterfaceATF2_Linac
                        self.selected_interface = InterfaceATF2_Linac(nsamples=3)

                    case 'InterfaceATF2_Ext_RFTrack':
                        from InterfaceATF2_Ext_RFTrack import InterfaceATF2_Ext_RFTrack
                        globals()['InterfaceATF2_Ext_RFTrack'] = InterfaceATF2_Ext_RFTrack
                        self.selected_interface = InterfaceATF2_Ext_RFTrack(jitter=0.05, bpm_resolution=0.02, nsamples=1)

                self.selected_interface_name = self.selected_interface.get_name()
                break
        super().accept()

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent, Qt

        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
                if ok_button.isEnabled():
                    ok_button.click()
                    return True
        return super().eventFilter(obj, event)