from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox,
    QRadioButton, QLabel
)

class SelectAcc(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select a machine")
        self.selected_machine = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose one of the following accelerators:"))
        accs= [ 'ATF2', 'CLEAR']

        self.radio_buttons = []
        for f in accs:
            rb = QRadioButton(f)
            self.radio_buttons.append(rb)
            layout.addWidget(rb)

        if self.radio_buttons:
            self.radio_buttons[0].setChecked(True) #default ATF2

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
                self.selected_machine = rb.text()
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



class InterfaceSelectionDialog(QDialog):
    def __init__(self, selected_acc,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select an Interface")
        self.selected_interface = None
        self.selected_acc = selected_acc
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose one of the following Interfaces:"))
        if selected_acc=='ATF2':
            interfaces = ['InterfaceATF2_DR', 'InterfaceATF2_Ext', 'InterfaceATF2_Linac', 'InterfaceATF2_Ext_RFTrack']
        elif selected_acc=='CLEAR':
            interfaces = ['InterfaceCLEAR_RFTrack' , 'InterfaceCLEAR_real']
        self.radio_buttons = []
        for f in interfaces:
            rb = QRadioButton(f)
            self.radio_buttons.append(rb)
            layout.addWidget(rb)

        if self.radio_buttons:
            self.radio_buttons[0].setChecked(True)

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
                text = rb.text()
                break
        else:
            super().accept()
            return

        if self.selected_acc == 'ATF2':
            match text:
                case 'InterfaceATF2_DR':
                    from Interfaces.ATF2.InterfaceATF2_DR import InterfaceATF2_DR
                    globals()['InterfaceATF2_DR'] = InterfaceATF2_DR
                    self.selected_interface = InterfaceATF2_DR(nsamples=3)

                case 'InterfaceATF2_Ext':
                    from Interfaces.ATF2.InterfaceATF2_Ext import InterfaceATF2_Ext
                    globals()['InterfaceATF2_Ext'] = InterfaceATF2_Ext
                    self.selected_interface = InterfaceATF2_Ext(nsamples=3)

                case 'InterfaceATF2_Linac':
                    from Interfaces.ATF2.InterfaceATF2_Linac import InterfaceATF2_Linac
                    globals()['InterfaceATF2_Linac'] = InterfaceATF2_Linac
                    self.selected_interface = InterfaceATF2_Linac(nsamples=3)

                case 'InterfaceATF2_Ext_RFTrack':
                    from Interfaces.ATF2.InterfaceATF2_Ext_RFTrack import InterfaceATF2_Ext_RFTrack
                    globals()['InterfaceATF2_Ext_RFTrack'] = InterfaceATF2_Ext_RFTrack
                    self.selected_interface = InterfaceATF2_Ext_RFTrack(jitter=0.0, bpm_resolution=0.00, nsamples=1)

                    #TESTS:

                    self.selected_interface.align_everything()
                    #self.selected_interface.misalign_quadrupoles()
                    self.selected_interface.misalign_bpms()


        elif self.selected_acc == 'CLEAR':
            match text:
                case 'InterfaceCLEAR_RFTrack':
                    from Interfaces.CLEAR.InterfaceCLEAR_RFTrack import InterfaceCLEAR_RFTrack
                    globals()['InterfaceCLEAR_RFTrack'] = InterfaceCLEAR_RFTrack
                    self.selected_interface = InterfaceCLEAR_RFTrack(jitter=0.1, bpm_resolution=0.05, nsamples=1)

                    # TESTS:
                    #
                    self.selected_interface.align_everything()
                    self.selected_interface.misalign_quadrupoles()
                    self.selected_interface.misalign_bpms()
                case 'InterfaceCLEAR_real':
                    from Interfaces.CLEAR.InterfaceCLEAR_real import InterfaceCLEAR_real
                    globals()['InterfaceCLEAR_real'] = InterfaceCLEAR_real
                    self.selected_interface = InterfaceCLEAR_real(nsamples=3)


        self.selected_interface_name = self.selected_interface.get_name()
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

def choose_acc_and_interface(parent=None):
    acc_dialog = SelectAcc(parent=parent)
    if acc_dialog.exec():
        accelerator = acc_dialog.selected_machine
    else:
        return None

    interface_dialog=InterfaceSelectionDialog(accelerator,parent)
    if interface_dialog.exec():
        return interface_dialog.selected_interface
    else:
        return None
