from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QVBoxLayout, QDialog, QLabel, QPlainTextEdit)
import matplotlib
matplotlib.use("QtAgg")

class LogConsole(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Console")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumSize(520, 320)
        self.resize(700, 420)
        self.setSizeGripEnabled(True)
        self.setSizeGripEnabled(True)
        self._title=None
        self.text=QPlainTextEdit(self)
        self.text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)

    def log(self,message):
        self.text.appendPlainText(message)
        print(message)


