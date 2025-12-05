# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SysID_GUI.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(751, 685)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setEnabled(True)
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(16, 16, 16, 16)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.setObjectName(u"horizontal_layout")
        self.left_layout = QVBoxLayout()
        self.left_layout.setSpacing(2)
        self.left_layout.setObjectName(u"left_layout")
        self.left_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.correctors_header_layout = QHBoxLayout()
        self.correctors_header_layout.setObjectName(u"correctors_header_layout")
        self.correctors_label = QLabel(self.centralwidget)
        self.correctors_label.setObjectName(u"correctors_label")

        self.correctors_header_layout.addWidget(self.correctors_label)


        self.left_layout.addLayout(self.correctors_header_layout)

        self.correctors_list = QListWidget(self.centralwidget)
        self.correctors_list.setObjectName(u"correctors_list")
        self.correctors_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.left_layout.addWidget(self.correctors_list)

        self.correctors_buttons_layout = QHBoxLayout()
        self.correctors_buttons_layout.setObjectName(u"correctors_buttons_layout")
        self.save_correctors_button = QPushButton(self.centralwidget)
        self.save_correctors_button.setObjectName(u"save_correctors_button")

        self.correctors_buttons_layout.addWidget(self.save_correctors_button)

        self.load_correctors_button = QPushButton(self.centralwidget)
        self.load_correctors_button.setObjectName(u"load_correctors_button")

        self.correctors_buttons_layout.addWidget(self.load_correctors_button)

        self.clear_correctors_button = QPushButton(self.centralwidget)
        self.clear_correctors_button.setObjectName(u"clear_correctors_button")

        self.correctors_buttons_layout.addWidget(self.clear_correctors_button)


        self.left_layout.addLayout(self.correctors_buttons_layout)

        self.bpms_header_layout = QHBoxLayout()
        self.bpms_header_layout.setObjectName(u"bpms_header_layout")
        self.bpms_label = QLabel(self.centralwidget)
        self.bpms_label.setObjectName(u"bpms_label")

        self.bpms_header_layout.addWidget(self.bpms_label)


        self.left_layout.addLayout(self.bpms_header_layout)

        self.bpms_list = QListWidget(self.centralwidget)
        self.bpms_list.setObjectName(u"bpms_list")
        self.bpms_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.left_layout.addWidget(self.bpms_list)

        self.bpms_buttons_layout = QHBoxLayout()
        self.bpms_buttons_layout.setObjectName(u"bpms_buttons_layout")
        self.save_bpms_button = QPushButton(self.centralwidget)
        self.save_bpms_button.setObjectName(u"save_bpms_button")

        self.bpms_buttons_layout.addWidget(self.save_bpms_button)

        self.load_bpms_button = QPushButton(self.centralwidget)
        self.load_bpms_button.setObjectName(u"load_bpms_button")

        self.bpms_buttons_layout.addWidget(self.load_bpms_button)

        self.clear_bpms_button = QPushButton(self.centralwidget)
        self.clear_bpms_button.setObjectName(u"clear_bpms_button")

        self.bpms_buttons_layout.addWidget(self.clear_bpms_button)


        self.left_layout.addLayout(self.bpms_buttons_layout)


        self.horizontal_layout.addLayout(self.left_layout)

        self.right_layout = QVBoxLayout()
        self.right_layout.setSpacing(2)
        self.right_layout.setObjectName(u"right_layout")
        self.info_label = QLabel(self.centralwidget)
        self.info_label.setObjectName(u"info_label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.info_label.sizePolicy().hasHeightForWidth())
        self.info_label.setSizePolicy(sizePolicy)

        self.right_layout.addWidget(self.info_label)

        self.working_directory_layout = QHBoxLayout()
        self.working_directory_layout.setObjectName(u"working_directory_layout")
        self.working_directory_input = QLineEdit(self.centralwidget)
        self.working_directory_input.setObjectName(u"working_directory_input")

        self.working_directory_layout.addWidget(self.working_directory_input)

        self.working_directory_dialog = QPushButton(self.centralwidget)
        self.working_directory_dialog.setObjectName(u"working_directory_dialog")

        self.working_directory_layout.addWidget(self.working_directory_dialog)


        self.right_layout.addLayout(self.working_directory_layout)

        self.options_layout = QVBoxLayout()
        self.options_layout.setObjectName(u"options_layout")
        self.options_label = QLabel(self.centralwidget)
        self.options_label.setObjectName(u"options_label")
        sizePolicy.setHeightForWidth(self.options_label.sizePolicy().hasHeightForWidth())
        self.options_label.setSizePolicy(sizePolicy)

        self.options_layout.addWidget(self.options_label)

        self.cycle_mode_layout = QHBoxLayout()
        self.cycle_mode_layout.setObjectName(u"cycle_mode_layout")
        self.cycle_mode_label = QLabel(self.centralwidget)
        self.cycle_mode_label.setObjectName(u"cycle_mode_label")

        self.cycle_mode_layout.addWidget(self.cycle_mode_label)

        self.cycle_mode_combobox = QComboBox(self.centralwidget)
        self.cycle_mode_combobox.addItem("")
        self.cycle_mode_combobox.addItem("")
        self.cycle_mode_combobox.setObjectName(u"cycle_mode_combobox")

        self.cycle_mode_layout.addWidget(self.cycle_mode_combobox)


        self.options_layout.addLayout(self.cycle_mode_layout)

        self.current_layout = QHBoxLayout()
        self.current_layout.setObjectName(u"current_layout")
        self.current_label = QLabel(self.centralwidget)
        self.current_label.setObjectName(u"current_label")

        self.current_layout.addWidget(self.current_label)

        self.current_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.current_layout.addItem(self.current_spacer)

        self.horizontal_current_label = QLabel(self.centralwidget)
        self.horizontal_current_label.setObjectName(u"horizontal_current_label")

        self.current_layout.addWidget(self.horizontal_current_label)

        self.max_horizontal_current_spinbox = QDoubleSpinBox(self.centralwidget)
        self.max_horizontal_current_spinbox.setObjectName(u"max_horizontal_current_spinbox")
        self.max_horizontal_current_spinbox.setSingleStep(0.010000000000000)

        self.current_layout.addWidget(self.max_horizontal_current_spinbox)

        self.vertical_current_label = QLabel(self.centralwidget)
        self.vertical_current_label.setObjectName(u"vertical_current_label")

        self.current_layout.addWidget(self.vertical_current_label)

        self.max_vertical_current_spinbox = QDoubleSpinBox(self.centralwidget)
        self.max_vertical_current_spinbox.setObjectName(u"max_vertical_current_spinbox")
        self.max_vertical_current_spinbox.setSingleStep(0.010000000000000)

        self.current_layout.addWidget(self.max_vertical_current_spinbox)


        self.options_layout.addLayout(self.current_layout)

        self.excursion_layout = QHBoxLayout()
        self.excursion_layout.setObjectName(u"excursion_layout")
        self.excursion_label = QLabel(self.centralwidget)
        self.excursion_label.setObjectName(u"excursion_label")

        self.excursion_layout.addWidget(self.excursion_label)

        self.excursion_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.excursion_layout.addItem(self.excursion_spacer)

        self.horizontal_excursion_label = QLabel(self.centralwidget)
        self.horizontal_excursion_label.setObjectName(u"horizontal_excursion_label")

        self.excursion_layout.addWidget(self.horizontal_excursion_label)

        self.horizontal_excursion_spinbox = QDoubleSpinBox(self.centralwidget)
        self.horizontal_excursion_spinbox.setObjectName(u"horizontal_excursion_spinbox")
        self.horizontal_excursion_spinbox.setSingleStep(0.100000000000000)
        self.horizontal_excursion_spinbox.setValue(0.500000000000000)

        self.excursion_layout.addWidget(self.horizontal_excursion_spinbox)

        self.vertical_excursion_label = QLabel(self.centralwidget)
        self.vertical_excursion_label.setObjectName(u"vertical_excursion_label")

        self.excursion_layout.addWidget(self.vertical_excursion_label)

        self.vertical_excursion_spinbox = QDoubleSpinBox(self.centralwidget)
        self.vertical_excursion_spinbox.setObjectName(u"vertical_excursion_spinbox")
        self.vertical_excursion_spinbox.setSingleStep(0.100000000000000)
        self.vertical_excursion_spinbox.setValue(0.500000000000000)

        self.excursion_layout.addWidget(self.vertical_excursion_spinbox)


        self.options_layout.addLayout(self.excursion_layout)

        self.plot_widget = QWidget(self.centralwidget)
        self.plot_widget.setObjectName(u"plot_widget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plot_widget.sizePolicy().hasHeightForWidth())
        self.plot_widget.setSizePolicy(sizePolicy1)
        self.plot_widget.setMinimumSize(QSize(400, 300))

        self.options_layout.addWidget(self.plot_widget)


        self.right_layout.addLayout(self.options_layout)


        self.horizontal_layout.addLayout(self.right_layout)


        self.verticalLayout.addLayout(self.horizontal_layout)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setObjectName(u"buttons_layout")
        self.start_button = QPushButton(self.centralwidget)
        self.start_button.setObjectName(u"start_button")
        self.start_button.setStyleSheet(u"background-color: red; color: white;")

        self.buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton(self.centralwidget)
        self.stop_button.setObjectName(u"stop_button")
        self.stop_button.setStyleSheet(u"background-color: green; color: white;")

        self.buttons_layout.addWidget(self.stop_button)


        self.verticalLayout.addLayout(self.buttons_layout)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SYSID", None))
        self.correctors_label.setText(QCoreApplication.translate("MainWindow", u"Correctors", None))
        self.save_correctors_button.setText(QCoreApplication.translate("MainWindow", u"Save As..", None))
        self.load_correctors_button.setText(QCoreApplication.translate("MainWindow", u"Load..", None))
        self.clear_correctors_button.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.bpms_label.setText(QCoreApplication.translate("MainWindow", u"BPMs", None))
        self.save_bpms_button.setText(QCoreApplication.translate("MainWindow", u"Save As..", None))
        self.load_bpms_button.setText(QCoreApplication.translate("MainWindow", u"Load..", None))
        self.clear_bpms_button.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.info_label.setText(QCoreApplication.translate("MainWindow", u"Data Storage:", None))
        self.working_directory_input.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Working directory:", None))
        self.working_directory_dialog.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.options_label.setText(QCoreApplication.translate("MainWindow", u"Options", None))
        self.cycle_mode_label.setText(QCoreApplication.translate("MainWindow", u"Cycle mode", None))
        self.cycle_mode_combobox.setItemText(0, QCoreApplication.translate("MainWindow", u"Repeat all", None))
        self.cycle_mode_combobox.setItemText(1, QCoreApplication.translate("MainWindow", u"Repeat selected", None))

        self.current_label.setText(QCoreApplication.translate("MainWindow", u"Max strength (gauss*m)", None))
        self.horizontal_current_label.setText(QCoreApplication.translate("MainWindow", u"H:", None))
        self.vertical_current_label.setText(QCoreApplication.translate("MainWindow", u" V:", None))
        self.excursion_label.setText(QCoreApplication.translate("MainWindow", u"Target orbit excursion (mm)", None))
        self.horizontal_excursion_label.setText(QCoreApplication.translate("MainWindow", u"H:", None))
        self.vertical_excursion_label.setText(QCoreApplication.translate("MainWindow", u"V:", None))
        self.start_button.setText(QCoreApplication.translate("MainWindow", u"START", None))
        self.stop_button.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
    # retranslateUi

