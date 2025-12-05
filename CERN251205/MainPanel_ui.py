# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainPanel.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(400, 200)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(16, 16, 16, 16)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.setObjectName(u"horizontal_layout")
        self.buttons_container = QWidget(self.centralwidget)
        self.buttons_container.setObjectName(u"buttons_container")
        self.buttons_container.setMaximumSize(QSize(320, 16777215))
        self.interfaces_buttons_layout = QVBoxLayout(self.buttons_container)
        self.interfaces_buttons_layout.setObjectName(u"interfaces_buttons_layout")
        self.interfaces_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.label_info = QLabel(self.buttons_container)
        self.label_info.setObjectName(u"label_info")

        self.interfaces_buttons_layout.addWidget(self.label_info, 0, Qt.AlignHCenter)

        self.sysid_interface_button = QPushButton(self.buttons_container)
        self.sysid_interface_button.setObjectName(u"sysid_interface_button")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sysid_interface_button.sizePolicy().hasHeightForWidth())
        self.sysid_interface_button.setSizePolicy(sizePolicy)

        self.interfaces_buttons_layout.addWidget(self.sysid_interface_button)

        self.compute_matrix_button = QPushButton(self.buttons_container)
        self.compute_matrix_button.setObjectName(u"compute_matrix_button")
        sizePolicy.setHeightForWidth(self.compute_matrix_button.sizePolicy().hasHeightForWidth())
        self.compute_matrix_button.setSizePolicy(sizePolicy)

        self.interfaces_buttons_layout.addWidget(self.compute_matrix_button)

        self.bba_interface_button = QPushButton(self.buttons_container)
        self.bba_interface_button.setObjectName(u"bba_interface_button")
        sizePolicy.setHeightForWidth(self.bba_interface_button.sizePolicy().hasHeightForWidth())
        self.bba_interface_button.setSizePolicy(sizePolicy)

        self.interfaces_buttons_layout.addWidget(self.bba_interface_button)

        self.emittance_interface_button = QPushButton(self.buttons_container)
        self.emittance_interface_button.setObjectName(u"emittance_interface_button")
        sizePolicy.setHeightForWidth(self.emittance_interface_button.sizePolicy().hasHeightForWidth())
        self.emittance_interface_button.setSizePolicy(sizePolicy)

        self.interfaces_buttons_layout.addWidget(self.emittance_interface_button)

        self.knobs_interface_button = QPushButton(self.buttons_container)
        self.knobs_interface_button.setObjectName(u"knobs_interface_button")
        sizePolicy.setHeightForWidth(self.knobs_interface_button.sizePolicy().hasHeightForWidth())
        self.knobs_interface_button.setSizePolicy(sizePolicy)

        self.interfaces_buttons_layout.addWidget(self.knobs_interface_button)


        self.horizontal_layout.addWidget(self.buttons_container, 0, Qt.AlignHCenter)


        self.verticalLayout.addLayout(self.horizontal_layout)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SYSID", None))
        self.label_info.setText(QCoreApplication.translate("MainWindow", u"Choose one of the following applications:", None))
        self.sysid_interface_button.setText(QCoreApplication.translate("MainWindow", u"SysID", None))
        self.compute_matrix_button.setText(QCoreApplication.translate("MainWindow", u"Compute Response Matrix", None))
        self.bba_interface_button.setText(QCoreApplication.translate("MainWindow", u"Beam Based Alignment", None))
        self.emittance_interface_button.setText(QCoreApplication.translate("MainWindow", u"Emittance Measurement", None))
        self.knobs_interface_button.setText(QCoreApplication.translate("MainWindow", u"Knobs", None))
    # retranslateUi

