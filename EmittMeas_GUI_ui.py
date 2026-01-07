# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'EmittMeas_GUI.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QVBoxLayout, QWidget)

class Ui_EmittMeasMainWindow(object):
    def setupUi(self, EmittMeasMainWindow):
        if not EmittMeasMainWindow.objectName():
            EmittMeasMainWindow.setObjectName(u"EmittMeasMainWindow")
        EmittMeasMainWindow.resize(1000, 700)
        self.centralwidget = QWidget(EmittMeasMainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.mainVerticalLayout = QVBoxLayout(self.centralwidget)
        self.mainVerticalLayout.setObjectName(u"mainVerticalLayout")
        self.mainVerticalLayout.setContentsMargins(8, 8, 8, 8)
        self.groupBoxDataOptics = QGroupBox(self.centralwidget)
        self.groupBoxDataOptics.setObjectName(u"groupBoxDataOptics")
        self.gridLayoutDataOptics = QGridLayout(self.groupBoxDataOptics)
        self.gridLayoutDataOptics.setObjectName(u"gridLayoutDataOptics")
        self.labelTwissFile = QLabel(self.groupBoxDataOptics)
        self.labelTwissFile.setObjectName(u"labelTwissFile")

        self.gridLayoutDataOptics.addWidget(self.labelTwissFile, 1, 0, 1, 1)

        self.twissFileLineEdit = QLineEdit(self.groupBoxDataOptics)
        self.twissFileLineEdit.setObjectName(u"twissFileLineEdit")
        self.twissFileLineEdit.setReadOnly(True)

        self.gridLayoutDataOptics.addWidget(self.twissFileLineEdit, 1, 1, 1, 1)

        self.loadTwissButton = QPushButton(self.groupBoxDataOptics)
        self.loadTwissButton.setObjectName(u"loadTwissButton")

        self.gridLayoutDataOptics.addWidget(self.loadTwissButton, 1, 2, 1, 1)

        self.labelAlgorithm = QLabel(self.groupBoxDataOptics)
        self.labelAlgorithm.setObjectName(u"labelAlgorithm")

        self.gridLayoutDataOptics.addWidget(self.labelAlgorithm, 1, 3, 1, 1)

        self.algorithmComboBox = QComboBox(self.groupBoxDataOptics)
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.addItem("")
        self.algorithmComboBox.setObjectName(u"algorithmComboBox")

        self.gridLayoutDataOptics.addWidget(self.algorithmComboBox, 1, 4, 1, 1)


        self.mainVerticalLayout.addWidget(self.groupBoxDataOptics)

        self.middleHorizontalLayout = QHBoxLayout()
        self.middleHorizontalLayout.setObjectName(u"middleHorizontalLayout")
        self.groupBoxControls = QGroupBox(self.centralwidget)
        self.groupBoxControls.setObjectName(u"groupBoxControls")
        self.groupBoxControls.setMaximumSize(QSize(100, 1000))
        self.controlsLayout = QVBoxLayout(self.groupBoxControls)
        self.controlsLayout.setObjectName(u"controlsLayout")
        self.labelScreensUsed = QLabel(self.groupBoxControls)
        self.labelScreensUsed.setObjectName(u"labelScreensUsed")

        self.controlsLayout.addWidget(self.labelScreensUsed)

        self.screensListWidget = QListWidget(self.groupBoxControls)
        self.screensListWidget.setObjectName(u"screensListWidget")
        self.screensListWidget.setMinimumSize(QSize(20, 120))
        self.screensListWidget.setMaximumSize(QSize(200, 16777215))

        self.controlsLayout.addWidget(self.screensListWidget)

        self.controlsSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.controlsLayout.addItem(self.controlsSpacer)


        self.middleHorizontalLayout.addWidget(self.groupBoxControls)

        self.groupBoxPlots = QGroupBox(self.centralwidget)
        self.groupBoxPlots.setObjectName(u"groupBoxPlots")
        self.gridLayoutPlots = QGridLayout(self.groupBoxPlots)
        self.gridLayoutPlots.setObjectName(u"gridLayoutPlots")
        self.plotFrame11 = QFrame(self.groupBoxPlots)
        self.plotFrame11.setObjectName(u"plotFrame11")
        self.plotFrame11.setFrameShape(QFrame.StyledPanel)
        self.plotFrame11.setFrameShadow(QFrame.Sunken)

        self.gridLayoutPlots.addWidget(self.plotFrame11, 0, 0, 1, 1)

        self.plotFrame12 = QFrame(self.groupBoxPlots)
        self.plotFrame12.setObjectName(u"plotFrame12")
        self.plotFrame12.setFrameShape(QFrame.StyledPanel)
        self.plotFrame12.setFrameShadow(QFrame.Sunken)

        self.gridLayoutPlots.addWidget(self.plotFrame12, 0, 1, 1, 1)

        self.plotFrame21 = QFrame(self.groupBoxPlots)
        self.plotFrame21.setObjectName(u"plotFrame21")
        self.plotFrame21.setFrameShape(QFrame.StyledPanel)
        self.plotFrame21.setFrameShadow(QFrame.Sunken)

        self.gridLayoutPlots.addWidget(self.plotFrame21, 1, 0, 1, 1)

        self.plotFrame22 = QFrame(self.groupBoxPlots)
        self.plotFrame22.setObjectName(u"plotFrame22")
        self.plotFrame22.setFrameShape(QFrame.StyledPanel)
        self.plotFrame22.setFrameShadow(QFrame.Sunken)

        self.gridLayoutPlots.addWidget(self.plotFrame22, 1, 1, 1, 1)


        self.middleHorizontalLayout.addWidget(self.groupBoxPlots)


        self.mainVerticalLayout.addLayout(self.middleHorizontalLayout)

        self.bottomButtonsLayout = QHBoxLayout()
        self.bottomButtonsLayout.setObjectName(u"bottomButtonsLayout")
        self.bottomSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottomButtonsLayout.addItem(self.bottomSpacer)

        self.measureButton = QPushButton(self.centralwidget)
        self.measureButton.setObjectName(u"measureButton")

        self.bottomButtonsLayout.addWidget(self.measureButton)

        self.exportButton = QPushButton(self.centralwidget)
        self.exportButton.setObjectName(u"exportButton")

        self.bottomButtonsLayout.addWidget(self.exportButton)


        self.mainVerticalLayout.addLayout(self.bottomButtonsLayout)

        EmittMeasMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(EmittMeasMainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1000, 21))
        EmittMeasMainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(EmittMeasMainWindow)
        self.statusbar.setObjectName(u"statusbar")
        EmittMeasMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(EmittMeasMainWindow)

        QMetaObject.connectSlotsByName(EmittMeasMainWindow)
    # setupUi

    def retranslateUi(self, EmittMeasMainWindow):
        EmittMeasMainWindow.setWindowTitle(QCoreApplication.translate("EmittMeasMainWindow", u"Emittance Measurement", None))
        self.groupBoxDataOptics.setTitle(QCoreApplication.translate("EmittMeasMainWindow", u"Data loading", None))
        self.labelTwissFile.setText(QCoreApplication.translate("EmittMeasMainWindow", u"Load data file:", None))
        self.loadTwissButton.setText(QCoreApplication.translate("EmittMeasMainWindow", u"...", None))
        self.labelAlgorithm.setText(QCoreApplication.translate("EmittMeasMainWindow", u"Reconstruction:", None))
        self.algorithmComboBox.setItemText(0, QCoreApplication.translate("EmittMeasMainWindow", u"2D projected (x, y)", None))
        self.algorithmComboBox.setItemText(1, QCoreApplication.translate("EmittMeasMainWindow", u"4D with coupling", None))

        self.labelScreensUsed.setText(QCoreApplication.translate("EmittMeasMainWindow", u"OTRs:", None))
        self.groupBoxPlots.setTitle(QCoreApplication.translate("EmittMeasMainWindow", u"Plots", None))
        self.measureButton.setText(QCoreApplication.translate("EmittMeasMainWindow", u"Measure", None))
        self.exportButton.setText(QCoreApplication.translate("EmittMeasMainWindow", u"Save as...", None))
    # retranslateUi

