# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Matvei\Desktop\untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(568, 736)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setKerning(True)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("background-color: rgb(220, 220, 220);")
        MainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Title = QtWidgets.QLineEdit(self.centralwidget)
        self.Title.setMinimumSize(QtCore.QSize(440, 46))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.Title.setFont(font)
        self.Title.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border-color: rgb(0, 0, 0);")
        self.Title.setFrame(True)
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setReadOnly(True)
        self.Title.setObjectName("Title")
        self.verticalLayout.addWidget(self.Title, 0, QtCore.Qt.AlignTop)
        spacerItem = QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.AlgFrame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AlgFrame.sizePolicy().hasHeightForWidth())
        self.AlgFrame.setSizePolicy(sizePolicy)
        self.AlgFrame.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border-color: rgb(0, 0, 0);")
        self.AlgFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.AlgFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AlgFrame.setObjectName("AlgFrame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.AlgFrame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.AlgTitle = QtWidgets.QLineEdit(self.AlgFrame)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.AlgTitle.setFont(font)
        self.AlgTitle.setStyleSheet("")
        self.AlgTitle.setFrame(False)
        self.AlgTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.AlgTitle.setReadOnly(True)
        self.AlgTitle.setObjectName("AlgTitle")
        self.verticalLayout_2.addWidget(self.AlgTitle, 0, QtCore.Qt.AlignTop)
        self.AlgList = QtWidgets.QFrame(self.AlgFrame)
        self.AlgList.setStyleSheet("background-color: rgb(245, 245, 245);\n"
"border-color: rgb(0, 0, 0);")
        self.AlgList.setFrameShape(QtWidgets.QFrame.Box)
        self.AlgList.setFrameShadow(QtWidgets.QFrame.Raised)
        self.AlgList.setObjectName("AlgList")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.AlgList)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.KNNBox = QtWidgets.QCheckBox(self.AlgList)
        self.KNNBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.KNNBox.sizePolicy().hasHeightForWidth())
        self.KNNBox.setSizePolicy(sizePolicy)
        self.KNNBox.setMinimumSize(QtCore.QSize(0, 28))
        self.KNNBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.KNNBox.setSizeIncrement(QtCore.QSize(0, 0))
        self.KNNBox.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.KNNBox.setFont(font)
        self.KNNBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.KNNBox.setIconSize(QtCore.QSize(16, 16))
        self.KNNBox.setCheckable(True)
        self.KNNBox.setChecked(True)
        self.KNNBox.setTristate(False)
        self.KNNBox.setObjectName("KNNBox")
        self.horizontalLayout.addWidget(self.KNNBox, 0, QtCore.Qt.AlignTop)
        self.GBTBox = QtWidgets.QCheckBox(self.AlgList)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GBTBox.sizePolicy().hasHeightForWidth())
        self.GBTBox.setSizePolicy(sizePolicy)
        self.GBTBox.setMinimumSize(QtCore.QSize(0, 28))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.GBTBox.setFont(font)
        self.GBTBox.setChecked(True)
        self.GBTBox.setObjectName("GBTBox")
        self.horizontalLayout.addWidget(self.GBTBox, 0, QtCore.Qt.AlignTop)
        self.verticalLayout_2.addWidget(self.AlgList, 0, QtCore.Qt.AlignTop)
        self.verticalLayout.addWidget(self.AlgFrame, 0, QtCore.Qt.AlignTop)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem1)
        self.DetectionFrame = QtWidgets.QFrame(self.centralwidget)
        self.DetectionFrame.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border-color: rgb(0, 0, 0);")
        self.DetectionFrame.setFrameShape(QtWidgets.QFrame.Box)
        self.DetectionFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.DetectionFrame.setObjectName("DetectionFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.DetectionFrame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.DetectionTitle = QtWidgets.QLineEdit(self.DetectionFrame)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.DetectionTitle.setFont(font)
        self.DetectionTitle.setFrame(False)
        self.DetectionTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.DetectionTitle.setReadOnly(True)
        self.DetectionTitle.setObjectName("DetectionTitle")
        self.verticalLayout_3.addWidget(self.DetectionTitle, 0, QtCore.Qt.AlignTop)
        self.Parameters = QtWidgets.QFrame(self.DetectionFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Parameters.sizePolicy().hasHeightForWidth())
        self.Parameters.setSizePolicy(sizePolicy)
        self.Parameters.setStyleSheet("background-color: rgb(245, 245, 245);\n"
"border-color: rgb(0, 0, 0);")
        self.Parameters.setFrameShape(QtWidgets.QFrame.Box)
        self.Parameters.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Parameters.setObjectName("Parameters")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.Parameters)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.TickerSymbol = QtWidgets.QFrame(self.Parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TickerSymbol.sizePolicy().hasHeightForWidth())
        self.TickerSymbol.setSizePolicy(sizePolicy)
        self.TickerSymbol.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.TickerSymbol.setFrameShadow(QtWidgets.QFrame.Raised)
        self.TickerSymbol.setObjectName("TickerSymbol")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.TickerSymbol)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.TickerSymbolTitle = QtWidgets.QLineEdit(self.TickerSymbol)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TickerSymbolTitle.sizePolicy().hasHeightForWidth())
        self.TickerSymbolTitle.setSizePolicy(sizePolicy)
        self.TickerSymbolTitle.setMinimumSize(QtCore.QSize(140, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.TickerSymbolTitle.setFont(font)
        self.TickerSymbolTitle.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.TickerSymbolTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.TickerSymbolTitle.setReadOnly(True)
        self.TickerSymbolTitle.setObjectName("TickerSymbolTitle")
        self.verticalLayout_4.addWidget(self.TickerSymbolTitle, 0, QtCore.Qt.AlignTop)
        self.TickerSymbolSpace = QtWidgets.QLineEdit(self.TickerSymbol)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TickerSymbolSpace.sizePolicy().hasHeightForWidth())
        self.TickerSymbolSpace.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TickerSymbolSpace.setFont(font)
        self.TickerSymbolSpace.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.TickerSymbolSpace.setAlignment(QtCore.Qt.AlignCenter)
        self.TickerSymbolSpace.setObjectName("TickerSymbolSpace")
        self.verticalLayout_4.addWidget(self.TickerSymbolSpace, 0, QtCore.Qt.AlignTop)
        self.horizontalLayout_2.addWidget(self.TickerSymbol, 0, QtCore.Qt.AlignTop)
        self.StartDate = QtWidgets.QFrame(self.Parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartDate.sizePolicy().hasHeightForWidth())
        self.StartDate.setSizePolicy(sizePolicy)
        self.StartDate.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.StartDate.setFrameShadow(QtWidgets.QFrame.Raised)
        self.StartDate.setObjectName("StartDate")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.StartDate)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.StartDateTitle = QtWidgets.QLineEdit(self.StartDate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartDateTitle.sizePolicy().hasHeightForWidth())
        self.StartDateTitle.setSizePolicy(sizePolicy)
        self.StartDateTitle.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.StartDateTitle.setFont(font)
        self.StartDateTitle.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.StartDateTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.StartDateTitle.setReadOnly(True)
        self.StartDateTitle.setObjectName("StartDateTitle")
        self.verticalLayout_5.addWidget(self.StartDateTitle, 0, QtCore.Qt.AlignTop)
        self.StartDateSpace = QtWidgets.QDateEdit(self.StartDate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartDateSpace.sizePolicy().hasHeightForWidth())
        self.StartDateSpace.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.StartDateSpace.setFont(font)
        self.StartDateSpace.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.StartDateSpace.setWrapping(False)
        self.StartDateSpace.setAlignment(QtCore.Qt.AlignCenter)
        self.StartDateSpace.setCalendarPopup(True)
        self.StartDateSpace.setDate(QtCore.QDate(2021, 1, 1))
        self.StartDateSpace.setObjectName("StartDateSpace")
        self.verticalLayout_5.addWidget(self.StartDateSpace, 0, QtCore.Qt.AlignTop)
        self.horizontalLayout_2.addWidget(self.StartDate, 0, QtCore.Qt.AlignTop)
        self.EndDate = QtWidgets.QFrame(self.Parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EndDate.sizePolicy().hasHeightForWidth())
        self.EndDate.setSizePolicy(sizePolicy)
        self.EndDate.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.EndDate.setFrameShadow(QtWidgets.QFrame.Raised)
        self.EndDate.setObjectName("EndDate")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.EndDate)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.EndDateTitle = QtWidgets.QLineEdit(self.EndDate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EndDateTitle.sizePolicy().hasHeightForWidth())
        self.EndDateTitle.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.EndDateTitle.setFont(font)
        self.EndDateTitle.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.EndDateTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.EndDateTitle.setReadOnly(True)
        self.EndDateTitle.setObjectName("EndDateTitle")
        self.verticalLayout_6.addWidget(self.EndDateTitle)
        self.EndDateSpace = QtWidgets.QDateEdit(self.EndDate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EndDateSpace.sizePolicy().hasHeightForWidth())
        self.EndDateSpace.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.EndDateSpace.setFont(font)
        self.EndDateSpace.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.EndDateSpace.setAlignment(QtCore.Qt.AlignCenter)
        self.EndDateSpace.setCalendarPopup(True)
        self.EndDateSpace.setDate(QtCore.QDate(2021, 1, 1))
        self.EndDateSpace.setObjectName("EndDateSpace")
        self.verticalLayout_6.addWidget(self.EndDateSpace)
        self.horizontalLayout_2.addWidget(self.EndDate, 0, QtCore.Qt.AlignTop)
        self.verticalLayout_3.addWidget(self.Parameters, 0, QtCore.Qt.AlignTop)
        self.OutputPath = QtWidgets.QFrame(self.DetectionFrame)
        self.OutputPath.setStyleSheet("background-color: rgb(245, 245, 245);\n"
"border-color: rgb(0, 0, 0);")
        self.OutputPath.setFrameShape(QtWidgets.QFrame.Box)
        self.OutputPath.setFrameShadow(QtWidgets.QFrame.Raised)
        self.OutputPath.setObjectName("OutputPath")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.OutputPath)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.ChooseButton = QtWidgets.QPushButton(self.OutputPath)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.ChooseButton.setFont(font)
        self.ChooseButton.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.ChooseButton.setObjectName("ChooseButton")
        self.verticalLayout_7.addWidget(self.ChooseButton)
        self.OutputPathLine = QtWidgets.QLineEdit(self.OutputPath)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.OutputPathLine.setFont(font)
        self.OutputPathLine.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.OutputPathLine.setReadOnly(True)
        self.OutputPathLine.setObjectName("OutputPathLine")
        self.verticalLayout_7.addWidget(self.OutputPathLine, 0, QtCore.Qt.AlignTop)
        self.verticalLayout_3.addWidget(self.OutputPath)
        self.DetectButton = QtWidgets.QPushButton(self.DetectionFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DetectButton.sizePolicy().hasHeightForWidth())
        self.DetectButton.setSizePolicy(sizePolicy)
        self.DetectButton.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.DetectButton.setFont(font)
        self.DetectButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.DetectButton.setStyleSheet("background-color: rgb(245, 245, 245);")
        self.DetectButton.setCheckable(False)
        self.DetectButton.setAutoDefault(False)
        self.DetectButton.setFlat(False)
        self.DetectButton.setObjectName("DetectButton")
        self.verticalLayout_3.addWidget(self.DetectButton)
        self.Error = QtWidgets.QFrame(self.DetectionFrame)
        self.Error.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Error.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Error.setObjectName("Error")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.Error)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ErrorTitile = QtWidgets.QLineEdit(self.Error)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ErrorTitile.sizePolicy().hasHeightForWidth())
        self.ErrorTitile.setSizePolicy(sizePolicy)
        self.ErrorTitile.setMinimumSize(QtCore.QSize(0, 30))
        self.ErrorTitile.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.ErrorTitile.setFont(font)
        self.ErrorTitile.setFrame(False)
        self.ErrorTitile.setReadOnly(True)
        self.ErrorTitile.setObjectName("ErrorTitile")
        self.horizontalLayout_3.addWidget(self.ErrorTitile, 0, QtCore.Qt.AlignLeft)
        self.ErrorLine = QtWidgets.QLineEdit(self.Error)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ErrorLine.sizePolicy().hasHeightForWidth())
        self.ErrorLine.setSizePolicy(sizePolicy)
        self.ErrorLine.setMinimumSize(QtCore.QSize(0, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ErrorLine.setFont(font)
        self.ErrorLine.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"color: rgb(255, 0, 0);")
        self.ErrorLine.setText("")
        self.ErrorLine.setReadOnly(True)
        self.ErrorLine.setObjectName("ErrorLine")
        self.horizontalLayout_3.addWidget(self.ErrorLine)
        self.verticalLayout_3.addWidget(self.Error)
        self.verticalLayout.addWidget(self.DetectionFrame, 0, QtCore.Qt.AlignTop)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Title.setText(_translate("MainWindow", "Market Manipulation Detector"))
        self.AlgTitle.setText(_translate("MainWindow", "Algorithms"))
        self.KNNBox.setText(_translate("MainWindow", "KNN"))
        self.GBTBox.setText(_translate("MainWindow", "GBT"))
        self.DetectionTitle.setText(_translate("MainWindow", "Detection"))
        self.TickerSymbolTitle.setText(_translate("MainWindow", "Ticker Symbol"))
        self.StartDateTitle.setText(_translate("MainWindow", "Start Date"))
        self.EndDateTitle.setText(_translate("MainWindow", "End Date"))
        self.ChooseButton.setText(_translate("MainWindow", "Choose the output path"))
        self.DetectButton.setText(_translate("MainWindow", "Detect"))
        self.ErrorTitile.setText(_translate("MainWindow", "Error:"))
