# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'smartdsp.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_smartdsp(object):
    def setupUi(self, smartdsp):
        smartdsp.setObjectName("smartdsp")
        smartdsp.resize(781, 550)
        self.centralwidget = QtWidgets.QWidget(smartdsp)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 170, 81, 81))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 330, 81, 81))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 250, 81, 81))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 10, 481, 451))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 10, 81, 81))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 10, 201, 491))
        self.label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label.setText("")
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 90, 81, 81))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(90, 270, 181, 171))
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(90, 90, 181, 171))
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(160, 50, 31, 31))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(160, 450, 31, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(290, 470, 481, 31))
        self.lineEdit.setObjectName("lineEdit")
        smartdsp.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(smartdsp)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 781, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        smartdsp.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(smartdsp)
        self.statusbar.setObjectName("statusbar")
        smartdsp.setStatusBar(self.statusbar)
        self.actionCT = QtWidgets.QAction(smartdsp)
        self.actionCT.setObjectName("actionCT")
        self.actionCXR = QtWidgets.QAction(smartdsp)
        self.actionCXR.setObjectName("actionCXR")
        self.action = QtWidgets.QAction(smartdsp)
        self.action.setObjectName("action")
        self.menu.addAction(self.actionCT)
        self.menu.addAction(self.actionCXR)
        self.menu_2.addAction(self.action)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(smartdsp)
        QtCore.QMetaObject.connectSlotsByName(smartdsp)

    def retranslateUi(self, smartdsp):
        _translate = QtCore.QCoreApplication.translate
        smartdsp.setWindowTitle(_translate("smartdsp", "MainWindow"))
        self.pushButton_3.setText(_translate("smartdsp", "prev"))
        self.pushButton_5.setText(_translate("smartdsp", "save"))
        self.pushButton_4.setText(_translate("smartdsp", "next"))
        self.pushButton.setText(_translate("smartdsp", "open"))
        self.pushButton_2.setText(_translate("smartdsp", "predict"))
        self.pushButton_8.setText(_translate("smartdsp", "PushButton"))
        self.pushButton_9.setText(_translate("smartdsp", "PushButton"))
        self.menu.setTitle(_translate("smartdsp", "模态选择"))
        self.menu_2.setTitle(_translate("smartdsp", "帮助"))
        self.actionCT.setText(_translate("smartdsp", "CT"))
        self.actionCXR.setText(_translate("smartdsp", "CXR"))
        self.action.setText(_translate("smartdsp", "联系我们"))
    
