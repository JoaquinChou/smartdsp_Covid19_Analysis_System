# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'classify_demo.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QCursor
from resnet_v2 import *
from cam import *
import os


class Ui_smartdsp(object):
    def setupUi(self, smartdsp):
        smartdsp.setObjectName("smartdsp")
        smartdsp.resize(800, 550)
        self.centralwidget = QtWidgets.QWidget(smartdsp)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 10, 81, 81))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 90, 81, 81))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setCursor(QtGui.QCursor(
            QtCore.Qt.PointingHandCursor))

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 170, 81, 81))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setCursor(QtGui.QCursor(
            QtCore.Qt.PointingHandCursor))

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 250, 81, 81))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setCursor(QtGui.QCursor(
            QtCore.Qt.PointingHandCursor))

        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 330, 81, 81))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setCursor(QtGui.QCursor(
            QtCore.Qt.PointingHandCursor))

        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(160, 50, 31, 31))
        self.pushButton_8.setObjectName("pushButton_8")

        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(160, 450, 31, 31))
        self.pushButton_9.setObjectName("pushButton_9")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 10, 201, 491))
        self.label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label.setText("")
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 10, 481, 451))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

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

        self.label_2.setStyleSheet(
            'background-color: rgb(74, 74, 74)')  ##############

        self.label_5.setStyleSheet(
            'background-color:rgb(74, 74, 74)')  ##############

        self.label_6.setStyleSheet(
            'background-color: rgb(74, 74, 74)')  ##############

        # 设置状态栏样式
        self.menu.setStyleSheet("color:white")
        self.menu.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.menu_2.setStyleSheet('color:white')
        self.menu_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.menubar.setStyleSheet("color:rgb(180, 180, 180)")

        self.retranslateUi(smartdsp)
        QtCore.QMetaObject.connectSlotsByName(smartdsp)

        self.actionCT.triggered.connect(
            self.processct)  ############################
        self.actionCXR.triggered.connect(
            self.processcxr)  ############################
        self.mode = 0  ################################################

        # 打开图像控件
        self.pushButton.setStyleSheet(
            "QPushButton:hover{color:red}"
            "QPushButton{background-color:black}"
            "QPushButton{color:white}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            "QPushButton{border-image: url(icons/open_file.png)}")  #####

        # 诊断预测控件
        self.pushButton_2.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{color:red}"
            "QPushButton{background-color:black}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            "QPushButton{border-image: url(icons/operation.png)}")  #####

        # 向上
        self.pushButton_3.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{color:red}"
            "QPushButton{background-color:black}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            # 'QPushButton{background-image:url(icons/open4.png)}'
            "QPushButton{border-image: url(icons/up.png)}"
            # "QPushButton:hover{border-image: url(img/open.png)}"
        )  #####

        # 向下
        self.pushButton_4.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{color:red}"
            "QPushButton{background-color:black}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            # 'QPushButton{background-image:url(icons/open4.png)}'
            "QPushButton{border-image: url(icons/down.png)}"
            # "QPushButton:hover{border-image: url(img/open.png)}"
        )  #####

        # 保存图像
        self.pushButton_5.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{color:red}"
            "QPushButton{background-color:black}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            # 'QPushButton{background-image:url(icons/open4.png)}'
            "QPushButton{border-image: url(icons/save_img.png)}"
            # "QPushButton:hover{border-image: url(img/open.png)}"
        )  #####
        self.pushButton_8.setStyleSheet(
            "QPushButton{color:black}"
            "QPushButton:hover{color:red}"
            # "QPushButton{background-color:lightgreen}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            # 'QPushButton{background-image:url(icons/open4.png)}'
            "QPushButton{border-image: url(icons/prev.png)}"
            # "QPushButton:hover{border-image: url(img/open.png)}"
        )  #####
        self.pushButton_9.setStyleSheet(
            "QPushButton{color:black}"
            "QPushButton:hover{color:red}"
            # "QPushButton{background-color:lightgreen}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:2px}"  ##10
            "QPushButton{padding:2px 4px}"
            # 'QPushButton{background-image:url(icons/open4.png)}'
            "QPushButton{border-image: url(icons/next.png)}"
            # "QPushButton:hover{border-image: url(img/open.png)}"
        )  #####

        self.lineEdit.setStyleSheet("color:red;background:black;")  ##
        font = QtGui.QFont()  ###
        font.setFamily('微软雅黑')  ##
        font.setBold(True)  ##
        font.setPointSize(10)  ##
        font.setWeight(75)  ##
        self.lineEdit.setFont(font)  ##
        # self.lineEdit.setStyleSheet("background:transparent;border-width:0;border-style:outset")  #######

        op = QtWidgets.QGraphicsOpacityEffect()  #######透明度
        op.setOpacity(0.9)
        self.pushButton.setGraphicsEffect(op)

    def retranslateUi(self, smartdsp):
        _translate = QtCore.QCoreApplication.translate
        smartdsp.setWindowTitle(_translate("smartdsp", "MainWindow"))
        self.pushButton_3.setText(_translate("smartdsp", "\n\n\n\n上一张"))
        self.pushButton_5.setText(_translate("smartdsp", "\n\n\n\n保存"))
        self.pushButton_4.setText(_translate("smartdsp", "\n\n\n\n下一张"))
        self.pushButton.setText(_translate("smartdsp", "\n\n\n\n打开图像"))
        self.pushButton_2.setText(_translate("smartdsp", "\n\n\n\n诊断预测"))
        self.pushButton_8.setText(_translate("smartdsp", ""))
        self.pushButton_9.setText(_translate("smartdsp", ""))
        self.menu.setTitle(_translate("smartdsp", "模态选择"))
        self.actionCT.setText(_translate("smartdsp", "CT"))
        self.actionCXR.setText(_translate("smartdsp", "CXR"))
        self.menu.setTitle(_translate("smartdsp", "模态选择"))
        self.menu_2.setTitle(_translate("smartdsp", "帮助"))
        self.actionCT.setText(_translate("smartdsp", "CT"))
        self.actionCXR.setText(_translate("smartdsp", "CXR"))
        self.action.setText(_translate("smartdsp", "联系我们"))

    def processct(self):
        self.mode = 0
        print("ct")

    def processcxr(self):
        self.mode = 1
        print("cxr")

    def openimage(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(
            self.pushButton, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        self.img = self.imgName.split('/')[-1]
        self.imgpath = self.imgName[0:-len(self.img)]
        print(self.imgName, imgType)  ###locname
        print(self.img, self.imgpath)
        print(type(self.imgName))
        jpg = QtGui.QPixmap(self.imgName).scaled(self.label_6.width(),
                                                 self.label_6.height())
        print(jpg)
        self.label_6.setPixmap(jpg)

        jpg1 = QtGui.QPixmap(self.imgName).scaled(self.label_2.width(),
                                                  self.label_2.height())
        self.label_2.setPixmap(jpg1)

    def predict(self):
        import shutil

        root_path = os.getcwd()
        root_path += '/temp/0/'
        if os.path.exists(root_path) == False:
            os.makedirs(root_path)
        files = os.listdir(root_path)
        if len(files) != 0:
            for file in files:
                os.remove(root_path + file)

        print("here")

        print(self.imgName)

        shutil.copyfile(self.imgName,
                        "./temp/0/" + str(self.imgName).split('/')[-1])
        prop, prelabel = predict_demo(self.mode)
        self.curimg = str(self.imgName).split('/')[-1]
        self.confi = prop
        self.predictlabel = prelabel
        self.lineEdit.setText("预测为" + self.predictlabel + "   " + "置信度:" +
                              self.confi)

        cam_demo(self.imgName, self.mode)
        jpg = QtGui.QPixmap("./cam.jpg").scaled(self.label_5.width(),
                                                self.label_5.height())
        print(jpg)
        self.label_5.setPixmap(jpg)
        jpg1 = QtGui.QPixmap("./cam.jpg").scaled(self.label_2.width(),
                                                 self.label_2.height())
        print(jpg)
        self.label_2.setPixmap(jpg1)

        f1 = open('predict_result.txt', 'a', encoding='utf-8')
        f1.write(self.curimg + " 预测为" + self.predictlabel + "   " + "置信度:" +
                 self.confi + '\n')
        f1.close()
        # os.remove("./temp/0/"+str(self.imgName).split('/')[-1])

    def previous(self):
        files = os.listdir(self.imgpath)
        print(files)
        self.img = self.imgName.split('/')[-1]
        id = files.index(self.img)
        print(self.img)
        if id != 0:
            print(files[id - 1])
            self.imgName = self.imgpath + files[id - 1]
            jpg = QtGui.QPixmap(self.imgName).scaled(self.label_6.width(),
                                                     self.label_6.height())
            self.label_6.setPixmap(jpg)

            jpg1 = QtGui.QPixmap(self.imgName).scaled(self.label_2.width(),
                                                      self.label_2.height())
            self.label_2.setPixmap(jpg1)

    def nextpic(self):
        files = os.listdir(self.imgpath)
        self.img = self.imgName.split('/')[-1]
        id = files.index(self.img)
        if id != (len(files) - 1):
            print(files[id + 1])
            self.imgName = self.imgpath + files[id + 1]
            jpg = QtGui.QPixmap(self.imgName).scaled(self.label_6.width(),
                                                     self.label_6.height())
            self.label_6.setPixmap(jpg)

            jpg1 = QtGui.QPixmap(self.imgName).scaled(self.label_2.width(),
                                                      self.label_2.height())
            self.label_2.setPixmap(jpg1)

    def save(self):
        import shutil
        root_path = os.getcwd()
        root_path += '/save/'
        if os.path.exists(root_path) == False:
            os.makedirs(root_path)
        shutil.copyfile("./cam.jpg",
                        './save/' + self.curimg.split('.')[0] + '.jpg')

    def origjpg(self):
        jpg1 = QtGui.QPixmap(self.imgName).scaled(self.label_2.width(),
                                                  self.label_2.height())
        self.label_2.setPixmap(jpg1)

    def heatmap(self):
        jpg1 = QtGui.QPixmap("./cam.jpg").scaled(self.label_2.width(),
                                                 self.label_2.height())
        self.label_2.setPixmap(jpg1)
