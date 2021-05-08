# 定义所有的弹出对话框
from PyQt5.QtWidgets import QDialog, QLabel, QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore


def connect_us_dialog(self):
    dialog = QDialog()
    dialog.setFixedSize(500, 100)
    dialog.setWindowTitle("联系我们")
    # 隐藏问号
    dialog.setWindowFlags(QtCore.Qt.WindowMinMaxButtonsHint
                          | QtCore.Qt.WindowCloseButtonHint)
    dialog.setStyleSheet("background-color:rgb(100, 100, 100)")
    dialog.setWindowIcon(QIcon('./config_images/logo1.png'))
    label_1 = QLabel("我们的官网是xmu-smartdsp.github.io!", dialog)
    label_1.setFixedSize(500, 100)
    label_1.setAlignment(QtCore.Qt.AlignCenter)
    label_1.setStyleSheet("color:white;font:18px;")

    label_1.show()
    dialog.exec_()
