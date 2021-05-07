import sys
import win
from PyQt5 import QtGui
from PyQt5 import QtWidgets, QtCore, Qt
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QPainter, QFont, QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget


class Ui_Launch_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(900, 600)
        self.setWindowTitle('COVID-19 Analysis System')
        self.setWindowIcon(QIcon('./config_images/logo1.png'))
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(425, 510, 85, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Enter")
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     background-color : white;
                     font: bold;
                     border-color: gray;
                     color:red;
                     border-width: 2px;
                     border-radius: 10px;
                     height : 14px;
                     border-style: outset;
                     font : 18px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     height : 14px;
                     border-style: outset;
                     font : 18px;}
                     ''')

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap('./config_images/first_edition.jpg')
        painter.drawPixmap(self.rect(), pixmap)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     mainWindow = Ui_Launch_Window()
#     mainWindow.show()
#     sys.exit(app.exec_())
