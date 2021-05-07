import sys
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
import sys
import win
from launch_window import Ui_Launch_Window

sys.setrecursionlimit(100000)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Launching the Window
    launchWindows = Ui_Launch_Window()
    launchWindows.show()

    MainWindow = QMainWindow()
    ui = win.Ui_smartdsp()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle('COVID-19 Analysis System')
    MainWindow.setWindowIcon(QIcon('./config_images/logo1.png'))
    MainWindow.setStyleSheet('background-color: black')

    launchWindows.setFixedSize(900, 600)
    MainWindow.setFixedSize(800, 550)
    # Connect to another Window
    launchWindows.pushButton.clicked.connect(
        lambda: {launchWindows.close(),
                 MainWindow.show()})

    imgName = ui.pushButton.clicked.connect(ui.openimage)
    ui.pushButton_2.clicked.connect(ui.predict)
    ui.pushButton_3.clicked.connect(ui.previous)
    ui.pushButton_4.clicked.connect(ui.nextpic)
    ui.pushButton_5.clicked.connect(ui.save)
    ui.pushButton_8.clicked.connect(ui.origjpg)
    ui.pushButton_9.clicked.connect(ui.heatmap)

    sys.exit(app.exec_())

    input('Press <Enter>')
