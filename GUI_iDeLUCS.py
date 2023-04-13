import sys
import time

sys.stdout.write(" ... loading libraries [5%]..........")
sys.stdout.flush()

import breeze_resources
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFile, QTextStream, QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtGui import QPixmap

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

sys.stdout.write("\r... loading libraries [20%]..........")
sys.stdout.flush()

from idelucs import LossFunctions, kmers_non_cython, PytorchUtils, ResNet, utils
from idelucs import models
from idelucs.utils import SummaryFasta, PlotPolygon, plot_confusion_matrix, \
                      compute_results
from idelucs.utils_GUI import define_ToolTips
import numpy as np
import torch
import random
import pandas as pd
import os
import csv
from idelucs.utils import label_features
sys.stdout.write("\r... loading libraries [50%]..........")
sys.stdout.flush()
import umap
sys.stdout.write("\r... loading libraries [75%]..........")
sys.stdout.flush()
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
plt.rcParams.update({'font.size': 8})
sys.stdout.write("\r... loading libraries [100%]..........")
sys.stdout.flush()


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=96):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)


class ParserThread(QThread):

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

    update_loading_progress = pyqtSignal(dict)
    parsing_error = pyqtSignal(str)
    def run(self):
        try:
            self.model.names, self.model.lengths, self.model.GT, self.model.cluster_dis = SummaryFasta(self.model.sequence_file, self.model.GT_file)
            stats = {"n_seq": len(self.model.lengths),
                     "min_len": np.min(self.model.lengths),
                     "max_len": np.max(self.model.lengths),
                     "avg_len": np.mean(self.model.lengths)}
            self.update_loading_progress.emit(stats)
        except Exception as e:
            self.parsing_error.emit(str(e))   


class WorkerThread(QThread):

    # Random Seeds for reproducibility.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args  

    update_progress = pyqtSignal(dict)
    update_coordinates = pyqtSignal(dict)
    update_status = pyqtSignal(str)

    def run(self):
        self.model.build_dataloader()
        self.model.predictions = []
        self.model.LOSS = []
        self.model.loss = 10
        
        for voter in range(self.args['n_voters']):
            self.model.net.apply(models.weights_init)
            self.model.epoch = 0

            for i in range(self.args['n_epochs']):
                if (i % 1)==0:
                    probabilities = self.model.calculate_probs()
                    self.update_coordinates.emit({"epoch":self.model.epoch, "probs":probabilities})

                loss = self.model.contrastive_training_epoch()
                self.update_progress.emit({"epoch":self.model.epoch, "loss":loss, "n_voters": voter})

                if self.isInterruptionRequested():
                    return

            y_pred, probabilities, latent = self.model.predict()
            self.model.LOSS.append(loss)

            if loss < self.model.loss:
                self.model.loss = loss
                self.model.latent = latent

            self.model.predictions.append(y_pred)

class ResultsThread(QThread):

    def __init__(self, model, n_voters):
        super().__init__()
        self.model = model
        self.n_voters = n_voters
    
    completed = pyqtSignal(dict)

    def run(self):
        y_pred, probabilities, latent = self.model.predict()

        if len(self.model.predictions) <= 1:
            y_pred, probabilities, latent = self.model.predict()
        else: # There is more than one assignment saved
           
            predictions = np.array(self.model.predictions)
            y_pred, probabilities = label_features(predictions, self.model.n_clusters)            
            latent = self.model.latent

        
        reducer = umap.UMAP()
        embedded = reducer.fit_transform(latent)

        if self.model.GT:
            
            unique_labels = list(np.unique(self.model.GT))
            numClasses = len(unique_labels)
            y = np.array(list(map(lambda x: unique_labels.index(x), self.model.GT)))

            results, ind = compute_results(y_pred, latent, y)
            d = {}
            for i, j in ind:
                d[i] = j
            w = np.zeros((numClasses, self.model.n_clusters), dtype=np.int64)
            for i in range(y.shape[0]):
                w[y[i], d[y_pred[i]]] += 1
            print(w)
            print(d)

            self.completed.emit({"assignments":y_pred, "probabilities":probabilities, "latent": embedded, 
                                 "w": w, "unique_labels":unique_labels, "results":results})
        else:
            results, ind = compute_results(y_pred, latent)
            self.completed.emit({"assignments":y_pred, "probabilities":probabilities, "latent": embedded,  "results":results})
            

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setStyleSheet("background-color: white;")
        
        MainWindow.resize(1280, 720)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 157))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 157))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 157))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))    ######################(238, 238, 236)#############################
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        MainWindow.setPalette(palette)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(40, 15, 1213, 653))  #40, 33, 1213, 653
        self.tabWidget.setStyleSheet("QTabBar::tab { height: 30px; width: 100px}") #; background: 'white'}")


        self.LOGO = QtWidgets.QLabel(MainWindow)
        pixmap = QPixmap('itallics_iDeLUCS_logo.png') #logo.png
        self.LOGO.setPixmap(pixmap)
        self.LOGO.setGeometry(QtCore.QRect(1052,   4,  200,   51)) #1050,   4,  200,   51
        self.LOGO.setScaledContents(True)

        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.tabWidget.setPalette(palette)
        self.tabWidget.setObjectName("tabWidget")
        self.Settings_tab = QtWidgets.QWidget()
        self.Settings_tab.setObjectName("Settings_tab")
        
        self.Reset_Button = QtWidgets.QPushButton(self.Settings_tab)
        self.Reset_Button.setGeometry(QtCore.QRect(566, 506, 200, 26)) #526, 506, 200, 26
        self.Reset_Button.setObjectName("Reset_Button")
        
        self.Submit_Button = QtWidgets.QPushButton(self.Settings_tab)
        self.Submit_Button.setGeometry(QtCore.QRect(593, 553, 146,  46)) #553, 553, 146,  46
        self.Submit_Button.setObjectName("Submit_Button")
        
        
        
        self.Settings_toolBox = QtWidgets.QToolBox(self.Settings_tab)
        self.Settings_toolBox.setGeometry(QtCore.QRect(300, 66, 693, 353)) #240, 66, 693, 353
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 157))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 157))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(246, 246, 245))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 159, 157))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(119, 119, 118))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(238, 238, 236))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        

        #self.LOGO.resize(pixmap.width(), pixmap.height())

        
        self.Settings_toolBox.setPalette(palette)
        self.Settings_toolBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Settings_toolBox.setAutoFillBackground(False)
        self.Settings_toolBox.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.Settings_toolBox.setObjectName("Settings_toolBox")
        self.basic_page = QtWidgets.QWidget()
        self.basic_page.setGeometry(QtCore.QRect(0, 0, 921, 423))
        self.basic_page.setObjectName("basic_page")
        self.label = QtWidgets.QLabel(self.basic_page)
        self.label.setGeometry(QtCore.QRect(33,  16, 207,  20))  #(33,  16, 207,  20)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.basic_page)
        self.label_2.setGeometry(QtCore.QRect(33,  50, 247,  20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.basic_page)
        self.label_3.setGeometry(QtCore.QRect(33,  86, 274,  27))
        self.label_3.setObjectName("label_3")

        self.label_k = QtWidgets.QLabel(self.basic_page)
        self.label_k.setGeometry(QtCore.QRect(34, 120, 274,  27)) #(34, 186, 274,  27)
        self.label_k.setObjectName("label_k")

        self.ChooseSeq_Button = QtWidgets.QPushButton(self.basic_page)
        self.ChooseSeq_Button.setGeometry(QtCore.QRect(533,   6, 127,  20))
        self.ChooseSeq_Button.setCheckable(False)
        self.ChooseSeq_Button.setFlat(True)
        self.ChooseSeq_Button.setObjectName("ChooseSeq_Button")
        self.FASTA_fname = [""]

        self.ChooseGT_Button = QtWidgets.QPushButton(self.basic_page)
        self.ChooseGT_Button.setGeometry(QtCore.QRect(533,  46, 127,  20))
        self.ChooseGT_Button.setCheckable(False)
        self.ChooseGT_Button.setFlat(True)
        self.ChooseGT_Button.setObjectName("ChooseGT_Button")
        
        self.input_n_clusters = QtWidgets.QSpinBox(self.basic_page)
        self.input_n_clusters.setGeometry(QtCore.QRect(566,  86,  80,  27))
        self.input_n_clusters.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.input_n_clusters.setMinimum(1)
        self.input_n_clusters.setMaximum(50)
        self.input_n_clusters.setValue(5)
        self.input_n_clusters.setObjectName("input_n_clusters")

        self.input_k = QtWidgets.QSpinBox(self.basic_page)
        self.input_k.setGeometry(QtCore.QRect(566, 120,  80,   2)) #(566, 186,  80,   2)
        self.input_k.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.input_k.setMinimum(4)
        self.input_k.setMaximum(6)
        self.input_k.setValue(6)
        self.input_k.setObjectName("input_k")
        
        self.Settings_toolBox.addItem(self.basic_page, "")

        self.advanced_page = QtWidgets.QWidget()
        self.advanced_page.setGeometry(QtCore.QRect(0,   0, 614, 282))
        self.advanced_page.setObjectName("advanced_page")

        self.label_5 = QtWidgets.QLabel(self.advanced_page)
        self.label_5.setGeometry(QtCore.QRect(33,  6, 274,  27))
        self.label_5.setObjectName("label_5")

        self.input_n_epochs = QtWidgets.QSpinBox(self.advanced_page)
        self.input_n_epochs.setGeometry(QtCore.QRect(566, 6,  94,  27))
        self.input_n_epochs.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.input_n_epochs.setMinimum(1)
        self.input_n_epochs.setMaximum(300)
        self.input_n_epochs.setProperty("value", 50)
        self.input_n_epochs.setObjectName("input_n_epochs")
        
        self.label_4 = QtWidgets.QLabel(self.advanced_page)
        self.label_4.setGeometry(QtCore.QRect(33, 43, 274,  27))
        self.label_4.setObjectName("label_4")

        
        self.input_n_mimics = QtWidgets.QSpinBox(self.advanced_page)
        self.input_n_mimics.setGeometry(QtCore.QRect(566, 43,  94,  27))
        self.input_n_mimics.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.input_n_mimics.setMinimum(2)
        self.input_n_mimics.setMaximum(10)
        self.input_n_mimics.setProperty("value", 3)
        self.input_n_mimics.setObjectName("input_n_mimics")


        self.label_7 = QtWidgets.QLabel(self.advanced_page)
        self.label_7.setGeometry(QtCore.QRect(33,  80, 247,  20))
        self.label_7.setObjectName("label_7")

        self.input_lambda = QtWidgets.QLineEdit(self.advanced_page)
        self.input_lambda.setGeometry(QtCore.QRect(566,  83,  94,  27))
        self.input_lambda.setObjectName("input_lambda")
        self.input_lambda.insert("2.8")

        self.label_9 = QtWidgets.QLabel(self.advanced_page)
        self.label_9.setGeometry(QtCore.QRect(33,  116, 207,  20))
        self.label_9.setObjectName("label_9")


        self.label_8 = QtWidgets.QLabel(self.advanced_page)
        self.label_8.setGeometry(QtCore.QRect(33, 150, 247,  20))
        self.label_8.setObjectName("label_8")


        self.label_17 = QtWidgets.QLabel(self.advanced_page)
        self.label_17.setGeometry(QtCore.QRect(33, 183, 207,  20))
        self.label_17.setObjectName("label_17")


        self.input_batch_sz = QtWidgets.QSpinBox(self.advanced_page)
        self.input_batch_sz.setGeometry(QtCore.QRect(566,  116,  94,  27))
        self.input_batch_sz.setMaximum(600)
        self.input_batch_sz.setObjectName("input_batch_sz")
        self.input_batch_sz.setProperty("value", 256)

        self.input_weight = QtWidgets.QLineEdit(self.advanced_page)
        self.input_weight.setGeometry(QtCore.QRect(566, 150,  94,  27))
        self.input_weight.setObjectName("input_weight")
        self.input_weight.insert("0.25")

       
        self.input_scheduler = QtWidgets.QComboBox(self.advanced_page)
        self.input_scheduler.setGeometry(QtCore.QRect(566,   183,  96,  27))
        self.input_scheduler.setObjectName("input_scheduler")
        self.input_scheduler.addItem("")
        self.input_scheduler.addItem("")
        self.input_scheduler.addItem("")


        self.Settings_toolBox.addItem(self.advanced_page, "")
        self.tabWidget.addTab(self.Settings_tab, "")
        self.GT_fname = None

        ## Definition of the Running Tab

        self.Running_tab = QtWidgets.QWidget()
        self.Running_tab.setObjectName("Running_tab")
        self.Running_tab.setStyleSheet("background-color: white;")
        self.progressBar = QtWidgets.QProgressBar(self.Running_tab)
        

        self.progressBar.setGeometry(QtCore.QRect(366, 420, 800,  50)) #366, 433, 800,  27
        self.progressBar.setProperty("value", 0)
        self.progressBar.setStyleSheet("background-color: white;")
        self.progressBar.setObjectName("progressBar")

        ### Definition of FASTA summary
        highlight_font = QtGui.QFont()
        highlight_font.setBold(True)
        highlight_font.setPointSize(13)
        self.Summary = QtWidgets.QLabel(self.Running_tab)
        self.Summary.setGeometry(QtCore.QRect(46,  305, 174,  34)) #46,  15, 174,  34
        self.Summary.setFont(highlight_font)
        self.summary_n_seq = QtWidgets.QLabel(self.Running_tab)
        self.summary_n_seq.setGeometry(QtCore.QRect(56,  343, 233,  20)) #46,  53, 233,  20
        self.summary_n_seq.setObjectName("summary_n_seq")
        self.summary_min_len = QtWidgets.QLabel(self.Running_tab)
        self.summary_min_len.setGeometry(QtCore.QRect(56, 376, 233,  20)) #46, 86, 233,  20
        self.summary_min_len.setObjectName("summary_min_len")
        self.summary_max_len = QtWidgets.QLabel(self.Running_tab)
        self.summary_max_len.setGeometry(QtCore.QRect(56, 409, 233,  20)) #46, 119, 233,  20
        self.summary_max_len.setObjectName("summary_max_len")
        
        self.summary_avg_len = QtWidgets.QLabel(self.Running_tab)
        self.summary_avg_len.setGeometry(QtCore.QRect(56, 440, 233,  20)) #46, 150, 233,  20
        self.summary_avg_len.setObjectName("summary_avg_len")

        ### Definition of Training Parameters
        highlight_font = QtGui.QFont()
        highlight_font.setBold(True)
        highlight_font.setPointSize(13)
        self.Parameter_Summary = QtWidgets.QLabel(self.Running_tab)
        self.Parameter_Summary.setGeometry(QtCore.QRect(46, 13, 174, 34))
        self.Parameter_Summary.setFont(highlight_font)
        self.Parameter_Summary.setText("Settings Summary")

        self.summary_FASTA = QtWidgets.QLabel(self.Running_tab)
        self.summary_FASTA.setGeometry(QtCore.QRect(56, 47, 250, 20))

        self.summary_n_cluster = QtWidgets.QLabel(self.Running_tab)
        self.summary_n_cluster.setGeometry(QtCore.QRect(56, 79, 233, 20))

        self.summary_mimics = QtWidgets.QLabel(self.Running_tab)
        self.summary_mimics.setGeometry(QtCore.QRect(56, 111, 233, 20))

        self.summary_batch_sz = QtWidgets.QLabel(self.Running_tab)
        self.summary_batch_sz.setGeometry(QtCore.QRect(56, 143, 233, 20))

        self.summary_optmizer = QtWidgets.QLabel(self.Running_tab)
        self.summary_optmizer.setGeometry(QtCore.QRect(56, 175, 233, 20))

        self.summary_lamda = QtWidgets.QLabel(self.Running_tab)
        self.summary_lamda.setGeometry(QtCore.QRect(56, 207, 233, 20))

        self.summary_GT = QtWidgets.QLabel(self.Running_tab)
        self.summary_GT.setGeometry(QtCore.QRect(56, 239, 250, 20))

        self.summary_k = QtWidgets.QLabel(self.Running_tab)
        self.summary_k.setGeometry(QtCore.QRect(56, 271, 233, 20))


        #---------------------------------------------------------
        self.status_info = QtWidgets.QLabel(self.Running_tab)
        self.status_info.setGeometry(QtCore.QRect(375, 380, 180, 27))  # 366, 400, 180,  27
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.status_info.setFont(font)
        self.status_info.setObjectName("progress_info")
        
        self.progress_info = QtWidgets.QLabel(self.Running_tab)
        self.progress_info.setGeometry(QtCore.QRect(425, 380, 180,  27)) #366, 400, 180,  27
        font = QtGui.QFont()
        font.setPointSize(10)
        self.progress_info.setFont(font)
        self.progress_info.setObjectName("progress_info")

        self.Training_Button = QtWidgets.QPushButton(self.Running_tab)
        self.Training_Button.setGeometry(QtCore.QRect(553, 533, 147,  47))
        self.Training_Button.setObjectName("Training_Button")
        self.Stop_Training_Button = QtWidgets.QPushButton(self.Running_tab)
        self.Stop_Training_Button.setGeometry(QtCore.QRect(886, 533, 147,  47))
        self.Stop_Training_Button.setObjectName("Stop_Training_Button")
        self.tabWidget.addTab(self.Running_tab, "")
        self.tabWidget.setTabEnabled(1,False)
        
        ## Define Results Tab
        self.Results_tab = QtWidgets.QWidget()
        self.Results_tab.setObjectName("Results_tab")
        self.save_results = QtWidgets.QPushButton(self.Results_tab)
        self.save_results.setGeometry(QtCore.QRect(200, 566, 123,  32))
        self.save_results.setObjectName("save_results")

        self.UMAP_Button = QtWidgets.QRadioButton(self.Results_tab)
        self.UMAP_Button.setGeometry(QtCore.QRect(740, 500, 150,  32))

        self.CM_Button = QtWidgets.QRadioButton(self.Results_tab)
        self.CM_Button.setGeometry(QtCore.QRect(930, 500, 300,  32))

        self.DB_Index = QtWidgets.QLabel(self.Results_tab)
        self.DB_Index.setGeometry(QtCore.QRect(740, 500, 200,  32))
        

        self.Silhouette = QtWidgets.QLabel(self.Results_tab)
        self.Silhouette.setGeometry(QtCore.QRect(950, 500, 300,  32))

        self.DB_Index.setVisible(False)
        self.Silhouette.setVisible(False)


        self.tabWidget.addTab(self.Results_tab, "")
        self.tabWidget.setTabEnabled(2, False)

        #--------------------------------------------------------- #---------------------------------------------------------
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1301, 39))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        

        self.Submit_Button.clicked.connect(self.parse)
        self.Reset_Button.clicked.connect(self.reset_to_default)
        self.Training_Button.clicked.connect(self.train)
        self.ChooseSeq_Button.clicked.connect(self.get_FASTA_file)
        self.ChooseGT_Button.clicked.connect(self.get_GT_file)
        self.Stop_Training_Button.clicked.connect(self.stop_training)
        self.save_results.clicked.connect(self.save_results_file)
        self.UMAP_Button.clicked.connect(self.plot_UMAP)
        self.CM_Button.clicked.connect(self.plot_CM)
        self.args = {}

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.Settings_toolBox.setCurrentIndex(0)
        self.progressBar.setValue(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        define_ToolTips(self)
        self.Settings_toolBox.setFont(highlight_font)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Submit_Button.setText(_translate("MainWindow", "Submit"))
        self.Reset_Button.setText(_translate("MainWindow", "Reset Parameters"))

        self.label.setText(_translate("MainWindow", "Sequence File:"))
        self.label_2.setText(_translate("MainWindow", "Ground Truth File (Optional):"))
        self.label_5.setText(_translate("MainWindow", "Training Epochs:"))
        self.label_k.setText(_translate("MainWindow","k-mer length:"))
        self.label_3.setText(_translate("MainWindow", "Number of Clusters:"))
        self.label_4.setText(_translate("MainWindow", "Number of mimics:"))
        self.ChooseSeq_Button.setText(_translate("MainWindow", "Choose File"))
        self.ChooseGT_Button.setText(_translate("MainWindow", "Choose File "))
        self.Settings_toolBox.setItemText(self.Settings_toolBox.indexOf(self.basic_page), _translate("MainWindow", '\u25BA'+" Basic"))

        #self.label_6.setText(_translate("MainWindow", "Model Optimizer:"))
        self.label_7.setText(_translate("MainWindow", "Balance Hyperparameter:"))
        self.label_8.setText(_translate("MainWindow", "Add Training Weight:"))
        self.input_scheduler.setItemText(0, _translate("MainWindow", "None"))
        self.input_scheduler.setItemText(1, _translate("MainWindow", "Triangle"))
        self.input_scheduler.setItemText(2, _translate("MainWindow", "Plateau"))

        self.label_9.setText(_translate("MainWindow", "Batch Size:"))
        self.label_17.setText(_translate("MainWindow", "Scheduler"))
        self.Settings_toolBox.setItemText(self.Settings_toolBox.indexOf(self.advanced_page), _translate("MainWindow",  '\u25BA'+ " Advanced"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Settings_tab), _translate("MainWindow", "Settings"))
        self.Summary.setText(_translate("MainWindow", "Dataset Summary"))
        self.summary_n_seq.setText(_translate("MainWindow", "Noclear. Sequences:"))
        self.summary_min_len.setText(_translate("MainWindow", "Min. Length:"))
        self.summary_max_len.setText(_translate("MainWindow", "Max. Length:"))
        self.summary_avg_len.setText(_translate("MainWindow", "Avg. Length:"))
        self.progress_info.setText(_translate("MainWindow", "Training Progress"))
        self.status_info.setText(_translate("MainWindow", "Status:"))
        self.Training_Button.setText(_translate("MainWindow", "Start"))
        self.Stop_Training_Button.setText(_translate("MainWindow", "Stop"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Running_tab), _translate("MainWindow", "Training"))
        self.save_results.setText(_translate("MainWindow", "Save Results"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Results_tab), _translate("MainWindow", "Results"))
        self.UMAP_Button.setText("Representation")
        self.CM_Button.setText("Confusion Matrix")
        
        self.UMAP_Button.setVisible(False)
        self.CM_Button.setVisible(False)
        self.Training_Button.setVisible(False)
        self.Stop_Training_Button.setVisible(False)

    def parse(self):

        self.args['sequence_file'] = self.FASTA_fname[0]
        self.args['n_clusters'] = int(self.input_n_clusters.text())

        self.args['n_epochs'] = int(self.input_n_epochs.text())
        self.args['n_mimics'] = int(self.input_n_mimics.text())
        self.args['batch_sz'] = int(self.input_batch_sz.text())

        self.args['GT_file'] = self.GT_fname
        self.args['k'] = int(self.input_k.text())

        self.args['optimizer'] = "RMSprop"
        self.args['batch_sz'] = int(self.input_batch_sz.text())
        self.args['lambda'] = float(self.input_lambda.text())
        self.args['weight'] =  float(self.input_weight.text())
        self.args['n_voters'] = 5
        self.args["lr"] = 1e-3  #5e-4
        self.args["model_size"] = "linear"  #5e-4
        self.args['scheduler'] = str(self.input_scheduler.currentText())

        if self.args['sequence_file'] == "":
            QMessageBox.critical(self.centralwidget, "Error!", "Wowza! \n Please Select a Fasta File")

        elif self.args['weight'] > 1.0 or self.args['weight'] < 0.1:
            QMessageBox.critical(self.centralwidget, "Error!", "Training weight bust be in [0.1, 1.0]")

        else:

            #Display Training Parameters
            if len(self.args['sequence_file'].split("/")[-1]) > 15:
                fname = self.args['sequence_file'].split("/")[-1][0:15]  + ' ...'
            else:
                fname = self.args['sequence_file'].split("/")[-1]

            self.summary_FASTA.setText("FASTA file:\t" + fname)

            if self.args['GT_file'] and len(str(self.args['GT_file']).split("/")[-1]) > 15:
                gtname = str(self.args['GT_file']).split("/")[-1][0:15]  + ' ...'
            else:
                gtname = "None"
            
            self.summary_GT.setText("GT:\t \t" + gtname)
            
            self.summary_k.setText("k:\t \t" + str(self.args['k']))
            self.summary_mimics.setText("No. Mimics: \t" + str(self.args['n_mimics']))
            self.summary_n_cluster.setText("No. Clusters: \t" + str(self.args['n_clusters']))
            self.summary_batch_sz.setText("Batch Size: \t" + str(self.args['batch_sz']))
            self.summary_optmizer.setText("Optimizer: \t" + self.args['optimizer'])
            self.summary_lamda.setText("Balance:   \t" + str(self.args['lambda']))


            #Parse Fasta File

            self.training_loss = []
            self.model = models.IID_model(self.args)
            self.worker = ParserThread(self.model, self.args)
            self.worker.start()
            self.worker.parsing_error.connect(self.evt_display_parsing_error)
            self.worker.update_loading_progress.connect(self.evt_update_loading_progress)

            self.tabWidget.setTabEnabled(1,True)

            self.progress_info.setText("Parsing Fasta File ....")
            self.Training_Button.setEnabled(True)

            self.display_polygon = MplCanvas(self.Running_tab, width=5, height=4, dpi=96)
            self.display_polygon.axes.axis('off')
            self.display_polygon.axes.set_title('Training Progress (Epoch 0)')
            self.display_polygon.setGeometry(QtCore.QRect(360,  20, 367, 280))
            self.display_polygon.setObjectName("display_polygon")
            

            self.display_training = MplCanvas(self.Running_tab, width=5, height=4, dpi=96)
            self.display_training.setGeometry(QtCore.QRect(733,  20, 440, 280))  #(733,  20, 440, 280)
            self.display_training.axes.set_xlim(1,self.args['n_epochs'])
            self.display_training.axes.grid(True)
            self.display_training.axes.set_title("Learning Curve")
            self.display_training.axes.set_xlabel("Epoch")
            self.display_training.axes.set_ylabel("Training Loss")
            toolbar = NavigationToolbar(self.display_training, self.Running_tab)
            #toolbar.setMaximumWidth(900) #700
            #toolbar.setMaximumHeight(100) #60
            #toolbar.setStyleSheet("QToolBar { border: none }")


            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(toolbar)
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(0)

            #Create a placeholder widget to hold our toolbar and canvas.
            #Graphic Container
            self.widget_display_training = QtWidgets.QWidget(self.Running_tab)
            self.widget_display_training.setGeometry(QtCore.QRect(733, 300, 600,  40)) #800, 300, 400, 40
            self.widget_display_training.setObjectName("display_training")
            self.widget_display_training.setLayout(layout)

            self.tabWidget.setCurrentIndex(1)
          
    def train(self):
        self.training_plot = [None] * self.args["n_voters"]
        self.worker = WorkerThread(self.model, self.args)
        self.worker.start()
        self.progress_info.setText("Training in Progress...")
        self.progressBar.setValue(5)
        self.worker.finished.connect(self.evt_training_finished)
        self.worker.update_progress.connect(self.evt_update_progress)
        self.worker.update_coordinates.connect(self.evt_plot_progress)

        self.Training_Button.setEnabled(False)
        self.DB_Index.setVisible(False)
        self.Silhouette.setVisible(False)
        


    def stop_training(self):
        self.worker.requestInterruption()
        self.worker.wait(1)


    def evt_show_results(self, info):
        length = len(self.model.names)

        self.Results_Table = QtWidgets.QTableWidget(self.Results_tab)
        self.Results_Table.setGeometry(QtCore.QRect(40,  33, 500, 500))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Results_Table.setFont(font)
        self.Results_Table.setObjectName("Results_Table")
        self.Results_Table.setColumnCount(3)
        self.Results_Table.setRowCount(length)

        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.Results_Table.setItem(0, 2, item)
        self.Results_Table.horizontalHeader().setDefaultSectionSize(145)

        item = self.Results_Table.verticalHeaderItem(0)
        item.setText("1")
        item = self.Results_Table.horizontalHeaderItem(0)
        item.setText("Seq. ID")
        item = self.Results_Table.horizontalHeaderItem(1)
        item.setText( "Cluster \n Assignment")
        item = self.Results_Table.horizontalHeaderItem(2)
        item.setText("Confidence \n Score")
        __sortingEnabled = self.Results_Table.isSortingEnabled()
        self.Results_Table.setSortingEnabled(False)
        self.Results_Table.setSortingEnabled(__sortingEnabled)

        for i in range(length):
            #print(self.model.dataset_info['names'][i], info['assignments'][i], info['probabilities'][i])
            
            item = QtWidgets.QTableWidgetItem()
            self.Results_Table.setItem(i, 0, item)
            item = QtWidgets.QTableWidgetItem()
            self.Results_Table.setItem(i, 1, item)
            item = QtWidgets.QTableWidgetItem()
            self.Results_Table.setItem(i, 2, item)

            item = self.Results_Table.item(i, 0)
            item.setText(self.model.names[i])
            item = self.Results_Table.item(i, 1)
            item.setText(str(info['assignments'][i]))
            item = self.Results_Table.item(i, 2)
            item.setText(str(100*info['probabilities'][i])[:5])

        

        self.display_results = MplCanvas(self.Results_tab, width=5, height=4, dpi=96)
        self.display_results.setGeometry(QtCore.QRect(650, 33, 500, 400))
        self.embedded = info['latent']
        self.assingments = info['assignments']
        self.results = info['results']

        if self.GT_fname:
            self.UMAP_Button.setVisible(True)
            self.CM_Button.setVisible(True)

            self.confussion_matrix = info['w']
            self.unique_labels = info['unique_labels']
            plot_confusion_matrix(self.confussion_matrix, self.unique_labels, pairs=None, ax=self.display_results.axes, normalize=False)
            
        else:
            self.UMAP_Button.setVisible(False)
            self.CM_Button.setVisible(False) 

            self.DB_Index.setText(f"Davies-Bouldin: {self.results['Davies-Boulding']}")
            self.Silhouette.setText(f"Silhouette: {self.results['Silhouette-Score']}")

            self.DB_Index.setVisible(True)
            self.Silhouette.setVisible(True)
            self.DB_Index.setText(f"Davies-Bouldin: {round(self.results['Davies-Boulding'],3)}")
            self.Silhouette.setText(f"Silhouette: {round(self.results['Silhouette-Score'],3)}")
            self.Silhouette.setVisible(True)


            scatter = self.display_results.axes.scatter(self.embedded [:, 0], self.embedded [:, 1], s=2, marker='.', c=info['assignments'].astype(np.int32))
            #self.display_results.axes.axis('off')
            self.display_results.axes.set_title('Latent Representation (UMAP)')
            self.display_results.axes.set_title("Representation of the Latent Space")
            self.display_results.axes.set_xlabel("UMAP 1")
            self.display_results.axes.set_ylabel("UMAP 2")
            # produce a legend with the unique colors from the scatter
            legend = self.display_results.axes.legend(*scatter.legend_elements(), title="Assignments")
            self.display_results.axes.add_artist(legend)
            #sns.scatterplot(ax=ax, x=embedding[:,0], y=embedding[:, 1], hue=colors, s=2, palette=['red','gray','green'], legend='full')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)	
            plt.tight_layout()
            #self.display_results.axes.legend(legend)

        toolbar = NavigationToolbar(self.display_results, self.Results_tab)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a placeholder widget to hold our toolbar and canvas.
        # Graphic Container
        self.widget_display_results = QtWidgets.QWidget(self.Results_tab)
        self.widget_display_results.setGeometry(QtCore.QRect(660, 440, 500, 40))  # 800, 300, 400, 40
        self.widget_display_results.setObjectName("display_results")
        self.widget_display_results.setLayout(layout)

        

        self.progress_info.setText("Results are ready")
        self.tabWidget.setTabEnabled(2, True)

    def evt_update_loading_progress(self, stats):
        self.summary_n_seq.setText( f'No. Sequences: \t {stats["n_seq"]:,}')
        self.summary_min_len.setText(f'Min. Length: \t {stats["min_len"]:,}')
        self.summary_max_len.setText(f'Max. Length: \t {stats["max_len"]:,}')
        self.summary_avg_len.setText(f'Avg. Length: \t {round(stats["avg_len"],2):,}')
        self.summary_avg_len.setObjectName("summary_avg_len")
        #if stats['gt_summary']:
        #    self.summary_GT.setText("Dataset Label Summary: \n{}".format(stats['gt_summary']))

        self.progress_info.setText("Ready to train!")
        self.Training_Button.setVisible(True)
        self.Stop_Training_Button.setVisible(True)


    def evt_training_finished(self):
        self.progress_info.setText("Training Complete")
        QMessageBox.information(self.centralwidget, "Done!", "Training Finished!")

        self.results_worker = ResultsThread(self.model, self.args['n_voters'])
        self.results_worker.start()
        self.results_worker.completed.connect(self.evt_show_results)
        self.progress_info.setText("Preparing results ...")

    def evt_display_parsing_error(self, msg):
        self.tabWidget.setTabEnabled(1, False)
        self.tabWidget.setCurrentIndex(0)
        QMessageBox.critical(self.centralwidget, "Check your Files", "Wowza! \n" + msg)


    def evt_update_progress(self, info):

        if self.training_plot[info["n_voters"]] is None:
            self.training_loss=[info["loss"]]
            self.display_training.axes.set_ylim(self.training_loss[0]-1.25, self.training_loss[0]+0.25)
            self.training_plot[info["n_voters"]], = self.display_training.axes.plot(np.arange(info["epoch"])+1, self.training_loss)
            self.training_plot[info["n_voters"]].set_label(f'Model {info["n_voters"] + 1} ')
            self.display_training.axes.legend()

        else:
            self.training_loss.append(info["loss"])
            self.training_plot[info["n_voters"]].set_ydata(self.training_loss)
            self.training_plot[info["n_voters"]].set_xdata(np.arange(info["epoch"])+1)
        
        self.display_training.draw()
        self.progressBar.setValue(10 + int(90 * info['epoch'] /self.args['n_epochs']))

    def evt_plot_progress(self, info):
        self.display_polygon.axes.clear()
        PlotPolygon(info['probs'], self.args['n_clusters'], 
                    self.display_polygon.axes, 
                    "Epoch {}".format(info['epoch']))
        self.display_polygon.draw()
              
    def get_FASTA_file(self):
        # Check that is not empty and that is a FASTA file
        self.FASTA_fname = QtWidgets.QFileDialog.getOpenFileName()
        short_name = self.FASTA_fname[0].split('/')[-1]
        #Display Training Parameters
        if len(short_name) > 12:
            short_name = short_name[0:12]  + ' ...'
        self.ChooseSeq_Button.setText(short_name)

    def get_GT_file(self):
        # Check that is not empty and that has the appropriate format   !!!!!!!!!!!!!
        self.GT_fname = QtWidgets.QFileDialog.getOpenFileName()[0]
        short_name = self.GT_fname.split('/')[-1]
        if len(short_name) > 12:
            short_name = short_name[0:12]  + ' ...'
        self.ChooseGT_Button.setText(short_name)

    def reset_to_default(self):
        self.input_n_epochs.setValue(30)
        self.input_n_mimics.setValue(3) 
        self.input_n_clusters.setValue(5)
        self.input_k.setValue(6)
        #self.input_optimizer.setValue("RMSprop")
        self.input_batch_sz.setValue(256)
        self.input_lambda.setText("2.8")
        self.input_weight.setText("0.25")
        self.ChooseGT_Button.setText('Choose File')
        self.GT_fname = None
        self.ChooseSeq_Button.setText('Choose File')
        self.GT_fname = ""

    
    def plot_UMAP(self):
        self.CM_Button.setChecked(False)
        self.display_results.axes.clear()

        scatter = self.display_results.axes.scatter(self.embedded [:, 0], self.embedded [:, 1], s=2, marker='.', c=self.assingments.astype(np.int32))
        self.display_results.axes.set_title('Latent Representation (UMAP)')

        self.display_results.axes.set_title('Latent Representation (UMAP)')
        self.display_results.axes.set_title("Representation of the Latent Space")
        self.display_results.axes.set_xlabel("UMAP 1")
        self.display_results.axes.set_ylabel("UMAP 2")
        

        # produce a legend with the unique colors from the scatter
        legend = self.display_results.axes.legend(*scatter.legend_elements(), title="Assignments") # , loc="lower left")
        self.display_results.axes.add_artist(legend)
        plt.tight_layout()
        self.display_results.draw()

    def plot_CM(self):
        self.UMAP_Button.setChecked(False)
        self.display_results.axes.clear()
        plot_confusion_matrix(self.confussion_matrix, self.unique_labels, pairs=None, ax=self.display_results.axes, normalize=False)
        self.display_results.draw()
    
    def save_results_file(self):
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(self.Results_tab, "Save Folder")
        
        now = time.asctime()
        time_stamp = now.split(' ')
        hour = now.split(' ')[3]
        time_stamp[3] = '-'.join(hour.split(':'))
        time_stamp = '_'.join(time_stamp[1:4])

        folder_name = f'{folder_name}/Results'

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        
        if not os.path.isdir(f'{folder_name}/{time_stamp}'):
            os.mkdir(f'{folder_name}/{time_stamp}')
        
        filename = f'{folder_name}/{time_stamp}/assignments.tsv'
        
        if filename:
            with open(filename, 'w') as handle:
                writer = csv.writer(handle, delimiter='\t')
                for row in range(self.Results_Table.rowCount()):
                    rowdata = []
                    for column in range(self.Results_Table.columnCount()):
                        item = self.Results_Table.item(row, column)
                        if item is not None:
                            rowdata.append(item.text())
                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)


        df = pd.Series(self.results, name='Value')
        df.to_csv(f'{folder_name}/{time_stamp}/metrics.tsv',sep='\t')



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)


    # set stylesheet
    file = QFile("dist/light/stylesheet.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())