#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import logging

from datetime import datetime
from math import sqrt
from scipy.misc import imread

from PyQt5.QtCore import Qt, QMutex, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPainter, QDoubleValidator, QIntValidator, QKeySequence)
from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox,
    QGridLayout, QCheckBox, QVBoxLayout, QApplication, QShortcut,
    QSlider, QDoubleSpinBox, QWidget, QFileDialog, QScrollArea,
    QMessageBox, QTabWidget,
    )

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

from slm import version
from slm.ext.czernike import RZern


"""SLM - spatial light modulator (SLM) controller.
"""


class Pupil():

    def __init__(self, holo):
        self.log = logging.getLogger(self.__class__.__name__)
        self.holo = holo

        self.rzern = None
        self.pupil_xy = [0.0, 0.0]
        self.pupil_rho = 50.0
        self.angle_xy = [0.0, 0.0]
        self.aberration = np.zeros((15, 1))
        self.mask2d_on = 0.0
        self.mask2d_sign = 1.0
        self.mask3d_on = 0.0
        self.mask3d_radius = 0.6
        self.mask3d_height = 1.0

    def parameters2dict(self):
        return {
            'pupil_xy': self.pupil_xy,
            'pupil_rho': self.pupil_rho,
            'angle_xy': self.angle_xy,
            'aberration': self.aberration.tolist(),
            'mask2d_on': self.mask2d_on,
            'mask2d_sign': self.mask2d_sign,
            'mask3d_on': self.mask3d_on,
            'mask3d_radius': self.mask3d_radius,
            'mask3d_height': self.mask3d_height,
        }

    def dict2parameters(self, d):
        self.pupil_rho = d['pupil_rho']
        self.aberration = np.array(d['aberration']).reshape((-1, 1))
        self.mask2d_on = d['mask2d_on']
        self.mask2d_sign = d['mask2d_sign']
        self.mask3d_on = d['mask3d_on']
        self.mask3d_radius = d['mask3d_radius']
        self.mask3d_height = d['mask3d_height']
        self.angle_xy = d['angle_xy']

    def refresh_pupil(self, hologram_geometry):
        dirty = False
        if (
                self.xv.shape[0] != self.hologram_geometry[3] or
                self.xv.shape[1] != self.hologram_geometry[2]):

            self.log.info('allocating Zernike')

            def make_dd(rho, n, x):
                scale = (n/2)/rho
                dd = np.linspace(-scale, scale, n)
                dd -= np.diff(dd)[0]*x
                return dd

            dd1 = make_dd(
                self.pupil_rho,
                self.hologram_geometry[2],
                self.pupil_xy[0])
            dd2 = make_dd(
                self.pupil_rho,
                self.hologram_geometry[3],
                self.pupil_xy[1])
            self.xv, self.yv = np.meshgrid(dd1, dd2)
            self.rzern.make_cart_grid(self.xv, self.yv)
            dirty = True

        if (
                dirty or
                self.rzern is None or
                self.aberration.size != self.rzern.nk):
            nnew = int((-3 + sqrt(9 - 4*2*(1 - self.aberration.size)))/2)
            self.rzern = RZern(nnew)
            self.theta = np.arctan2(self.yv, self.xv)
            self.rho = np.sqrt(self.xv**2 + self.yv**2)

        self.make_phi2d()
        assert(np.all(np.isfinite(self.phi2d)))
        self.make_phi3d()
        assert(np.all(np.isfinite(self.phi3d)))
        self.make_phi()
        assert(np.all(np.isfinite(self.phi)))
        self.make_grating()
        assert(np.all(np.isfinite(self.grating)))

        def printout(t, x):
            if isinstance(x, np.ndarray):
                self.log.info(
                    f'{t} [{x.min():g}, {x.max():g}] {x.mean():g}')
            else:
                self.log.info(str(t) + ' [0.0, 0.0] 0.0')

        printout('phi', self.phi)
        printout('phi2d', self.phi2d)
        printout('phi3d', self.phi3d)

        phase = (
            self.phi +
            self.mask2d_on*self.phi2d +
            self.mask3d_on*self.phi3d +
            self.grating)

        return phase

    def make_phi2d(self):
        # [-pi, pi] principal branch
        phi2d = self.mask2d_sign*self.theta
        phi2d[self.rho >= 1.0] = 0
        self.phi2d = np.flipud(phi2d)

    def make_phi3d(self):
        # [-pi, pi] principal branch
        phi3d = np.zeros_like(self.rho)
        phi3d[self.rho <= self.mask3d_radius] = self.mask3d_height*np.pi
        phi3d[self.rho >= 1] = 0
        # induce zero mean
        phi3d -= phi3d.mean()
        phi3d[self.rho >= 1] = 0
        self.phi3d = np.flipud(phi3d)

    def make_grating(self):
        m = self.hologram_geometry[3]
        n = self.hologram_geometry[2]
        value_max = 15

        masks = np.indices((m, n), dtype="float")
        tt = self.angle_xy[0]*(
            masks[0, :, :] - self.pupil_xy[0] - n/2) + self.angle_xy[1]*(
            masks[1, :, :] - self.pupil_xy[1] - m/2)
        self.log.info(f'"make grating {str(self.angle_xy)}')
        tt = tt/value_max*2*np.pi
        tt[self.rho >= 1.0] = 0
        self.grating = np.flipud(tt)

    def make_phi(self):
        # [-pi, pi] principal branch
        phi = np.pi + self.rzern.eval_grid(self.aberration)
        phi = np.ascontiguousarray(
            phi.reshape(self.arr.shape, order='F'))
        phi[self.rho >= 1.0] = 0
        self.phi = np.flipud(phi)

    def set_pupil_xy(self, xy):
        if xy is None:
            self.pupil_xy = [0.0, 0.0]
            self.rzern = None
        else:
            if self.pupil_xy[0] != xy[0]:
                self.pupil_xy[0] = xy[0]
                self.rzern = None

            if self.pupil_xy[1] != xy[1]:
                self.pupil_xy[1] = xy[1]
                self.rzern = None
        self.rzern = None
        self.holo.refresh_hologram()

    def set_pupil_rho(self, rho):
        if rho is None:
            self.pupil_rho = min(self.hologram_geometry[2:])/2*.9
        else:
            self.pupil_rho = rho
        self.rzern = None
        self.holo.refresh_hologram()

    def set_mask2d_sign(self, s):
        self.mask2d_sign = s
        self.holo.refresh_hologram()

    def set_mask3d_height(self, s):
        self.mask3d_height = s
        self.holo.refresh_hologram()

    def set_anglexy(self, val, ind):
        self.angle_xy[ind] = val
        self.holo.refresh_hologram()

    def set_aberration(self, aberration):
        if aberration is None:
            self.aberration = np.zeros((self.rzern.nk, 1))
        else:
            self.aberration = np.array(aberration).reshape((-1, 1))
        self.holo.refresh_hologram()

    def set_mask2d_on(self, on):
        self.mask2d_on = on
        self.holo.refresh_hologram()

    def set_mask3d_on(self, on):
        self.mask3d_on = on
        self.holo.refresh_hologram()

    def set_mask3d_radius(self, rho):
        if rho is None:
            self.mask3d_radius = 0.6*self.pupil_rho
        else:
            self.mask3d_radius = rho
        self.holo.refresh_hologram()


class SLM(QDialog):

    refreshHologramSignal = pyqtSignal()

    def __init__(self, d={}):
        super().__init__(
            parent=None,
            flags=Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.log = logging.getLogger(self.__class__.__name__)
        self.flat_file = None
        self.flat = None
        self.flat_on = 0.0
        self.hologram_geometry = [0, 0, 400, 200]

        self.pupils = [Pupil()]

        self.arr = None
        self.qim = None
        self.wrap_value = 0xff

    def parameters2dict(self):
        """Stores all relevant parameters in a dictionary. Useful for saving"""
        d = {
            'hologram_geometry': self.hologram_geometry,
            'wrap_value': self.wrap_value,
            'flat_file': self.flat_file,
            'flat_on': self.flat_on,
            'pupils': [p.parameters2dict() for p in self.pupils],
            }
        return d

    def dict2parameters(self, d):
        """Sets each SLM parameter value according to the ones stored in dictionary
        d"""
        self.hologram_geometry = d['hologram_geometry']
        self.wrap_value = d['wrap_value']
        if "flat_file" in d:
            self.set_flat(d['flat_file'])
            self.flat_on = d['flat_on']
        else:
            self.flat_on = False

        self.pupils.clear()
        for pd in d['pupils']:
            p = Pupil()
            p.dict2parameters(pd)

    def load(self, f):
        d = json.load(f)
        self.dict2parameters(d)
        self.refresh_hologram()
        return d

    def save(self, f, merge=None):
        d = self.parameters2dict()
        if merge:
            merge.update(d)
        else:
            merge = d
        json.dump(merge, f)

    def refresh_hologram(self):
        # flat file overwrites hologram dimensions
        if self.flat_file is None:
            # [0, 1]
            self.flat = 0.0
        else:
            self.copy_flat_shape()
        self.setGeometry(*self.hologram_geometry)
        self.setFixedSize(
            self.hologram_geometry[2], self.hologram_geometry[3])

        if (
                self.arr is None or
                self.qim is None or
                self.arr.shape[0] != self.hologram_geometry[3] or
                self.arr.shape[1] != self.hologram_geometry[2]):
            self.log.info('refresh_hologram(): ALLOCATING arr & qim')
            self.arr = np.zeros(
                shape=(self.hologram_geometry[3], self.hologram_geometry[2]),
                dtype=np.uint32)
            self.qim = QImage(
                self.arr.data, self.arr.shape[1], self.arr.shape[0],
                QImage.Format_RGB32)

        self.info.log('refresh_hologram(): repaint')

        phase = 0
        for p in self.pupils:
            phase += p.refresh_pupil(self.hologram_geometry)

        def printout(t, x):
            if isinstance(x, np.ndarray):
                self.log.info(
                    f'{t} [{x.min():g}, {x.max():g}] {x.mean():g}')
            else:
                self.log.info(str(t) + ' [0.0, 0.0] 0.0')

        # [0, 1] waves
        background = self.flat_on*self.flat
        # [-pi, pi] principal branch rads (zero mean)
        phase /= (2*np.pi)  # phase in waves
        # all in waves
        gray = background + phase
        printout('gray', gray)
        gray -= np.floor(gray.min())
        assert(gray.min() >= -1e-9)
        gray *= self.wrap_value
        printout('gray', gray)
        gray %= self.wrap_value
        printout('gray', gray)
        assert(gray.min() >= 0)
        assert(gray.max() <= 255)
        gray = gray.astype(np.uint8)
        self.arr[:] = gray.astype(np.uint32)*0x010101

        self.refreshHologramSignal.emit()

    def copy_flat_shape(self):
        self.hologram_geometry[2] = self.flat.shape[1]
        self.hologram_geometry[3] = self.flat.shape[0]

    def set_flat(self, fname, refresh_hologram=True):
        if fname is None or fname == '':
            self.flat_file = None
            self.flat = 0.0
        else:
            try:
                self.flat_file = fname
                self.flat = np.ascontiguousarray(
                    imread(fname), dtype=np.float)/255
                self.copy_flat_shape()
            except Exception:
                self.flat_file = None
                self.flat = 0.0
        if refresh_hologram:
            self.refresh_hologram()

    def set_hologram_geometry(self, geometry, refresh=True):
        if isinstance(self.flat, np.ndarray) and len(self.flat.shape) == 2:
            self.hologram_geometry[:2] = geometry[:2]
            self.copy_flat_shape()
        elif geometry is not None:
            self.hologram_geometry[:] = geometry[:]
        if refresh:
            self.refresh_hologram()

    def set_wrap_value(self, wrap_value):
        if wrap_value is None:
            self.wrap_value = 255
        else:
            self.wrap_value = wrap_value
        self.refresh_hologram()

    def set_flat_on(self, on):
        self.flat_on = on
        self.refresh_hologram()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Q:
            self.close()

    def paintEvent(self, e):
        if self.qim is not None:
            qp = QPainter()
            qp.begin(self)
            qp.drawImage(0, 0, self.qim)
            qp.end()


class DoubleSLM(SLM):

    newHologramSignal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.double_flat_on = False
        self.slm2 = SLM()
        self.slm2.refresh_hologram()
        self.slm2.refreshHologramSignal.connect(self.refresh_hologram)

        self.slm1 = SLM()
        self.slm1.refreshHologramSignal.connect(self.refresh_hologram)
        self.slm1.refresh_hologram()

    def refresh_hologram(self,**kwargs):
        """rewrite refresh hologram for double SLM"""

        if self.flat_file is None:
            # [0, 1]
            self.flat = 0.0
        else:
            self.copy_flat_shape()
        
        self.setGeometry(*self.hologram_geometry)
        self.setFixedSize(
            self.hologram_geometry[2], self.hologram_geometry[3])
        if (
                self.arr is None or
                self.qim is None or
                self.arr.shape[0] != self.hologram_geometry[3] or
                self.arr.shape[1] != self.hologram_geometry[2]):

            print('double refresh_hologram(): ALLOCATING arr & qim')
            self.arr = np.ndarray(
                shape=(self.hologram_geometry[3], self.hologram_geometry[2]),
                dtype=np.uint32)
            self.qim = QImage(
                self.arr.data, self.arr.shape[1], self.arr.shape[0],
                QImage.Format_RGB32)
            
        # [0, 1] waves
        background = self.flat_on*self.flat
        # [-pi, pi] principal branch rads (zero mean)
        phase = self.slm1.all_phase + self.slm2.all_phase

        if (self.double_flat_on and 
            self.flat_on and isinstance(self.flat, np.ndarray) and 
            len(self.flat.shape) == 2):
            
            
            #flipud as in make_phi for instance: all phase patterns are inverted
            rho1 = np.flipud(self.slm1.rho)
            rho2 = np.flipud(self.slm2.rho)
            
            prho1 = self.slm1.pupil_rho
            prho2 = self.slm2.pupil_rho
            
            def double_position_checker(slm):
                rho = slm.pupil_rho
                geom = np.array(slm.hologram_geometry[2:])
                xy = np.array(slm.pupil_xy) + geom//2
                print(rho,xy,geom)
                return (rho + xy < geom).all() and (rho - xy < np.zeros(2)).all()
            ok_to_pursue = ( double_position_checker(self.slm1) and 
                            double_position_checker(self.slm2) )
            if ok_to_pursue:
                #try:
                    #use prho1/prho2 to ensure the exact same number of pixels in
                    #both masks
                print("ok to pursue")
                phase[rho1*prho1/prho2<=prho1/prho2] += \
                    self.flat[::-1,::-1][rho2[::-1,::-1]<=prho1/prho2]*2*np.pi
                phase[rho2*prho2/prho1<=prho2/prho1] += \
                    self.flat[::-1,::-1][rho1[::-1,::-1]<=prho2/prho1]*2*np.pi
            else:
                print("Double flat disabled")
                #self.set_double_flat_on(False,refresh=False)
            """except Exception as e:
                message = "Double flat disabled: both pupils must be in the FOV"
                message+="\n"+str(e)
                QMessageBox.information(self, 
                                        'Error', message)"""
       
        phase /= (2*np.pi)  # phase in waves
        gray = background+phase
        
        
        def printout(t, x):
            if isinstance(x, np.ndarray):
                print('{} [{:g}, {:g}] {:g}'.format(
                    t, x.min(), x.max(), x.mean()))
            else:
                print(t + ' [0.0, 0.0] 0.0')

        printout('gray', gray)
        gray -= np.floor(gray.min())
        assert(gray.min() >= -1e-9)
        print("wrap",self.wrap_value)
        gray *= self.wrap_value
        printout('gray', gray)
        gray %= self.wrap_value
        printout('gray', gray)
        assert(gray.min() >= 0)
        assert(gray.max() <= 255)
        gray = gray.astype(np.uint8)
        
        self.arr[:] = gray.astype(np.uint32)*0x010101
            
        self.newHologramSignal.emit(gray)
        self.update()

    def set_flat(self, fname):
        super().set_flat(fname,refresh_hologram = False)
        if isinstance(self.flat,np.ndarray) and len(self.flat.shape)==2:
            self.hologram_geometry[2:] = self.flat.shape[::-1]
            self.copy_flat_shape()
            self.refresh_hologram()
        
    def set_double_flat_on(self,on,refresh = True):
        self.double_flat_on = on
        if refresh:
            self.refresh_hologram()
        
    #Overrinding methods for compatibility
    def set_hologram_geometry(self, geometry):
        #Disconnection to avoid conflict when resizing
        self.geometry = geometry
        self.slm2.blockSignals(True)
        self.slm2.set_hologram_geometry(geometry)
        self.slm2.blockSignals(False)
        
        self.slm1.set_hologram_geometry(geometry)
        
    def set_wrap_value(self,wrap_value):
        self.wrap_value = wrap_value

    def copy_flat_shape(self):
        self.slm2.hologram_geometry[2] = self.flat.shape[1]
        self.slm2.hologram_geometry[3] = self.flat.shape[0]   
        
        self.slm1.hologram_geometry[2] = self.flat.shape[1]
        self.slm1.hologram_geometry[3] = self.flat.shape[0]  
        
        self.slm1.blockSignals(True)
        self.slm1.refresh_hologram()
        self.slm1.blockSignals(False)
        
        self.slm2.blockSignals(True)
        self.slm2.refresh_hologram()
        self.slm2.blockSignals(False)
        
    def load(self, f):
        d = json.load(f)
        d1 = d["pupil1"]
        d2 = d["pupil2"]
            
        #Avoids refreshing hologram 1 when loading hologram 2 to avoid
        #conflicts
        
            
        self.slm2.blockSignals(True)
        self.slm2.dict2parameters(d2)
        self.slm2.blockSignals(False)
        
        self.slm1.blockSignals(True)
        self.slm1.dict2parameters(d1)
        self.slm1.blockSignals(False)

        self.hologram_geometry = d1["hologram_geometry"]
        self.wrap_value = d1["wrap_value"]
        
        if "common_parameters" in d:
            d3 = d["common_parameters"]
            self.dict2parameters(d3)
        self.refresh_hologram()
        return d
    
    def save(self, f, merge=None):
        d1 = self.slm1.parameters2dict()
        d2 = self.slm2.parameters2dict()
        d3 = self.parameters2dict()
        d = {"pupil1":d1,
             "pupil2":d2,
             "common_parameters": d3,
             "double_flat_on":self.double_flat_on}
        if merge:
            merge.update(d)
        else:
            merge = d
        json.dump(merge, f)
    
    def paintEvent(self, e):
        print("double paint",self.qim)
        if self.qim is not None:
            qp = QPainter()
            qp.begin(self)
            qp.drawImage(0, 0, self.qim)
            qp.end()


class PhaseDisplay(QWidget):
    # TODO pthread me or the SLM class itself
    dirty = False
    size = [0, 0]
    phase = None
    arr = None
    qim = None
    rzern = None

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(200)
        self.setMinimumHeight(200)

    def update_phase(self, n, z):
        if self.rzern is None or self.rzern.n != n:
            rzern = RZern(n)
            dd = np.linspace(-1, 1, 100)
            xv, yv = np.meshgrid(dd, dd)
            rzern.make_cart_grid(xv, yv)
            self.rzern = rzern
        phi = self.rzern.eval_grid(z).reshape((100, 100), order='F')
        self.phi = phi
        self.dirty = True

    def resizeEvent(self, event):
        self.size[0] = event.size().width()
        self.size[1] = event.size().height()
        self.minsize = min(self.size)

        self.arr = np.ndarray(
            shape=(self.minsize, self.minsize), dtype=np.uint32)
        self.qim = QImage(
            self.arr.data, self.arr.shape[1], self.arr.shape[0],
            QImage.Format_RGB32)

    def paintEvent(self, e):
        if self.qim is not None:
            if self.dirty:
                fig10 = plt.figure(
                    10, figsize=(self.minsize/100, self.minsize/100), dpi=100)
                ax = fig10.gca()
                ax.axis('off')
                plt.imshow(self.phi)
                cb = plt.colorbar()
                cb.ax.tick_params(labelsize=6)
                fig10.canvas.draw()
                size = fig10.canvas.get_width_height()
                assert(size[0] == self.arr.shape[0])
                assert(size[1] == self.arr.shape[1])
                drawn = np.fromstring(
                            fig10.canvas.tostring_rgb(),
                            dtype=np.uint8, sep='').reshape(
                                (self.minsize, self.minsize, 3))
                self.arr[:] = (
                    drawn[:, :, 0]*0x01 +
                    drawn[:, :, 1]*0x0100 +
                    drawn[:, :, 2]*0x010000)
                plt.close(fig10)
            qp = QPainter()
            qp.begin(self)
            qp.drawImage(0, 0, self.qim)
            qp.end()


class MatplotlibWindow(QDialog):

    def __init__(self, parent=None,toolbar=False,figsize=None):
        super().__init__(parent)

        # a figure instance to plot on
        if figsize is None:
            self.figure = Figure()
        else:
            self.figure = Figure(figsize)
        self.ax = self.figure.add_subplot(111)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvasQTAgg(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # set the layout
        layout = QVBoxLayout()
        if toolbar:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        
    def update_array(self,arr):
        self.ax.cla()
        self.ax.imshow(arr,cmap="gray")
        self.ax.axis("off")
        self.canvas.draw()


class Control(QDialog):
    # TODO refactor and cleanup awful Qt code

    close_slm = True

    @staticmethod
    def helper1(name, labels, mins, handlers, curvals, Validator):
        group = QGroupBox(name)
        l1 = QGridLayout()
        for i, tup in enumerate(zip(labels, mins, handlers, curvals)):
            txt, mini, handler, curval = tup
            l1.addWidget(QLabel(txt), 0, 2*i)
            le = QLineEdit(str(curval))
            le.editingFinished.connect(handler(i, le))
            le.setMaximumWidth(50)
            val = Validator()
            le.setValidator(val)
            if mini:
                val.setBottom(mini)
            l1.addWidget(le, 0, 2*i + 1)
        group.setLayout(l1)
        return group

    @staticmethod
    def helper_boolupdate(mycallback, myupdate):
        def f(i):
            mycallback(i)
            myupdate()
        return f

    def make_geometry_tab(self, slm):
        def handle_geometry(ind, le):
            def f():
                try:
                    ival = int(le.text())
                except Exception:
                    le.setText(str(slm.hologram_geometry[ind]))
                    return
                slm.hologram_geometry[ind] = ival
                slm.set_hologram_geometry(slm.hologram_geometry)
                le.setText(str(slm.hologram_geometry[ind]))
            return f

        self.group_geometry = self.helper1(
            'Geometry',
            ['x', 'y', 'width', 'height'],
            [None, None, 100, 100],
            [handle_geometry]*4,
            slm.hologram_geometry, QIntValidator)

    def make_pupil_tab(self, slm):

        def handle_pupil_xy(ind, le):
            def f():
                try:
                    fval = float(le.text())
                    print(fval)
                except Exception:
                    le.setText(str(slm.pupil_xy[ind]))
                    return
                slm.pupil_xy[ind] = fval
                slm.set_pupil_xy(slm.pupil_xy)
                le.setText(str(slm.pupil_xy[ind]))
                slm.update()
            return f

        def handle_pupil_rho(ind, le):
            def f():
                try:
                    fval = float(le.text())
                except Exception:
                    le.setText(str(slm.pupil_rho))
                    return
                slm.set_pupil_rho(fval)
                le.setText(str(slm.pupil_rho))
                slm.update()
            return f

        self.group_pupil = self.helper1(
            'Pupil',
            ['x0', 'y0', 'radius'],
            [None, None, 10],
            [handle_pupil_xy, handle_pupil_xy, handle_pupil_rho],
            [slm.pupil_xy[0], slm.pupil_xy[1], slm.pupil_rho],
            QDoubleValidator)

    def make_flat_tab(self, slm):
        def helper_load_flat1():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName()
                if fdiag:
                    slm.set_flat(fdiag)
                    self.reinitialize(slm)
            return myf1

        g = QGroupBox('Flattening')
        l1 = QGridLayout()
        cboxlf = QCheckBox('flat on')
        cboxlf.toggled.connect(self.helper_boolupdate(
            slm.set_flat_on, slm.update))
        cboxlf.setChecked(slm.flat_on)
        l1.addWidget(cboxlf, 0, 0)
        loadbut = QPushButton('load')
        loadbut.clicked.connect(helper_load_flat1())
        l1.addWidget(loadbut, 0, 1)
        g.setLayout(l1)

        self.group_flat = g

    def make_wrap_tab(self, slm):
        g = QGroupBox('Wrap value')
        l1 = QGridLayout()
        lewrap = QLineEdit(str(slm.wrap_value))
        lewrap.setMaximumWidth(50)
        lewrap.setValidator(QIntValidator(1, 255))

        def handle_wrap(lewrap1):
            def f():
                try:
                    ival = int(lewrap1.text())
                except Exception:
                    lewrap1.setText(str(slm.wrap_value))
                    return
                slm.set_wrap_value(ival)
                slm.update()
                lewrap1.setText(str(slm.wrap_value))
            return f

        lewrap.editingFinished.connect(handle_wrap(lewrap))
        l1.addWidget(lewrap, 0, 0)
        g.setLayout(l1)

        self.group_wrap = g

    def make_2d_tab(self, slm):
        g = QGroupBox('2D STED')
        l1 = QGridLayout()
        c = QCheckBox('2D on')
        c.setChecked(slm.mask2d_on)
        c.toggled.connect(self.helper_boolupdate(
            slm.set_mask2d_on, slm.update))
        l1.addWidget(c, 0, 0)
        sign2d = QComboBox()
        sign2d.addItem('+1')
        sign2d.addItem('-1')
        if slm.mask2d_sign == 1:
            sign2d.setCurrentIndex(0)
        else:
            sign2d.setCurrentIndex(1)

        def toggle_float(fun):
            def f(val):
                if val == 0:
                    fun(float(1))
                else:
                    fun(float(-1))
                slm.update()
            return f

        sign2d.activated.connect(toggle_float(slm.set_mask2d_sign))
        l1.addWidget(sign2d, 0, 1)
        g.setLayout(l1)

        self.group_2d = g

    def make_3d_tab(self, slm):
        g = QGroupBox('3D STED')

        def update_radius(slider, what):
            def f(r):
                slider.setValue(int(r*100))
                what(r)
                slm.update()
            return f

        def update_spinbox(s):
            def f(t):
                s.setValue(t/100)
            return f

        l1 = QGridLayout()
        c = QCheckBox('3D on')
        c.setChecked(slm.mask3d_on)
        c.toggled.connect(self.helper_boolupdate(
            slm.set_mask3d_on, slm.update))
        l1.addWidget(c, 0, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(20)
        slider.setSingleStep(0.1)
        slider.setValue(int(100*slm.mask3d_radius))
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.0, 1.0)
        spinbox.setSingleStep(0.01)
        spinbox.setValue(slm.mask3d_radius)
        slider.valueChanged.connect(update_spinbox(spinbox))

        spinbox.valueChanged.connect(update_radius(
            slider, slm.set_mask3d_radius))
        l1.addWidget(QLabel('radius'), 0, 1)
        l1.addWidget(spinbox, 0, 2)
        l1.addWidget(slider, 0, 3)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(200)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(40)
        slider.setSingleStep(0.1)
        slider.setValue(int(100*slm.mask3d_height))
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.0, 2.0)
        spinbox.setSingleStep(0.01)
        spinbox.setValue(slm.mask3d_height)
        slider.valueChanged.connect(update_spinbox(spinbox))

        spinbox.valueChanged.connect(update_radius(
            slider, slm.set_mask3d_height))
        l1.addWidget(QLabel('height'), 1, 1)
        l1.addWidget(spinbox, 1, 2)
        l1.addWidget(slider, 1, 3)
        g.setLayout(l1)

        self.group_3d = g

    def make_phase_tab(self):
        g = QGroupBox('Phase')
        phase_display = PhaseDisplay()
        l1 = QGridLayout()
        l1.addWidget(phase_display, 0, 0)
        g.setLayout(l1)

        self.group_phase = g
        self.phase_display = phase_display

    def make_aberration_tab(self, slm, phase_display):
        def default_zernike_name(i, n, m):
            if i == 1:
                return 'piston'
            elif i == 2:
                return 'tip'
            elif i == 3:
                return 'tilt'
            elif i == 4:
                return 'defocus'
            elif m == 0:
                return 'spherical'
            elif abs(m) == 1:
                return 'coma'
            elif abs(m) == 2:
                return 'astigmatism'
            elif abs(m) == 3:
                return 'trefoil'
            elif abs(m) == 4:
                return 'quadrafoil'
            elif abs(m) == 5:
                return 'pentafoil'
            else:
                return ''

        if 'zernike_labels' not in self.settings.keys():
            self.settings['zernike_labels'] = dict()

        multiplier = 100

        top = QGroupBox('Zernike aberrations')
        toplay = QGridLayout()
        top.setLayout(toplay)
        labzm = QLabel('max radial order')
        print(slm)
        print(slm.rzern)
        lezm = QLineEdit(str(slm.rzern.n))
        lezm.setMaximumWidth(50)
        lezm.setValidator(QIntValidator(1, 255))
        reset = QPushButton('reset')
        toplay.addWidget(labzm, 0, 0)
        toplay.addWidget(lezm, 0, 1)
        toplay.addWidget(reset, 0, 4)

        scroll = QScrollArea()
        toplay.addWidget(scroll, 1, 0, 1, 5)
        scroll.setWidget(QWidget())
        scrollLayout = QGridLayout(scroll.widget())
        scroll.setWidgetResizable(True)

        def fto100(f, amp):
            maxrad = float(amp.text())
            return int((f + maxrad)/(2*maxrad)*multiplier)

        def update_coeff(slider, ind, amp):
            def f(r):
                slider.blockSignals(True)
                slider.setValue(fto100(r, amp))
                slider.blockSignals(False)
                slm.aberration[ind, 0] = r
                slm.set_aberration(slm.aberration)
                slm.update()

                phase_display.update_phase(slm.rzern.n, slm.aberration)
                phase_display.update()
            return f

        def update_amp(spinbox, slider, le, i):
            def f():
                amp = float(le.text())
                spinbox.setRange(-amp, amp)
                spinbox.setValue(spinbox.value())
                slider.setValue(fto100(slm.aberration[i, 0], le))
            return f

        def update_zlabel(le, settings, i):
            def f():
                settings['zernike_labels'][str(i)] = le.text()
            return f

        def update_spinbox(s, amp):
            def f(t):
                maxrad = float(amp.text())
                s.setValue(t/multiplier*(2*maxrad) - maxrad)
            return f

        def update_zernike_rows():
            mynk = slm.rzern.nk
            ntab = slm.rzern.ntab
            mtab = slm.rzern.mtab
            if len(zernike_rows) < mynk:
                for i in range(len(zernike_rows), mynk):
                    lab = QLabel(
                        'Z<sub>{}</sub> Z<sub>{}</sub><sup>{}</sup>'.format(
                            i + 1, ntab[i], mtab[i]))
                    slider = QSlider(Qt.Horizontal)
                    spinbox = QDoubleSpinBox()
                    maxamp = max((4, slm.aberration[i, 0]))
                    if str(i) in self.settings['zernike_labels'].keys():
                        zname = self.settings['zernike_labels'][str(i)]
                    else:
                        zname = default_zernike_name(i + 1, ntab[i], mtab[i])
                        self.settings['zernike_labels'][str(i)] = zname
                    lbn = QLineEdit(zname)
                    lbn.setMaximumWidth(120)
                    amp = QLineEdit(str(maxamp))
                    amp.setMaximumWidth(50)
                    val = QDoubleValidator()
                    val.setBottom(0.0)
                    amp.setValidator(val)

                    slider.setMinimum(0)
                    slider.setMaximum(multiplier)
                    slider.setFocusPolicy(Qt.StrongFocus)
                    slider.setTickPosition(QSlider.TicksBothSides)
                    slider.setTickInterval(20)
                    slider.setSingleStep(0.1)
                    slider.setValue(fto100(slm.aberration[i, 0], amp))
                    spinbox.setRange(-maxamp, maxamp)
                    spinbox.setSingleStep(0.01)
                    spinbox.setValue(slm.aberration[i, 0])

                    hand1 = update_spinbox(spinbox, amp)
                    hand2 = update_coeff(slider, i, amp)
                    hand3 = update_amp(spinbox, slider, amp, i)
                    hand4 = update_zlabel(lbn, self.settings, i)
                    slider.valueChanged.connect(hand1)
                    spinbox.valueChanged.connect(hand2)
                    amp.editingFinished.connect(hand3)
                    lbn.editingFinished.connect(hand4)

                    scrollLayout.addWidget(lab, i, 0)
                    scrollLayout.addWidget(lbn, i, 1)
                    scrollLayout.addWidget(spinbox, i, 2)
                    scrollLayout.addWidget(slider, i, 3)
                    scrollLayout.addWidget(amp, i, 4)

                    zernike_rows.append((
                        lab, slider, spinbox, hand1, hand2, amp,
                        hand3, lbn, hand4))

                assert(len(zernike_rows) == mynk)

            elif len(zernike_rows) > mynk:
                for i in range(len(zernike_rows) - 1, mynk - 1, -1):
                    tup = zernike_rows.pop()
                    lab, slider, spinbox, h1, h2, amp, h3, lbn, h4 = tup

                    scrollLayout.removeWidget(lab)
                    scrollLayout.removeWidget(lbn)
                    scrollLayout.removeWidget(spinbox)
                    scrollLayout.removeWidget(slider)
                    scrollLayout.removeWidget(amp)

                    slider.valueChanged.disconnect(h1)
                    spinbox.valueChanged.disconnect(h2)
                    amp.editingFinished.disconnect(h3)
                    lbn.editingFinished.disconnect(h4)

                    lab.setParent(None)
                    slider.setParent(None)
                    spinbox.setParent(None)
                    amp.setParent(None)
                    lbn.setParent(None)

                assert(len(zernike_rows) == mynk)

        def reset_fun():
            for t in zernike_rows:
                t[2].setValue(0.0)

        def change_radial():
            try:
                ival = int(lezm.text())
            except Exception:
                lezm.setText(str(slm.rzern.n))
                return
            n = (ival + 1)*(ival + 2)//2
            newab = np.zeros((n, 1))
            minn = min((n, slm.rzern.n))
            newab[:minn, 0] = slm.aberration[:minn, 0]
            slm.set_aberration(newab)
            slm.update()

            update_zernike_rows()
            phase_display.update_phase(slm.rzern.n, slm.aberration)
            phase_display.update()
            lezm.setText(str(slm.rzern.n))

        phase_display.update_phase(slm.rzern.n, slm.aberration)
        zernike_rows = list()
        update_zernike_rows()

        reset.clicked.connect(reset_fun)
        lezm.editingFinished.connect(change_radial)

        self.group_aberration = top

    def make_grating_tab(self,slm):
        """Position tab is meant to help positionning the phase mask
        without using tip and tilt"""
        pos = QGroupBox('Blazed grating')
        poslay = QGridLayout()
        pos.setLayout(poslay)
        labx = QLabel('x')
        laby = QLabel('y')

        multiplier = 100
        amp = 4

        def fto100(f, amp):
            maxrad = float(3)
            return int((f + maxrad)/(2*maxrad)*multiplier)

        # x position
        slider_x = QSlider(Qt.Horizontal)
        slider_x.setMinimum(0)
        slider_x.setMaximum(multiplier)
        slider_x.setSingleStep(0.01)
        slider_x.setValue(fto100(slm.angle_xy[0], amp))

        spinbox_x = QDoubleSpinBox()
        spinbox_x.setRange(-amp, amp)
        spinbox_x.setSingleStep(0.01)
        spinbox_x.setValue(slm.angle_xy[0])

        # y position
        slider_y = QSlider(Qt.Horizontal)
        slider_y.setMinimum(0)
        slider_y.setMaximum(multiplier)
        slider_y.setSingleStep(0.01)
        slider_y.setValue(fto100(slm.angle_xy[1], amp))

        spinbox_y = QDoubleSpinBox()
        spinbox_y.setRange(-amp, amp)
        spinbox_y.setSingleStep(0.01)
        spinbox_y.setValue(slm.angle_xy[1])

        # x position
        poslay.addWidget(labx, 0, 0)
        poslay.addWidget(slider_x, 0, 1)
        poslay.addWidget(spinbox_x, 0, 2)
        # y position
        poslay.addWidget(laby, 0, 3)
        poslay.addWidget(slider_y, 0, 4)
        poslay.addWidget(spinbox_y, 0, 5)

        def update_coeff(slider, amp, axis):

            def f(r):
                print("UPDATE COEFF SA MERE LA P")
                slider.blockSignals(True)
                slider.setValue(fto100(r, amp))
                slider.blockSignals(False)
                slm.set_anglexy(r,axis)
                slm.update()
                """self.slm.angle_xy[axis] = r
                self.slm.refresh_hologram()"""
            return f

        def update_spinbox(s, amp):

            def f(t):
                maxrad = float(amp)
                s.setValue(t/multiplier*(2*maxrad) - maxrad)
            return f


        hand1 = update_spinbox(spinbox_x, 4)
        hand2 = update_coeff(slider_x, 4, 0)
        slider_x.valueChanged.connect(hand1)
        spinbox_x.valueChanged.connect(hand2)

        hand3 = update_spinbox(spinbox_y, amp)
        hand4 = update_coeff(slider_y, amp, 1)
        slider_y.valueChanged.connect(hand3)
        spinbox_y.valueChanged.connect(hand4)

        self.group_grating = pos

    def make_file_tab(self, slm):
        g = QGroupBox('File')
        load = QPushButton('load')
        save = QPushButton('save')
        l1 = QGridLayout()
        l1.addWidget(load, 0, 1)
        l1.addWidget(save, 0, 0)
        g.setLayout(l1)

        def helper_load():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName()
                if fdiag:
                    try:
                        with open(fdiag, 'r') as f:
                            d = slm.load(f)['control']
                            Control(self.slm, d).show()
                            self.close_slm = False
                            self.close()
                    except Exception as e:
                        QMessageBox.information(self, 'Helper load Error', str(e))

                    print(fdiag)
            return myf1

        def helper_save():
            # self.setGeometry(*self.hologram_geometry)
            def myf1():
                curg = self.geometry()
                fdiag, _ = QFileDialog.getSaveFileName(
                    directory=datetime.now().strftime('%Y%m%d_%H%M%S.json'))
                if fdiag:
                    try:
                        with open(fdiag, 'w') as f:
                            self.settings['window'] = [
                                curg.x(), curg.y(),
                                curg.width(), curg.height()]
                            slm.save(f, {'control': self.settings})
                    except Exception as e:
                        QMessageBox.information(self, 'Error', str(e))

                    print(fdiag)
            return myf1

        load.clicked.connect(helper_load())
        save.clicked.connect(helper_save())

        self.group_file = g

    def __init__(self, slm, settings,is_parent=True):
        """Subclass for a control GUI.
        Parameters:
            slm: SLM instance
            settings: dict, saved settings
            is_parent: bool. Useful in the case of doublepass to determine
                for instance which widget determines the overall geometry"""
        super().__init__(parent=None)
        self.setWindowTitle(
            'SLM ' + version.__version__ + ' ' + version.__date__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        self.settings = settings
        
        if 'window' in settings.keys():
            self.setGeometry(
                settings['window'][0], settings['window'][1],
                settings['window'][2], settings['window'][3])
            
        self.is_parent = is_parent
        
        self.make_geometry_tab(slm)
        self.make_pupil_tab(slm)
        self.make_flat_tab(slm)
        self.make_wrap_tab(slm)
        self.make_2d_tab(slm)
        self.make_3d_tab(slm)
        self.make_phase_tab()
        self.make_grating_tab(slm)
        self.make_aberration_tab(slm, self.phase_display)
        self.make_file_tab(slm)

        top = QGridLayout()
        self.top = top
        if is_parent: #Single pupil case, we add all the buttons
            top.addWidget(self.group_geometry, 0, 0, 1, 2)
            top.addWidget(self.group_pupil, 1, 0)
            top.addWidget(self.group_wrap, 1, 1)
            
            top.addWidget(self.group_flat, 2, 0)
            top.addWidget(self.group_2d, 3, 0)
            top.addWidget(self.group_3d, 4, 0)
            top.addWidget(self.group_phase, 0, 2, 2, 1)
            top.addWidget(self.group_grating, 2, 1, 1, 2)
            top.addWidget(self.group_aberration, 3, 1, 3, 2)
            top.addWidget(self.group_file, 5, 0)
        else:
            #!!!
            #top.addWidget(self.group_geometry, 0, 0, 1, 2)            
            #top.addWidget(self.group_wrap, 1, 1)
            #top.addWidget(self.group_flat, 2, 0)
            #top.addWidget(self.group_file, 5, 0)

            top.addWidget(self.group_phase, 0, 0, 4, 1)
            top.addWidget(self.group_pupil, 0, 1)
            top.addWidget(self.group_grating, 1, 1)
            top.addWidget(self.group_2d, 2, 1)
            top.addWidget(self.group_3d, 3, 1)

            top.addWidget(self.group_aberration, 4, 0, 2, 2)
        self.setLayout(top)

        self.top = top
        self.slm = slm
        # self.resize(QDesktopWidget().availableGeometry().size()*0.5)

    def reinitialize(self,slm):
        """Useful after loading a new correction file, sets all parameters to
        correct value"""
        self.group_pupil.deleteLater()
        self.group_2d.deleteLater()
        self.group_3d.deleteLater()
        self.group_phase.deleteLater()
        self.group_grating.deleteLater()
        self.group_aberration.deleteLater()
        
        #if self.is_parent:
        self.group_wrap.deleteLater()
        self.group_flat.deleteLater()
        self.group_file.deleteLater()
        self.group_geometry.deleteLater()
        
        self.make_pupil_tab(slm)
        self.make_2d_tab(slm)
        self.make_3d_tab(slm)
        self.make_phase_tab()
        self.make_aberration_tab(slm, self.phase_display)
        self.make_grating_tab(slm)
        
        #if self.is_parent:
        self.make_wrap_tab(slm)
        self.make_file_tab(slm)
        self.make_geometry_tab(slm)
        self.make_flat_tab(slm)
    
        if self.is_parent: #Single pupil case, we add all the buttons
            self.top.addWidget(self.group_geometry, 0, 0, 1, 2)
            self.top.addWidget(self.group_pupil, 1, 0)
            self.top.addWidget(self.group_wrap, 1, 1)
            
            self.top.addWidget(self.group_flat, 2, 0)
            self.top.addWidget(self.group_2d, 3, 0)
            self.top.addWidget(self.group_3d, 4, 0)
            self.top.addWidget(self.group_phase, 0, 2, 2, 1)
            self.top.addWidget(self.group_grating, 2, 1, 1, 2)
            self.top.addWidget(self.group_aberration, 3, 1, 3, 2)
            self.top.addWidget(self.group_file, 5, 0)
        else:
            #!!!
            #top.addWidget(self.group_geometry, 0, 0, 1, 2)            
            #top.addWidget(self.group_wrap, 1, 1)
            #top.addWidget(self.group_flat, 2, 0)
            #top.addWidget(self.group_file, 5, 0)
            self.top.addWidget(self.group_phase, 0, 0, 4, 1)
            self.top.addWidget(self.group_pupil, 0, 1)
            self.top.addWidget(self.group_grating, 1, 1)
            self.top.addWidget(self.group_2d, 2, 1)
            self.top.addWidget(self.group_3d, 3, 1)

            self.top.addWidget(self.group_aberration, 4, 0, 2, 2)
        
        print("end reinitialize")

    def closeEvent(self, event):
        if self.close_slm:
            self.slm.close()
        super().close()

    def keyPressEvent(self, event):
        pass


def get_default_parameters():
    return {
        'control': {
            'SingleZernike': {
                'include': [],
                'exclude': [1, 2, 3, 4],
                'min': 5,
                'max': 6,
                'all': 1,
                'pupil': 1,
                },
            'DoubleZernike': {
                'include': [],
                'exclude': [1, 2, 3, 4],
                'min': 5,
                'max': 6,
                'all': 1,
                },
            }
        }


def get_parameters_info():
    return {
        'control': {
            'SingleZernike': {
                'include': (list, int, 'Zernike indices to include'),
                'exclude': (list, int, 'Zernike indices to include'),
                'min': (int, (1, None), 'Minimum Zernike index'),
                'max': (int, (1, None), 'Maximum Zernike index'),
                'all': (
                    int, (0, 1), 'Use all Zernike available in calibration'),
                'pupil': (
                    int, (1, 2), 'SLM pupil number'),
                },
            'DoubleZernike': {
                'include': (list, int, 'Zernike indices to include'),
                'exclude': (list, int, 'Zernike indices to include'),
                'min': (int, (1, None), 'Minimum Zernike index'),
                'max': (int, (1, None), 'Maximum Zernike index'),
                'all': (
                    int, (0, 1), 'Use all Zernike available in calibration'),
                },
            }
        }


def merge_pars(dp, up):
    p = {}
    for k, v in dp.items():
        if type(v) == dict:
            options = list(v.keys())
            if k not in up:
                p[k] = {options[0]: dp[k][options[0]]}
            else:
                choice = list(up[k].keys())
                assert(len(choice) == 1)
                choice = choice[0]
                assert(choice in dp[k].keys())
                p[k] = {choice: merge_pars(dp[k][choice], up[k][choice])}
        else:
            if k in up:
                p[k] = up[k]
            else:
                p[k] = dp[k]
    return p


def get_noll_indices(params):
    p = params['control']
    if 'Zernike' in p:
        z = p['Zernike']
        noll_min = z['min']
        noll_max = z['max']
        minclude = z['include']
        mexclude = z['exclude']
    else:
        RuntimeError()

    mrange = np.arange(noll_min, noll_max + 1)
    zernike_indices = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))
    return zernike_indices


class SingleZernikeControl:

    def __init__(self, slm, pars={}, h5f=None):
        pars = merge_pars(get_default_parameters(), pars)
        self.pars = pars
        self.log = logging.getLogger(self.__class__.__name__)
        self.slm = slm

        if pars['control']['SingleZernike']['pupil'] == 1:
            self.slm1 = slm.slm1
        else:
            self.slm1 = slm.slm2

        nz = self.slm1.aberration.size
        if pars['control']['SingleZernike']['pupil'] == 1:
            indices = np.arange(1, nz + 1)
        else:
            indices = get_noll_indices(pars)
        self.indices = indices
        ndof = indices.size

        self.ndof = ndof
        self.h5f = h5f
        self.z0 = self.slm1.aberration.ravel()

        self.h5_save('indices', self.indices)
        self.h5_save('flat', self.slm1.flat)
        self.h5_save('z0', self.z0)
        self.P = None

        if h5f:
            self.h5_make_empty('flat_on', (1,), np.bool)
            self.h5_make_empty('x', (ndof,))
            self.h5_make_empty('z2', (self.z0.size,))

        self.h5_save('name', self.__class__.__name__)
        self.h5_save('P', np.eye(nz))

    def h5_make_empty(self, name, shape, dtype=np.float):
        if self.h5f:
            name = self.__class__.__name__ + '/' + name
            if name in self.h5f:
                del self.h5f[name]
            self.h5f.create_dataset(
                name, shape + (0,), maxshape=shape + (None,),
                dtype=dtype)

    def h5_append(self, name, what):
        if self.h5f:
            name = self.__class__.__name__ + '/' + name
            self.h5f[name].resize((
                self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            name = self.__class__.__name__ + '/' + where
            if name in self.h5f:
                del self.h5f[name]
            self.h5f[name] = what

    def write(self, x):
        assert(x.size == self.ndof)
        z1 = np.zeros(self.slm1.aberration.size)
        z1[self.indices - 1] = x[:]
        if self.P is not None:
            z2 = np.dot(self.P, z1 + self.z0)
        else:
            z2 = z1 + self.z0
        self.slm1.set_aberration(z2.reshape(-1, 1))

        self.h5_append('flat_on', bool(self.slm1.flat_on))
        self.h5_append('x', x)
        self.h5_append('z2', z2)


class DoubleControl(QDialog):

    can_close = True
    close_slm = True
    sig_acquire = pyqtSignal(tuple)
    sig_release = pyqtSignal(tuple)

    def __init__(self, double_slm, settings):
        super().__init__(parent=None)
        self.mutex = QMutex()
        self.setWindowTitle(
            'SLM ' + version.__version__ + ' ' + version.__date__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        self.settings = settings

        if 'window' in settings.keys():
            self.setGeometry(
                settings['window'][0], settings['window'][1],
                settings['window'][2], settings['window'][3])

        self.double_slm = double_slm

        self.control1 = Control(self.double_slm.slm1,settings,is_parent=False)
        self.control2 = Control(self.double_slm.slm2,{},is_parent = False)
        
        self.make_pupils_tabs()
        self.make_geometry_tab(double_slm)
        self.make_general_display()
        self.make_parameters_group()
        
        self.top = QGridLayout()
        self.top.addWidget(self.display,0,0)
        self.top.addWidget(self.pupilsTab,0,1,2,1)
        self.top.addWidget(self.parametersGroup,1,0)
        
        self.setLayout(self.top)

        def make_release_hand():
            def f(t):
                # self.control.u[:] = t[0].u
                # self.zpanel.z[:] = self.control.u2z()
                # self.zpanel.update_controls()
                # self.zpanel.update_gui()
                self.setEnabled(True)
                self.can_close = True
                self.mutex.unlock()
            return f

        def make_acquire_hand():
            def f(t):
                self.mutex.lock()
                self.can_close = False
                self.setEnabled(False)
            return f

        self.sig_release.connect(make_release_hand())
        self.sig_acquire.connect(make_acquire_hand())

        
    @staticmethod
    def helper_boolupdate(mycallback, myupdate):
        def f(i):
            mycallback(i)
            myupdate()
        return f

    @staticmethod
    def helper1(name, labels, mins, handlers, curvals, Validator):
        group = QGroupBox(name)
        l1 = QGridLayout()
        for i, tup in enumerate(zip(labels, mins, handlers, curvals)):
            txt, mini, handler, curval = tup
            l1.addWidget(QLabel(txt), 0, 2*i)
            le = QLineEdit(str(curval))
            le.editingFinished.connect(handler(i, le))
            le.setMaximumWidth(50)
            val = Validator()
            le.setValidator(val)
            if mini:
                val.setBottom(mini)
            l1.addWidget(le, 0, 2*i + 1)
        group.setLayout(l1)
        return group

    def make_pupils_tabs(self):
        self.pupilsTab = QTabWidget()
        self.pupilsTab.addTab(self.control1,"Pupil 1")
        self.pupilsTab.addTab(self.control2,"Pupil 2")

    def make_general_display(self):
        self.display = MatplotlibWindow(figsize = (8,6))
        self.double_slm.newHologramSignal.connect(self.display.update_array)
    
    def make_file_tab(self, slm):
        """Rewriting the file tab to facilitate loading of 2-pupil files"""
        g = QGroupBox('File')
        load = QPushButton('load')
        save = QPushButton('save')
        l1 = QGridLayout()
        l1.addWidget(load, 0, 1)
        l1.addWidget(save, 0, 0)
        g.setLayout(l1)

        def helper_load():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName()
                if fdiag:
                    try:
                        with open(fdiag, 'r') as f:
                            slm.load(f)
                            self.control1.reinitialize(slm.slm1)
                            self.control2.reinitialize(slm.slm2)
                            self.reinitialise_parameters_group()
                    except Exception as e:
                        QMessageBox.information(self, 'Helper load 2 Error', str(e))

                    print(fdiag)
            return myf1

        def helper_save():
            # self.setGeometry(*self.hologram_geometry)
            def myf1():
                curg = self.geometry()
                fdiag, _ = QFileDialog.getSaveFileName(
                    directory=datetime.now().strftime('%Y%m%d_%H%M%S.json'))
                if fdiag:
                    try:
                        with open(fdiag, 'w') as f:
                            self.settings['window'] = [
                                curg.x(), curg.y(),
                                curg.width(), curg.height()]
                            slm.save(f, {'control': self.settings})
                    except Exception as e:
                        QMessageBox.information(self, 'Error', str(e))

                    print(fdiag)
            return myf1

        load.clicked.connect(helper_load())
        save.clicked.connect(helper_save())

        self.group_file = g
        
    def make_flat_tab(self):
        def helper_load_flat1():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName()
                if fdiag:
                    slm.set_flat(fdiag)
                    self.control1.reinitialize(self.double_slm.slm1) 
                    self.control2.reinitialize(self.double_slm.slm2) 
                    self.reinitialise_parameters_group()
            return myf1

        g = QGroupBox('Flattening')
        l1 = QGridLayout()
        cboxlf = QCheckBox('flat on')
        cboxlf.toggled.connect(self.helper_boolupdate(
            self.double_slm.set_flat_on, self.double_slm.update))
        cboxlf.setChecked(self.double_slm.flat_on)
        l1.addWidget(cboxlf, 0, 0)
        loadbut = QPushButton('load')
        loadbut.clicked.connect(helper_load_flat1())
        l1.addWidget(loadbut, 0, 1)
        g.setLayout(l1)
        self.group_flat = g
        
    def make_geometry_tab(self, double_slm):
            def handle_geometry(ind, le):
                def f():
                    try:
                        slm
                        ival = int(le.text())
                    except Exception:
                        le.setText(str(double_slm.hologram_geometry[ind]))
                        return
                    double_slm.slm1.hologram_geometry[ind] = ival
                    double_slm.slm2.hologram_geometry[ind] = ival
                    double_slm.hologram_geometry[ind] = ival
                    
                    double_slm.set_hologram_geometry(double_slm.hologram_geometry)
                    le.setText(str(double_slm.hologram_geometry[ind]))
                return f
    
            self.group_geometry = self.helper1(
                'Geometry',
                ['x', 'y', 'width', 'height'],
                [None, None, 100, 100],
                [handle_geometry]*4,
                double_slm.hologram_geometry, QIntValidator)

    def make_parameters_group(self):
        self.make_file_tab(self.double_slm)
        self.make_flat_tab()
        self.doubleFlatOnCheckBox = QCheckBox("Double flat on")
        self.doubleFlatOnCheckBox.setChecked(self.double_slm.double_flat_on)
        self.doubleFlatOnCheckBox.toggled.connect(self.double_slm.set_double_flat_on)
        
        group = QGroupBox("Parameters")
        top = QGridLayout()
        top.addWidget(self.group_geometry, 0, 0,1,3)     
        
        top.addWidget(self.group_flat, 1, 0)
        top.addWidget(self.control1.group_wrap, 1, 1)
        top.addWidget(self.doubleFlatOnCheckBox, 1, 2)
        
        top.addWidget(self.group_file, 2, 0,1,3)
        group.setLayout(top)
        self.parametersGroup = group
        
    def reinitialise_parameters_group(self):
        """Reinitializes the parameters when loading a correction file"""
        self.parametersGroup.deleteLater()
        self.group_geometry.deleteLater()
        self.make_parameters_group()
        self.make_geometry_tab(self.double_slm)
        self.top.addWidget(self.pupilsTab,0,1,2,1)
        self.top.addWidget(self.parametersGroup,1,0)

    def closeEvent(self, event):
        if self.can_close:
            if self.close_slm:
                self.double_slm.slm2.close()
                self.double_slm.close()
            super().close()
            event.accept()
        else:
            event.ignore()

    def acquire_control(self, h5f):
        self.sig_acquire.emit((h5f,))
        return SingleZernikeControl(self.double_slm, h5f=h5f)

    def release_control(self, control, h5f):
        self.sig_release.emit((control, h5f))
        self.control1.reinitialize(self.double_slm.slm1)
        self.control2.reinitialize(self.double_slm.slm2)
        self.reinitialise_parameters_group()


class Console(QDialog):

    def __init__(self, slm, control):
        super().__init__(parent=None)

        self.slm = slm
        self.control = control

        self.setWindowTitle('console' + version.__version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()
        kernel_client.namespace = self

        def stop():
            control.close()
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            self.close()

        layout = QVBoxLayout(self)
        widget = RichJupyterWidget(parent=self)
        layout.addWidget(widget)
        widget.kernel_manager = kernel_manager
        widget.kernel_client = kernel_client
        widget.exit_requested.connect(stop)
        ipython_widget = widget
        ipython_widget.show()
        kernel.shell.push({
            'plt': plt,
            'np': np,
            'slm': slm,
            'control': control,
            'widget': widget,
            'kernel': kernel,
            'parent': self})


def add_arguments(parser):
    parser.add_argument('--slm-dump', action='store_true')
    parser.add_argument('--slm-double', action='store_true')
    parser.add_argument('--slm-single', action='store_false', dest='double')
    parser.set_defaults(slm_double=True)
    parser.add_argument(
        '--slm-settings', type=argparse.FileType('r'), default=None,
        metavar='JSON', help='Load a previous configuration file')


def new_slm_window(app, args):
    if args.double:
        slm = DoubleSLM()
    else:
        slm = SLM()
    slm.show()
    slm.refresh_hologram()

    if args.slm_settings:
        d = slm.load(args.slm_settings)['control']
        args.slm_settings.close()
        args.slm_settings = args.slm_settings.name
    else:
        d = {}
    if args.double:
        control = DoubleControl(slm, d)
    else:
        control = Control(slm, d)
    control.show()

    return control


if __name__ == '__main__':
    app = QApplication(sys.argv)

    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='SLM control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--single', action='store_false', dest='double')
    parser.set_defaults(double=True)
    parser.add_argument(
        '--load', type=argparse.FileType('r'), default=None,
        metavar='JSON',
        help='Load a previous configuration file')
    args = parser.parse_args(args[1:])
    if args.double:
        slm = DoubleSLM()
    else:
        slm = SLM()
    slm.show()
    slm.refresh_hologram()

    if args.load:
        d = slm.load(args.load)['control']
        args.load.close()
    else:
        d = {}
    if args.double:
        control = DoubleControl(slm, d)
    else:
        control = Control(slm, d)
    control.show()

    if args.console:
        console = Console(slm, control)
        console.show()

    sys.exit(app.exec_())
