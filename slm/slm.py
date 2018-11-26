#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from datetime import datetime
from math import sqrt
from scipy.misc import imread
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QDoubleValidator
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QComboBox
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QCheckBox, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QShortcut, QSlider, QDoubleSpinBox
from PyQt5.QtWidgets import QWidget, QFileDialog, QScrollArea, QMessageBox
from PyQt5.QtCore import pyqtSignal

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

#from slm import version
import version
#from slm.ext.czernike import RZern
from ext.czernike import RZern

"""SLM - spatial light modulator (SLM) controller.
"""


class SLM(QDialog):
    flat_file = None
    flat = None
    flat_on = 0.0
    hologram_geometry = [0, 0, 400, 200]
    rzern = None
    arr = None
    qim = None
    pupil_xy = [0.0, 0.0]
    pupil_rho = 50.0
    angle_xy = [0, 0]
    aberration = np.zeros((15, 1))
    wrap_value = 0xff
    mask2d_on = 0.0
    mask2d_sign = 1.0
    mask3d_on = 0.0
    mask3d_radius = 0.6
    mask3d_height = 1.0
    all_phase=0
    refreshHologramSignal=pyqtSignal()
    def __init__(self, d={}):
        super().__init__(
            parent=None,
            flags=Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

    def load(self, f):
        d = json.load(f)
        self.hologram_geometry = d['hologram_geometry']
        pupil_xy = d['pupil_xy']
        if pupil_xy[0] != self.pupil_xy[0] or pupil_xy[1] != self.pupil_xy[1]:
            # Necessary to make sure that the new pupil is calculated at the
            # right position
            self.rzern = None
            self.pupil_xy = pupil_xy
        self.pupil_rho = d['pupil_rho']
        self.aberration = np.array(d['aberration']).reshape((-1, 1))
        self.wrap_value = d['wrap_value']
        self.mask2d_on = d['mask2d_on']
        self.mask2d_sign = d['mask2d_sign']
        self.mask3d_on = d['mask3d_on']
        self.mask3d_radius = d['mask3d_radius']
        self.mask3d_height = d['mask3d_height']
        self.angle_xy = d['angle_xy']
        self.set_flat(d['flat_file'])
        self.flat_on = d['flat_on']
        
        self.refresh_hologram()
        return d

    def save(self, f, merge=None):
        d = {
            'hologram_geometry': self.hologram_geometry,
            'pupil_xy': self.pupil_xy,
            'pupil_rho': self.pupil_rho,
            'aberration': self.aberration.tolist(),
            'wrap_value': self.wrap_value,
            'mask2d_on': self.mask2d_on,
            'mask2d_sign': self.mask2d_sign,
            'mask3d_on': self.mask3d_on,
            'mask3d_radius': self.mask3d_radius,
            'mask3d_height': self.mask3d_height,
            'flat_file': self.flat_file,
            'flat_on': self.flat_on,
            'angle_xy': self.angle_xy
            }
        if merge:
            merge.update(d)
        else:
            merge = d
        json.dump(merge, f)

    def refresh_hologram(self):
        self.refreshHologramSignal.emit()
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

            print('refresh_hologram(): ALLOCATING arr & qim')
            self.arr = np.ndarray(
                shape=(self.hologram_geometry[3], self.hologram_geometry[2]),
                dtype=np.uint32)
            self.qim = QImage(
                self.arr.data, self.arr.shape[1], self.arr.shape[0],
                QImage.Format_RGB32)

            self.rzern = None

        if self.rzern is None or self.aberration.size != self.rzern.nk:
            print('refresh_hologram(): ALLOCATING Zernike')

            def make_dd(rho, n, x):
                scale = (n/2)/rho
                dd = np.linspace(-scale, scale, n)
                dd -= np.diff(dd)[0]*x
                return dd

            nnew = int((-3 + sqrt(9 - 4*2*(1 - self.aberration.size)))/2)
            self.rzern = RZern(nnew)
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
                print('{} [{:g}, {:g}] {:g}'.format(
                    t, x.min(), x.max(), x.mean()))
            else:
                print(t + ' [0.0, 0.0] 0.0')

        print('refresh_hologram(): repaint')

        printout('flat', self.flat)
        printout('phi', self.phi)
        printout('phi2d', self.phi2d)
        printout('phi3d', self.phi3d)

        # [0, 1] waves
        background = self.flat_on*self.flat
        # [-pi, pi] principal branch rads (zero mean)
        phase = (
            self.phi +
            self.mask2d_on*self.phi2d +
            self.mask3d_on*self.phi3d +
            self.grating)
        #!!! not clean
        self.all_phase = phase.copy()
        
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

        # traceback.print_stack()

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
        value_max = 50

        masks = np.indices((m, n), dtype="float")
        tt = self.angle_xy[0]*(
            masks[0, :, :] - self.pupil_xy[0] - n/2) + self.angle_xy[1]*(
            masks[1, :, :] - self.pupil_xy[1] - m/2)
        tt = tt/value_max*2*np.pi
        tt[self.rho >= 1.0] = 0
        self.grating = np.flipud(tt)

        assert((np.flipud(self.grating)[self.rho>=1.0]==0).all())
        
    def make_phi(self):
        # [-pi, pi] principal branch
        phi = np.pi + self.rzern.eval_grid(self.aberration)
        phi = np.ascontiguousarray(
            phi.reshape(self.arr.shape, order='F'))
        phi[self.rho >= 1.0] = 0
        self.phi = np.flipud(phi)

    def copy_flat_shape(self):
        self.hologram_geometry[2] = self.flat.shape[1]
        self.hologram_geometry[3] = self.flat.shape[0]

    def set_flat(self, fname):
        if fname is None or fname == '':
            self.flat_file = None
            self.flat = 0.0
        else:
            try:
                self.flat_file = fname
                self.flat = np.ascontiguousarray(
                    imread(fname), dtype=np.float)/255
                print(self.flat.shape)
                self.copy_flat_shape()
            except Exception:
                self.flat_file = None
                self.flat = 0.0

        self.refresh_hologram()

    def set_hologram_geometry(self, geometry):
        if isinstance(self.flat, np.ndarray) and len(self.flat.shape) == 2:
            self.hologram_geometry[:2] = geometry[:2]
            self.copy_flat_shape()
        elif geometry is not None:
            self.hologram_geometry[:] = geometry[:]

        self.refresh_hologram()

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
        self.refresh_hologram()

    def set_pupil_rho(self, rho):
        if rho is None:
            self.pupil_rho = min(self.hologram_geometry[2:])/2*.9
        else:
            self.pupil_rho = rho

        self.rzern = None
        self.refresh_hologram()

    def set_mask2d_sign(self, s):
        self.mask2d_sign = s
        self.refresh_hologram()

    def set_mask3d_height(self, s):
        self.mask3d_height = s
        self.refresh_hologram()

    def set_wrap_value(self, wrap_value):
        if wrap_value is None:
            self.wrap_value = 255
        else:
            self.wrap_value = wrap_value

        self.refresh_hologram()

    def set_aberration(self, aberration):
        if aberration is None:
            self.aberration = np.zeros((self.rzern.nk, 1))
        else:
            self.aberration = np.array(aberration).reshape((-1, 1))

        self.refresh_hologram()

    def set_mask2d_on(self, on):
        self.mask2d_on = on
        self.refresh_hologram()

    def set_mask3d_on(self, on):
        self.mask3d_on = on
        self.refresh_hologram()

    def set_flat_on(self, on):
        self.flat_on = on
        self.refresh_hologram()

    def set_mask3d_radius(self, rho):
        if rho is None:
            self.mask3d_radius = 0.6*self.pupil_rho
        else:
            self.mask3d_radius = rho
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
    def __init__(self):
        super().__init__()
        self.slm2 = SLM()
        self.slm2.refreshHologramSignal.connect( 
                lambda: self.refresh_hologram(refresh_slm2=False))
        
    def refresh_hologram(self,refresh_slm2 = True):
        """rewrite refresh hologram for double SLM"""
        if refresh_slm2:
            self.slm2.refresh_hologram()
            print("refresh hologram from slm1")
        else:
            print("refresh hologram from slm2")
        print("Grating values, slm 1 and 2:",self.angle_xy,self.slm2.angle_xy)
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

            print('refresh_hologram(): ALLOCATING arr & qim')
            self.arr = np.ndarray(
                shape=(self.hologram_geometry[3], self.hologram_geometry[2]),
                dtype=np.uint32)
            self.qim = QImage(
                self.arr.data, self.arr.shape[1], self.arr.shape[0],
                QImage.Format_RGB32)

            self.rzern = None

        if self.rzern is None or self.aberration.size != self.rzern.nk:
            print('refresh_hologram(): ALLOCATING Zernike')

            def make_dd(rho, n, x):
                scale = (n/2)/rho
                dd = np.linspace(-scale, scale, n)
                dd -= np.diff(dd)[0]*x
                return dd

            nnew = int((-3 + sqrt(9 - 4*2*(1 - self.aberration.size)))/2)
            self.rzern = RZern(nnew)
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
                print('{} [{:g}, {:g}] {:g}'.format(
                    t, x.min(), x.max(), x.mean()))
            else:
                print(t + ' [0.0, 0.0] 0.0')

        print('refresh_hologram(): repaint')

        printout('flat', self.flat)
        printout('phi', self.phi)
        printout('phi2d', self.phi2d)
        printout('phi3d', self.phi3d)

        # [0, 1] waves
        background = self.flat_on*self.flat
        # [-pi, pi] principal branch rads (zero mean)
        phase = (
            self.phi +
            self.mask2d_on*self.phi2d +
            self.mask3d_on*self.phi3d +
            self.grating + self.slm2.all_phase)
        #Adding second pupil
        #phase += self.slm2.all_phase
        
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
        self.update()
    def load(self):
        pass
    def save(self):
        pass
    
    
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

    def make_grating_tab(self, slm):
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
                slider.blockSignals(True)
                slider.setValue(fto100(r, amp))
                slider.blockSignals(False)
                slm.angle_xy[axis] = r
                slm.refresh_hologram()
            return f

        def update_amp(spinbox, slider, le, i):
            def f():
                amp = float(le.text())
                spinbox.setRange(-amp, amp)
                spinbox.setValue(spinbox.value())
                slider.setValue(fto100(slm.aberration[i, 0], le))
            return f

        def update_spinbox(s, amp):
            def f(t):
                maxrad = float(amp)
                s.setValue(t/multiplier*(2*maxrad) - maxrad)
            return f

        hand1 = update_spinbox(spinbox_x, amp)
        hand2 = update_coeff(slider_x, amp, 0)
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
                        QMessageBox.information(self, 'Error', str(e))

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
        self.slm = slm
        
        if 'window' in settings.keys():
            self.setGeometry(
                settings['window'][0], settings['window'][1],
                settings['window'][2], settings['window'][3])

        self.make_geometry_tab(self.slm)
        self.make_pupil_tab(self.slm)
        self.make_flat_tab(self.slm)
        self.make_wrap_tab(self.slm)
        self.make_2d_tab(self.slm)
        self.make_3d_tab(self.slm)
        self.make_phase_tab()
        self.make_grating_tab(self.slm)
        self.make_aberration_tab(self.slm, self.phase_display)
        self.make_file_tab(self.slm)

        top = QGridLayout()
        if is_parent:
            top.addWidget(self.group_geometry, 0, 0, 1, 2)
        top.addWidget(self.group_pupil, 1, 0)
        top.addWidget(self.group_wrap, 1, 1)
        
        if is_parent:
            top.addWidget(self.group_flat, 2, 0)
        top.addWidget(self.group_2d, 3, 0)
        top.addWidget(self.group_3d, 4, 0)
        top.addWidget(self.group_phase, 0, 2, 2, 1)
        top.addWidget(self.group_grating, 2, 1, 1, 2)
        top.addWidget(self.group_aberration, 3, 1, 3, 2)
        top.addWidget(self.group_file, 5, 0)
        self.setLayout(top)

        self.top = top
        self.slm = slm
        # self.resize(QDesktopWidget().availableGeometry().size()*0.5)

    def closeEvent(self, event):
        if self.close_slm:
            self.slm.close()
        super().close()

    def keyPressEvent(self, event):
        pass

class DoubleControl(QDialog):
    
    close_slm = True
    
    def __init__(self, double_slm, settings):
        super().__init__(parent=None)
        self.setWindowTitle(
            'SLM ' + version.__version__ + ' ' + version.__date__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        self.settings = settings

        if 'window' in settings.keys():
            self.setGeometry(
                settings['window'][0], settings['window'][1],
                settings['window'][2], settings['window'][3])
        
        self.double_slm = double_slm
        
        self.control1 = Control(self.double_slm,settings)
        self.control2 = Control(self.double_slm.slm2,{})
        top = QGridLayout()
        top.addWidget(self.control1,0,0)
        top.addWidget(self.control2,0,1)
        self.setLayout(top)

    def closeEvent(self, event):
        if self.close_slm:
            self.double_slm.slm2.close()
            self.double_slm.close()
        super().close()
        
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


if __name__ == '__main__':
    DOUBLE = True
    app = QApplication(sys.argv)

    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='SLM control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dump', action='store_true')
    parser.add_argument(
        '--console', action='store_true')
    parser.add_argument(
        '--load', type=argparse.FileType('r'), default=None,
        metavar='JSON',
        help='Load a previous configuration file')
    args = parser.parse_args(args[1:])
    if DOUBLE:
        slm = DoubleSLM()
    else:
        slm=SLM()
    slm.show()
    slm.refresh_hologram()

    if args.load:
        d = slm.load(args.load)['control']
        args.load.close()
    else:
        d = {}
    if DOUBLE:
        control = DoubleControl(slm, d)
    else:
        control = Control(slm,d)
    control.show()

    if args.console:
        console = Console(slm, control)
        console.show()

    sys.exit(app.exec_())
