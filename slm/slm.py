#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import logging

from time import time
from datetime import datetime
from math import sqrt

from PyQt5.QtCore import Qt, QMutex, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPainter, QDoubleValidator, QIntValidator, QKeySequence)
from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox,
    QGridLayout, QCheckBox, QVBoxLayout, QApplication, QShortcut,
    QSlider, QDoubleSpinBox, QWidget, QFileDialog, QScrollArea,
    QMessageBox, QTabWidget, QFrame,
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


class MyQDoubleValidator(QDoubleValidator):
    def setFixup(self, val):
        self.fixupval = val

    def fixup(self, txt):
        return str(self.fixupval)


class MyQIntValidator(QIntValidator):
    def setFixup(self, val):
        self.fixupval = val

    def fixup(self, txt):
        return str(self.fixupval)


class Pupil():

    def __init__(self, holo, settings=None):
        self.xv = None
        self.yv = None
        self.name = None
        self.rzern = None
        self.xy = [0.0, 0.0]
        self.rho = 50.0
        self.angle_xy = [0.0, 0.0]
        self.aberration = np.zeros((15, 1))
        self.mask2d_on = 0.0
        self.mask2d_sign = 1.0
        self.mask3d_on = 0.0
        self.mask3d_radius = 0.6
        self.mask3d_height = 1.0
        self.zernike_labels = {}

        self.log = logging.getLogger(self.__class__.__name__)
        self.holo = holo
        self.name = f'pupil {len(self.holo.pupils)}'

        if settings:
            self.dict2parameters(settings)

    def parameters2dict(self):
        return {
            'name': self.name,
            'zernike_labels': self.zernike_labels,
            'xy': self.xy,
            'rho': self.rho,
            'angle_xy': self.angle_xy,
            'aberration': self.aberration.ravel().tolist(),
            'mask2d_on': self.mask2d_on,
            'mask2d_sign': self.mask2d_sign,
            'mask3d_on': self.mask3d_on,
            'mask3d_radius': self.mask3d_radius,
            'mask3d_height': self.mask3d_height,
        }

    def dict2parameters(self, d):
        self.name = d['name']
        self.zernike_labels.update(d['zernike_labels'])
        self.xy = d['xy']
        self.rho = d['rho']
        self.aberration = np.array(d['aberration']).reshape((-1, 1))
        self.mask2d_on = d['mask2d_on']
        self.mask2d_sign = d['mask2d_sign']
        self.mask3d_on = d['mask3d_on']
        self.mask3d_radius = d['mask3d_radius']
        self.mask3d_height = d['mask3d_height']
        self.angle_xy = d['angle_xy']

    def refresh_pupil(self):
        self.log.info(f'refresh_pupil {self.name} START xy:{self.xy}')
        dirty = False
        if (
                self.xv is None or
                self.yv is None or
                self.xv.shape[0] != self.holo.hologram_geometry[3] or
                self.xv.shape[1] != self.holo.hologram_geometry[2]):

            self.log.info(f'refresh_pupil {self.name} allocating Zernike')

            def make_dd(rho, n, x):
                scale = (n/2)/rho
                dd = np.linspace(-scale, scale, n)
                dd -= np.diff(dd)[0]*x
                return dd

            dd1 = make_dd(
                self.rho, self.holo.hologram_geometry[2], self.xy[0])
            dd2 = make_dd(
                self.rho,
                self.holo.hologram_geometry[3], self.xy[1])
            self.xv, self.yv = np.meshgrid(dd1, dd2)
            dirty = True

        if (
                dirty or
                self.rzern is None or
                self.aberration.size != self.rzern.nk):
            nnew = int((-3 + sqrt(9 - 4*2*(1 - self.aberration.size)))/2)
            self.rzern = RZern(nnew)
            self.rzern.make_cart_grid(self.xv, self.yv)
            self.theta = np.arctan2(self.yv, self.xv)
            self.rr = np.sqrt(self.xv**2 + self.yv**2)

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
                    f'refresh_pupil {self.name} {t} ' +
                    f'[{x.min():g}, {x.max():g}] {x.mean():g}')
            else:
                self.log.info(
                    f'refresh_pupil {self.name} ' + str(t) + ' [0.0, 0.0] 0.0')

        printout('phi', self.phi)
        printout('phi2d', self.phi2d)
        printout('phi3d', self.phi3d)

        phase = (
            self.phi +
            self.mask2d_on*self.phi2d +
            self.mask3d_on*self.phi3d +
            self.grating)

        self.log.info(f'refresh_pupil {self.name} END')
        return phase

    def make_phi2d(self):
        # [-pi, pi] principal branch
        phi2d = self.mask2d_sign*self.theta
        phi2d[self.rr >= 1.0] = 0
        self.phi2d = np.flipud(phi2d)

    def make_phi3d(self):
        # [-pi, pi] principal branch
        phi3d = np.zeros_like(self.rr)
        phi3d[self.rr <= self.mask3d_radius] = self.mask3d_height*np.pi
        phi3d[self.rr >= 1] = 0
        # induce zero mean
        phi3d -= phi3d.mean()
        phi3d[self.rr >= 1] = 0
        self.phi3d = np.flipud(phi3d)

    def make_grating(self):
        m = self.holo.hologram_geometry[3]
        n = self.holo.hologram_geometry[2]
        value_max = 15

        masks = np.indices((m, n), dtype="float")
        tt = self.angle_xy[0]*(
            masks[0, :, :] - self.xy[0] - n/2) + self.angle_xy[1]*(
            masks[1, :, :] - self.xy[1] - m/2)
        tt = tt/value_max*2*np.pi
        tt[self.rr >= 1.0] = 0
        self.grating = np.flipud(tt)

    def make_phi(self):
        # [-pi, pi] principal branch
        phi = np.pi + self.rzern.eval_grid(self.aberration)
        phi = np.ascontiguousarray(
            phi.reshape((
                self.holo.hologram_geometry[3],
                self.holo.hologram_geometry[2]), order='F'))
        phi[self.rr >= 1.0] = 0
        self.phi = np.flipud(phi)

    def set_xy(self, xy):
        if self.xy[0] != xy[0]:
            self.xy[0] = xy[0]
            self.xv = None

        if self.xy[1] != xy[1]:
            self.xy[1] = xy[1]
            self.xv = None

        if self.xv is None:
            self.holo.refresh_hologram()

    def set_rho(self, rho):
        if self.rho != rho:
            self.xv = None
            self.rho = rho
        if self.xv is None:
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
            self.mask3d_radius = 0.6*self.rho
        else:
            self.mask3d_radius = rho
        self.holo.refresh_hologram()


class SLM(QDialog):

    refreshHologramSignal = pyqtSignal()

    def __init__(self, settings={}):
        super().__init__(
            parent=None,
            flags=Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.hologram_geometry = [0, 0, 400, 200]
        self.pupils = []

        self.log = logging.getLogger(self.__class__.__name__)
        self.flat_file = None
        self.flat = None
        self.flat_on = 0.0
        self.double_flat_on = False

        self.arr = None
        self.qim = None
        self.wrap_value = 0xff

        if settings:
            self.dict2parameters(settings)

        if len(self.pupils) == 0:
            self.pupils.append(Pupil(self))

        self.refreshHologramSignal.connect(self.update)
        self.refresh_hologram()

    def add_pupil(self):
        p = Pupil(self)
        self.pupils.append(p)
        self.refresh_hologram()
        return p

    def pop_pupil(self):
        self.pupils.pop()
        self.refresh_hologram()

    def parameters2dict(self):
        """Stores all relevant parameters in a dictionary. Useful for saving"""
        d = {
            'hologram_geometry': self.hologram_geometry,
            'wrap_value': self.wrap_value,
            'flat_file': self.flat_file,
            'flat_on': self.flat_on,
            'double_flat_on': self.double_flat_on,
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
        self.double_flat_on = d['double_flat_on']

        self.pupils.clear()
        for ps in d['pupils']:
            self.pupils.append(Pupil(self, ps))

    def load(self, d):
        self.dict2parameters(d)
        self.refresh_hologram()

    def save(self):
        return self.parameters2dict()

    def refresh_hologram(self):
        self.log.info('refresh_hologram START')

        # [0, 1]
        if self.flat_file is None:
            self.flat = 0.
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
            self.log.info('refresh_hologram ALLOCATING arr & qim')
            self.arr = np.zeros(
                shape=(self.hologram_geometry[3], self.hologram_geometry[2]),
                dtype=np.uint32)
            self.qim = QImage(
                self.arr.data, self.arr.shape[1], self.arr.shape[0],
                QImage.Format_RGB32)

        phase = 0
        for p in self.pupils:
            phase += p.refresh_pupil()

        def printout(t, x):
            if isinstance(x, np.ndarray):
                self.log.info(
                    f'refresh_hologram {t} ' +
                    f'[{x.min():g}, {x.max():g}] {x.mean():g}')
            else:
                self.log.info(
                    f'refresh_hologram {t} ' + str(t) + ' [0.0, 0.0] 0.0')

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
        self.gray = gray
        self.arr[:] = gray.astype(np.uint32)*0x010101

        self.log.info(f'refresh_hologram END {str(time())}')
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
                    plt.imread(fname), dtype=np.float)/255
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
            self.hologram_geometry = geometry
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

    def set_double_flat_on(self, on, refresh=True):
        self.double_flat_on = on
        if refresh:
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


class PhaseDisplay(QWidget):

    def __init__(self):
        super().__init__()

        self.dirty = False
        self.size = [0, 0]
        self.phase = None
        self.arr = None
        self.qim = None
        self.rzern = None

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
                drawn = np.frombuffer(
                            fig10.canvas.tostring_rgb(),
                            dtype=np.uint8).reshape(
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


class MatplotlibWindow(QFrame):

    def __init__(self, slm, parent=None, toolbar=True, figsize=None):
        super().__init__(parent)

        self.shape = None
        self.im = None

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
        self.slm = slm
        self.figure.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    def update_array(self):
        if self.slm.gray is None:
            return
        if self.im is None or self.slm.gray.shape != self.shape:
            self.im = self.ax.imshow(
                self.slm.gray, cmap="gray", vmin=0, vmax=0xff)
            self.ax.axis("off")
            self.shape = self.slm.gray.shape
        self.im.set_data(self.slm.gray)
        self.canvas.draw()


class PupilPanel(QFrame):
    def __init__(self, pupil, ptabs, parent=None):
        """Subclass for a control GUI.
        Parameters:
            slm: SLM instance
            settings: dict, saved settings
            is_parent: bool. Useful in the case of doublepass to determine
                for instance which widget determines the overall geometry"""
        super().__init__(parent)
        self.refresh_gui = []
        self.pupil = pupil
        self.ptabs = ptabs

        self.make_pupil_tab()
        self.make_2d_tab()
        self.make_3d_tab()
        self.make_phase_tab()
        self.make_grating_tab()
        self.make_aberration_tab()

        top = QGridLayout()
        self.top = top
        top.addWidget(self.group_phase, 0, 0, 4, 1)
        top.addWidget(self.group_pupil, 0, 1)
        top.addWidget(self.group_grating, 1, 1)
        top.addWidget(self.group_2d, 2, 1)
        top.addWidget(self.group_3d, 3, 1)
        top.addWidget(self.group_aberration, 4, 0, 2, 2)
        self.setLayout(top)
        self.top = top

        self.pindex = self.ptabs.count()
        self.ptabs.addTab(self, self.pupil.name)

    def helper_boolupdate(self, mycallback):
        def f(i):
            mycallback(i)
        return f

    def get_pupil(self):
        return self.pupil

    def make_pupil_tab(self):
        def handle_xy(ind, le):
            def f():
                p = self.get_pupil()
                try:
                    fval = float(le.text())
                except Exception:
                    le.setText(str(p.xy[ind]))
                    return
                xy = [p.xy[0], p.xy[1]]
                xy[ind] = fval
                p.set_xy(xy)
                le.setText(str(p.xy[ind]))
            return f

        def handle_rho(ind, le):
            def f():
                try:
                    fval = float(le.text())
                except Exception:
                    le.setText(str(self.pupil.rho))
                    return
                self.pupil.set_rho(fval)
                le.setText(str(self.pupil.rho))
            return f

        group = QGroupBox('Pupil')
        l1 = QGridLayout()

        def help1(txt, mini, handler, curval, i):
            l1.addWidget(QLabel(txt), 0, 2*i)
            le = QLineEdit(str(curval))
            le.editingFinished.connect(handler(i, le))
            le.setMaximumWidth(50)
            val = MyQDoubleValidator()
            val.setFixup(curval)
            le.setValidator(val)
            if mini:
                val.setBottom(mini)
            l1.addWidget(le, 0, 2*i + 1)
            return le

        lex = help1('x0', None, handle_xy, self.pupil.xy[0], 0)
        ley = help1('y0', None, handle_xy, self.pupil.xy[1], 1)
        lerho = help1('radius', 10, handle_rho, self.pupil.rho, 2)
        lename = QLineEdit(self.pupil.name)
        l1.addWidget(lename, 1, 0, 1, 6)

        def fname():
            def f():
                self.pupil.name = lename.text()
                self.ptabs.setTabText(self.pindex, self.pupil.name)
            return f

        lename.editingFinished.connect(fname())

        def make_f():
            def t1():
                return self.pupil.xy, self.pupil.rho

            def f():
                xy, rho = t1()
                for p in (lex, ley, lerho):
                    p.blockSignals(True)
                lex.setText(str(xy[0]))
                ley.setText(str(xy[1]))
                lerho.setText(str(rho))
                for p in (lex, ley, lerho):
                    p.blockSignals(False)
            return f

        self.refresh_gui.append(make_f())

        group.setLayout(l1)
        self.group_pupil = group

    def make_2d_tab(self):
        g = QGroupBox('2D STED')
        l1 = QGridLayout()
        c = QCheckBox('2D on')
        c.setChecked(self.pupil.mask2d_on)
        c.toggled.connect(self.helper_boolupdate(
            self.pupil.set_mask2d_on))
        l1.addWidget(c, 0, 0)
        sign2d = QComboBox()
        sign2d.addItem('+1')
        sign2d.addItem('-1')
        if self.pupil.mask2d_sign == 1:
            sign2d.setCurrentIndex(0)
        else:
            sign2d.setCurrentIndex(1)

        def toggle_float(fun):
            def f(val):
                if val == 0:
                    fun(float(1))
                else:
                    fun(float(-1))
            return f

        sign2d.activated.connect(toggle_float(self.pupil.set_mask2d_sign))
        l1.addWidget(sign2d, 0, 1)
        g.setLayout(l1)

        self.group_2d = g

        def f():
            def f():
                c.setChecked(self.pupil.mask2d_on)
                if self.pupil.mask2d_sign == 1:
                    sign2d.setCurrentIndex(0)
                else:
                    sign2d.setCurrentIndex(1)
            return f

        self.refresh_gui.append(f())

    def make_3d_tab(self):
        g = QGroupBox('3D STED')

        def update_radius(slider, what):
            def f(r):
                slider.setValue(int(r*100))
                what(r)
            return f

        def update_spinbox(s):
            def f(t):
                s.setValue(t/100)
            return f

        l1 = QGridLayout()
        c = QCheckBox('3D on')
        c.setChecked(self.pupil.mask3d_on)
        c.toggled.connect(self.helper_boolupdate(
            self.pupil.set_mask3d_on))
        l1.addWidget(c, 0, 0)
        slider1 = QSlider(Qt.Horizontal)
        slider1.setMinimum(0)
        slider1.setMaximum(100)
        slider1.setFocusPolicy(Qt.StrongFocus)
        slider1.setTickPosition(QSlider.TicksBothSides)
        slider1.setTickInterval(20)
        slider1.setSingleStep(0.1)
        slider1.setValue(int(100*self.pupil.mask3d_radius))
        spinbox1 = QDoubleSpinBox()
        spinbox1.setRange(0.0, 1.0)
        spinbox1.setSingleStep(0.01)
        spinbox1.setValue(self.pupil.mask3d_radius)
        slider1.valueChanged.connect(update_spinbox(spinbox1))

        spinbox1.valueChanged.connect(update_radius(
            slider1, self.pupil.set_mask3d_radius))
        l1.addWidget(QLabel('radius'), 0, 1)
        l1.addWidget(spinbox1, 0, 2)
        l1.addWidget(slider1, 0, 3)

        slider2 = QSlider(Qt.Horizontal)
        slider2.setMinimum(0)
        slider2.setMaximum(200)
        slider2.setFocusPolicy(Qt.StrongFocus)
        slider2.setTickPosition(QSlider.TicksBothSides)
        slider2.setTickInterval(40)
        slider2.setSingleStep(0.1)
        slider2.setValue(int(100*self.pupil.mask3d_height))
        spinbox2 = QDoubleSpinBox()
        spinbox2.setRange(0.0, 2.0)
        spinbox2.setSingleStep(0.01)
        spinbox2.setValue(self.pupil.mask3d_height)
        slider2.valueChanged.connect(update_spinbox(spinbox2))

        spinbox2.valueChanged.connect(update_radius(
            slider2, self.pupil.set_mask3d_height))
        l1.addWidget(QLabel('height'), 1, 1)
        l1.addWidget(spinbox2, 1, 2)
        l1.addWidget(slider2, 1, 3)
        g.setLayout(l1)

        self.group_3d = g

        def f():
            def f():
                c.setChecked(self.pupil.mask3d_on)
                for p in (slider1, slider2, spinbox1, spinbox2):
                    p.blockSignals(True)
                spinbox1.setValue(self.pupil.mask3d_radius)
                slider1.setValue(int(100*self.pupil.mask3d_radius))
                spinbox2.setValue(self.pupil.mask3d_height)
                slider2.setValue(int(100*self.pupil.mask3d_height))
                for p in (slider1, slider2, spinbox1, spinbox2):
                    p.blockSignals(False)
            return f

        self.refresh_gui.append(f())

    def make_phase_tab(self):
        g = QGroupBox('Phase')
        phase_display = PhaseDisplay()
        l1 = QGridLayout()
        l1.addWidget(phase_display, 0, 0)
        g.setLayout(l1)

        self.group_phase = g
        self.phase_display = phase_display

    def make_aberration_tab(self):
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

        multiplier = 100

        top = QGroupBox('Zernike aberrations')
        toplay = QGridLayout()
        top.setLayout(toplay)
        labzm = QLabel('max radial order')
        lezm = QLineEdit(str(self.pupil.rzern.n))
        lezm.setMaximumWidth(50)
        val = MyQIntValidator(1, 255)
        val.setFixup(self.pupil.rzern.n)
        lezm.setValidator(val)
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
                self.pupil.aberration[ind, 0] = r
                self.pupil.set_aberration(self.pupil.aberration)

                self.phase_display.update_phase(
                    self.pupil.rzern.n, self.pupil.aberration)
                self.phase_display.update()
            return f

        def update_amp(spinbox, slider, le, i):
            def f():

                amp = float(le.text())
                spinbox.setRange(-amp, amp)
                spinbox.setValue(spinbox.value())
                slider.setValue(fto100(self.pupil.aberration[i, 0], le))
            return f

        def update_zlabel(le, i):
            def f():
                self.pupil.zernike_labels[str(i)] = le.text()
            return f

        def update_spinbox(s, amp):
            def f(t):
                maxrad = float(amp.text())
                s.setValue(t/multiplier*(2*maxrad) - maxrad)
            return f

        def update_zernike_rows(mynk=None):
            if mynk is None:
                mynk = self.pupil.rzern.nk
            ntab = self.pupil.rzern.ntab
            mtab = self.pupil.rzern.mtab
            if len(zernike_rows) < mynk:
                for i in range(len(zernike_rows), mynk):
                    lab = QLabel(
                        'Z<sub>{}</sub> Z<sub>{}</sub><sup>{}</sup>'.format(
                            i + 1, ntab[i], mtab[i]))
                    slider = QSlider(Qt.Horizontal)
                    spinbox = QDoubleSpinBox()
                    maxamp = max((4, self.pupil.aberration[i, 0]))
                    try:
                        zname = self.pupil.zernike_labels[str(i)]
                    except KeyError:
                        zname = default_zernike_name(i + 1, ntab[i], mtab[i])
                        self.pupil.zernike_labels[str(i)] = zname
                    lbn = QLineEdit(zname)
                    lbn.setMaximumWidth(120)
                    amp = QLineEdit(str(maxamp))
                    amp.setMaximumWidth(50)
                    val = MyQDoubleValidator()
                    val.setBottom(0.1)
                    val.setFixup(str(maxamp))
                    amp.setValidator(val)

                    slider.setMinimum(0)
                    slider.setMaximum(multiplier)
                    slider.setFocusPolicy(Qt.StrongFocus)
                    slider.setTickPosition(QSlider.TicksBothSides)
                    slider.setTickInterval(20)
                    slider.setSingleStep(0.1)
                    slider.setValue(fto100(self.pupil.aberration[i, 0], amp))
                    spinbox.setRange(-maxamp, maxamp)
                    spinbox.setSingleStep(0.01)
                    spinbox.setValue(self.pupil.aberration[i, 0])

                    hand1 = update_spinbox(spinbox, amp)
                    hand2 = update_coeff(slider, i, amp)
                    hand3 = update_amp(spinbox, slider, amp, i)
                    hand4 = update_zlabel(lbn, i)
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
            self.pupil.aberration[:] = 0
            self.pupil.set_aberration(self.pupil.aberration)
            self.phase_display.update_phase(
                self.pupil.rzern.n, self.pupil.aberration)
            self.phase_display.update()
            update_zernike_rows(0)
            update_zernike_rows()

        def change_radial():
            try:
                ival = int(lezm.text())
            except Exception:
                lezm.setText(str(self.pupil.rzern.n))
                return
            n = (ival + 1)*(ival + 2)//2
            newab = np.zeros((n, 1))
            minn = min((n, self.pupil.rzern.n))
            newab[:minn, 0] = self.pupil.aberration[:minn, 0]
            self.pupil.set_aberration(newab)

            update_zernike_rows()
            self.phase_display.update_phase(
                self.pupil.rzern.n, self.pupil.aberration)
            self.phase_display.update()
            lezm.setText(str(self.pupil.rzern.n))

        self.phase_display.update_phase(
            self.pupil.rzern.n, self.pupil.aberration)
        zernike_rows = list()
        update_zernike_rows()

        reset.clicked.connect(reset_fun)
        lezm.editingFinished.connect(change_radial)

        self.group_aberration = top

        def f():
            def f():
                update_zernike_rows(0)
                update_zernike_rows()
                self.phase_display.update_phase(
                    self.pupil.rzern.n, self.pupil.aberration)
                self.phase_display.update()
            return f

        self.refresh_gui.append(f())

    def make_grating_tab(self):
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
        slider_x.setValue(fto100(self.pupil.angle_xy[0], amp))

        spinbox_x = QDoubleSpinBox()
        spinbox_x.setRange(-amp, amp)
        spinbox_x.setSingleStep(0.01)
        spinbox_x.setValue(self.pupil.angle_xy[0])

        # y position
        slider_y = QSlider(Qt.Horizontal)
        slider_y.setMinimum(0)
        slider_y.setMaximum(multiplier)
        slider_y.setSingleStep(0.01)
        slider_y.setValue(fto100(self.pupil.angle_xy[1], amp))

        spinbox_y = QDoubleSpinBox()
        spinbox_y.setRange(-amp, amp)
        spinbox_y.setSingleStep(0.01)
        spinbox_y.setValue(self.pupil.angle_xy[1])

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
                self.pupil.set_anglexy(r, axis)
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

        def f():
            def p1():
                return self.pupil.angle_xy[0], self.pupil.angle_xy[1]

            def f():
                for p in (spinbox_x, spinbox_y, slider_x, slider_y):
                    p.blockSignals(True)
                xy = p1()
                spinbox_x.setValue(xy[0])
                spinbox_y.setValue(xy[1])
                slider_x.setValue(fto100(xy[0], amp))
                slider_y.setValue(fto100(xy[1], amp))
                for p in (spinbox_x, spinbox_y, slider_x, slider_y):
                    p.blockSignals(False)
            return f

        self.refresh_gui.append(f())

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
                'pupil': 0,
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
    # TODO fixme
    p = params['control']['SingleZernike']

    noll_min = p['min']
    noll_max = p['max']
    minclude = np.array(p['include'], dtype=np.int)
    mexclude = np.array(p['exclude'], dtype=np.int)

    mrange = np.arange(noll_min, noll_max + 1, dtype=np.int)
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
        self.pupil_index = pars['control']['SingleZernike']['pupil']
        self.pupil = self.slm.pupils[self.pupil_index]

        nz = self.pupil.aberration.size

        if pars['control']['SingleZernike']['all'] == 1:
            indices = np.arange(1, nz + 1)
        else:
            indices = get_noll_indices(pars)

        self.indices = indices
        ndof = indices.size

        self.ndof = ndof
        self.h5f = h5f
        self.z0 = self.pupil.aberration.ravel()

        self.h5_save('slm', json.dumps(self.slm.save()))
        self.h5_save('indices', self.indices)
        self.h5_save('flat', self.slm.flat)
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
        z1 = np.zeros(self.pupil.aberration.size)
        z1[self.indices - 1] = x[:]
        if self.P is not None:
            z2 = np.dot(self.P, z1 + self.z0)
        else:
            z2 = z1 + self.z0
        self.pupil.set_aberration(z2.reshape(-1, 1))

        self.h5_append('flat_on', bool(self.slm.flat_on))
        self.h5_append('x', x)
        self.h5_append('z2', z2)


class ControlWindow(QDialog):

    sig_acquire = pyqtSignal(tuple)
    sig_release = pyqtSignal(tuple)

    def __init__(self, slm, settings={}):
        super().__init__(parent=None)
        self.pupilPanels = []
        self.refresh_gui = []
        self.can_close = True
        self.close_slm = True

        self.slm = slm
        self.settings = settings
        self.mutex = QMutex()

        self.setWindowTitle(
            'SLM ' + version.__version__ + ' ' + version.__date__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        if 'controlwindow' in settings.keys():
            self.setGeometry(
                settings['controlwindow'][0], settings['controlwindow'][1],
                settings['controlwindow'][2], settings['controlwindow'][3])

        self.pupilsTab = QTabWidget()
        for p in self.slm.pupils:
            pp = PupilPanel(p, self.pupilsTab)
            self.pupilPanels.append(pp)

        self.make_geometry_tab()
        self.make_general_display()
        self.make_parameters_group()

        self.top = QGridLayout()
        self.top.addWidget(self.display, 0, 0)
        self.top.addWidget(self.pupilsTab, 0, 1, 2, 1)
        self.top.addWidget(self.parametersGroup, 1, 0)
        self.setLayout(self.top)

        def make_release_hand():
            def f(t):
                for pp in self.pupilPanels:
                    for f in pp.refresh_gui:
                        f()
                for f in self.refresh_gui:
                    f()
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
        self.slm.refresh_hologram()

    @staticmethod
    def helper_boolupdate(mycallback, myupdate):
        def f(i):
            mycallback(i)
            myupdate()
        return f

    def make_general_display(self):
        self.display = MatplotlibWindow(self.slm, figsize=(8, 6))
        self.slm.refreshHologramSignal.connect(self.display.update_array)

    def make_file_tab(self):
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
                            self.load(json.load(f))
                    except Exception as e:
                        QMessageBox.information(self, 'Error', str(e))
            return myf1

        def helper_save():
            def myf1():
                fdiag, _ = QFileDialog.getSaveFileName(
                    directory=datetime.now().strftime('%Y%m%d_%H%M%S.json'))
                if fdiag:
                    try:
                        with open(fdiag, 'w') as f:
                            json.dump(self.save(), f)
                    except Exception as e:
                        QMessageBox.information(self, 'Error', str(e))
            return myf1

        load.clicked.connect(helper_load())
        save.clicked.connect(helper_save())

        self.group_file = g

    def load(self, d):
        self.setGeometry(*d['controlwindow']['geometry'])
        self.slm.load(d['slm'])
        self.pupilsTab.clear()
        while self.pupilsTab.count():
            self.pupilsTab.removeTab(self.pupilsTab.count() - 1)
        for i in range(len(self.slm.pupils)):
            p = self.slm.pupils[i]
            pp = PupilPanel(p, self.pupilsTab)
            self.pupilPanels.append(pp)
        for pp in self.pupilPanels:
            for f in pp.refresh_gui:
                f()
        for f in self.refresh_gui:
            f()

    def save(self):
        curg = self.geometry()
        return {
            'controlwindow': {
                'geometry': [
                    curg.x(), curg.y(), curg.width(), curg.height()],
                },
            'slm': self.slm.save(),
            }

    def make_flat_tab(self):
        def helper_load_flat1():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName()
                if fdiag:
                    self.slm.set_flat(fdiag)
            return myf1

        g = QGroupBox('Flattening')
        l1 = QGridLayout()
        cboxlf = QCheckBox('flat on')
        cboxlf.toggled.connect(self.helper_boolupdate(
            self.slm.set_flat_on, self.slm.update))
        cboxlf.setChecked(self.slm.flat_on)
        l1.addWidget(cboxlf, 0, 0)
        loadbut = QPushButton('load')
        loadbut.clicked.connect(helper_load_flat1())
        l1.addWidget(loadbut, 0, 1)
        g.setLayout(l1)
        self.group_flat = g

        def f():
            def f():
                cboxlf.setChecked(self.slm.flat_on)
            return f

        self.refresh_gui.append(f())

    def make_geometry_tab(self):
        def handle_geometry(ind, le):
            def f():
                try:
                    ival = int(le.text())
                except Exception:
                    le.setText(str(self.slm.hologram_geometry[ind]))
                    return
                self.slm.hologram_geometry[ind] = ival
                self.slm.set_hologram_geometry(self.slm.hologram_geometry)
                le.setText(str(self.slm.hologram_geometry[ind]))
            return f

        group = QGroupBox('Geometry')
        l1 = QGridLayout()
        labels = ['x', 'y', 'width', 'height']
        mins = [None, None, 100, 100]
        les = []

        for i, tup in enumerate(zip(labels, mins)):
            txt, mini = tup
            l1.addWidget(QLabel(txt), 0, 2*i)
            le = QLineEdit(str(self.slm.hologram_geometry[i]))
            le.editingFinished.connect(handle_geometry(i, le))
            le.setMaximumWidth(50)
            val = MyQIntValidator()
            val.setFixup(self.slm.hologram_geometry[i])
            le.setValidator(val)
            if mini:
                val.setBottom(mini)
            l1.addWidget(le, 0, 2*i + 1)
            les.append(le)

        def f():
            def t():
                return self.slm.hologram_geometry

            def f():
                g = t()
                for i, le in enumerate(les):
                    le.blockSignals(True)
                    le.setText(str(g[i]))
                    le.blockSignals(False)
            return f

        self.refresh_gui.append(f())
        group.setLayout(l1)
        self.group_geometry = group

    def make_wrap_tab(self):
        g = QGroupBox('Wrap value')
        l1 = QGridLayout()
        lewrap = QLineEdit(str(self.slm.wrap_value))
        lewrap.setMaximumWidth(50)
        val = MyQIntValidator(1, 255)
        val.setFixup(self.slm.wrap_value)
        lewrap.setValidator(val)

        def handle_wrap(lewrap1):
            def f():
                try:
                    ival = int(lewrap1.text())
                except Exception:
                    lewrap1.setText(str(self.slm.wrap_value))
                    return
                self.slm.set_wrap_value(ival)
                self.slm.update()
                lewrap1.setText(str(self.slm.wrap_value))
            return f

        lewrap.editingFinished.connect(handle_wrap(lewrap))
        l1.addWidget(lewrap, 0, 0)
        g.setLayout(l1)
        self.group_wrap = g

        def f():
            def f():
                lewrap.setText(str(self.slm.wrap_value))
            return f

        self.refresh_gui.append(f())

    def make_parameters_group(self):
        self.make_file_tab()
        self.make_flat_tab()
        self.make_wrap_tab()
        self.doubleFlatOnCheckBox = QCheckBox("Double flat on")
        self.doubleFlatOnCheckBox.setChecked(self.slm.double_flat_on)
        self.doubleFlatOnCheckBox.toggled.connect(self.slm.set_double_flat_on)
        bpls = QPushButton('+ pupil')
        bmin = QPushButton('- pupil')

        group = QGroupBox("Parameters")
        top = QGridLayout()
        top.addWidget(self.group_geometry, 0, 0, 1, 3)

        top.addWidget(self.group_flat, 1, 0)
        top.addWidget(self.group_wrap, 1, 1)
        top.addWidget(self.doubleFlatOnCheckBox, 1, 2)
        top.addWidget(bmin, 2, 0)
        top.addWidget(bpls, 2, 1)

        top.addWidget(self.group_file, 3, 0, 1, 3)
        group.setLayout(top)
        self.parametersGroup = group

        def fp():
            def f():
                p = self.slm.add_pupil()
                pp = PupilPanel(p, self.pupilsTab)
                self.pupilPanels.append(pp)
            return f

        def fm():
            def f():
                if len(self.slm.pupils) == 1:
                    return
                self.pupilsTab.removeTab(len(self.slm.pupils) - 1)
                self.pupilPanels.pop()
                self.slm.pop_pupil()
            return f

        bpls.clicked.connect(fp())
        bmin.clicked.connect(fm())

    def closeEvent(self, event):
        if self.can_close:
            if self.close_slm:
                self.slm.close()
            event.accept()
        else:
            event.ignore()

    def acquire_control(self, h5f):
        self.sig_acquire.emit((h5f,))
        pars = {
            'control': {
                'SingleZernike': {
                    'include': [],
                    'exclude': [1, 2, 3, 4],
                    'min': 5,
                    'max': 6,
                    'all': 0,
                    'pupil': 0,
                    },
                }
            }
        return SingleZernikeControl(self.slm, pars=pars, h5f=h5f)

    def release_control(self, control, h5f):
        self.sig_release.emit((control, h5f))


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
    parser.add_argument(
        '--slm-settings', type=argparse.FileType('r'), default=None,
        metavar='JSON', help='Load a previous configuration file')


def new_slm_window(app, args, settings=None):
    slm = SLM()
    slm.show()

    cwin = ControlWindow(slm)
    cwin.show()

    if args.slm_settings is not None and settings is not None:
        raise RuntimeError('Both file and dict settings specified')

    if args.slm_settings is not None:
        d = json.loads(args.slm_settings.read())
        args.slm_settings = args.slm_settings.name
        cwin.load(d)
    elif settings is not None:
        cwin.load(settings)

    return cwin


if __name__ == '__main__':
    app = QApplication(sys.argv)

    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='SLM control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--no-file-log', action='store_true')
    parser.add_argument('--file-log', action='store_false', dest='no_file_log')
    parser.set_defaults(no_file_log=True)
    parser.add_argument(
        '--load', type=argparse.FileType('r'), default=None,
        metavar='JSON',
        help='Load a previous configuration file')
    args = parser.parse_args(args[1:])

    if not args.no_file_log:
        fn = datetime.now().strftime(
            '%Y%m%d-%H%M%S-' + str(os.getpid()) + '.log')
        logging.basicConfig(filename=fn, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    slm = SLM()
    slm.show()

    cwin = ControlWindow(slm)
    cwin.show()

    if args.load:
        d = json.loads(args.load.read())
        cwin.load(d)

    if args.console:
        console = Console(slm, cwin)
        console.show()

    sys.exit(app.exec_())
