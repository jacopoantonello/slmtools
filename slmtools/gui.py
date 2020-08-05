#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from math import sqrt
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import (FigureCanvas,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5.QtCore import QMutex, Qt, pyqtSignal
from PyQt5.QtGui import (QDoubleValidator, QImage, QIntValidator, QKeySequence,
                         QPainter)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QErrorMessage, QFileDialog,
                             QFrame, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QMainWindow, QPushButton, QScrollArea, QShortcut,
                             QSlider, QSplitter, QTabWidget, QVBoxLayout,
                             QWidget)

from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from slmtools import version
from zernike import RZern
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


def transform_pupil(rzern, alpha=0., flipx=False, flipy=False):
    if alpha != 0.:
        R = rzern.make_rotation(alpha)
    else:
        R = 1

    if flipx:
        Fx = rzern.make_xflip()
    else:
        Fx = 1

    if flipy:
        Fy = rzern.make_yflip()
    else:
        Fy = 1

    return np.dot(Fy, np.dot(Fx, R))


class Pupil():
    "A pupil within the SLM window."

    def_pars = {
        'name': 'pupil',
        'enabled': 1,
        'zernike_labels': {},
        'xy': [0.0, 0.0],
        'rho': 50.0,
        'angle_xy': [0.0, 0.0],
        'aberration': np.zeros((15, 1)),
        'mask2d_on': 0,
        'mask2d_sign': 1.0,
        'mask2d_mul': 1.0,
        'mask3d_on': 0,
        'mask3d_radius': 0.63,
        'mask3d_height': 1.0,
        'align_grid_on': 0,
        'align_grid_pitch': 16,
        'flipx': 0,
        'flipy': 0,
        'rotate': 0.0,
    }

    def __init__(self, holo, pars={}):
        self.xv = None
        self.yv = None
        self.rzern = None
        self.mask = None
        self.zernike_labels = {}

        self.log = logging.getLogger(self.__class__.__name__)
        self.holo = holo

        self.dict2parameters({**deepcopy(self.def_pars), **deepcopy(pars)})

    def parameters2dict(self):
        return {
            'name': self.name,
            'enabled': self.enabled,
            'zernike_labels': self.zernike_labels,
            'xy': self.xy,
            'rho': self.rho,
            'angle_xy': self.angle_xy,
            'aberration': self.aberration.ravel().tolist(),
            'mask2d_on': self.mask2d_on,
            'mask2d_sign': self.mask2d_sign,
            'mask2d_mul': self.mask2d_mul,
            'mask3d_on': self.mask3d_on,
            'mask3d_radius': self.mask3d_radius,
            'mask3d_height': self.mask3d_height,
            'align_grid_on': self.align_grid_on,
            'align_grid_pitch': self.align_grid_pitch,
            'flipx': self.flipx,
            'flipy': self.flipy,
            'rotate': self.rotate,
        }

    def dict2parameters(self, d):
        self.name = d['name']
        self.enabled = d['enabled']
        self.zernike_labels.update(d['zernike_labels'])
        self.xy = d['xy']
        self.rho = d['rho']
        self.aberration = np.array(d['aberration']).reshape((-1, 1))
        self.mask2d_on = d['mask2d_on']
        self.mask2d_sign = d['mask2d_sign']
        self.mask2d_mul = d['mask2d_mul']
        self.mask3d_on = d['mask3d_on']
        self.mask3d_radius = d['mask3d_radius']
        self.mask3d_height = d['mask3d_height']
        self.angle_xy = d['angle_xy']
        self.align_grid_on = d['align_grid_on']
        self.align_grid_pitch = d['align_grid_pitch']
        self.flipx = d['flipx']
        self.flipy = d['flipy']
        self.rotate = d['rotate']

    def refresh_pupil(self):
        self.log.info(f'refresh_pupil {self.name} START xy:{self.xy}')

        if not self.enabled:
            if self.rzern is None:
                nnew = int(
                    (-3 + sqrt(9 - 4 * 2 * (1 - self.aberration.size))) / 2)
                self.rzern = RZern(nnew)
            self.mask = np.ones((self.holo.hologram_geometry[3],
                                 self.holo.hologram_geometry[2]),
                                dtype=np.bool)
            self.log.info(f'refresh_pupil {self.name} END')
            return 0

        dirty = False
        if (self.xv is None or self.yv is None
                or self.xv.shape[0] != self.holo.hologram_geometry[3]
                or self.xv.shape[1] != self.holo.hologram_geometry[2]
                or self.mask is None):

            self.log.debug(f'refresh_pupil {self.name} allocating Zernike')

            def make_dd(rho, n, x):
                scale = (n / 2) / rho
                dd = np.linspace(-scale, scale, n)
                dd -= (dd[1] - dd[0]) * x
                return dd

            dd1 = make_dd(self.rho, self.holo.hologram_geometry[2], self.xy[0])
            dd2 = make_dd(self.rho, self.holo.hologram_geometry[3], self.xy[1])
            self.xv, self.yv = np.meshgrid(dd1, dd2)
            dirty = True

        if (dirty or self.rzern is None
                or self.aberration.size != self.rzern.nk):
            nnew = int(
                np.ceil(
                    (-3 + sqrt(9 - 4 * 2 * (1 - self.aberration.size))) / 2))
            self.rzern = RZern(nnew)
            self.R = transform_pupil(self.rzern, self.rotate, self.flipx,
                                     self.flipy)
            self.rzern.make_cart_grid(self.xv, self.yv)
            self.theta = np.arctan2(self.yv, self.xv)
            self.rr = np.sqrt(self.xv**2 + self.yv**2)
            self.mask = self.rr >= 1.
            if self.aberration.size != self.rzern.nk:
                tmp = np.zeros((self.rzern.nk, 1))
                tmp[:self.aberration.size, 0] = self.aberration.ravel()
                self.aberration = tmp

        self.make_phi2d()
        assert (np.all(np.isfinite(self.phi2d)))
        self.make_phi3d()
        assert (np.all(np.isfinite(self.phi3d)))
        self.make_phi()
        assert (np.all(np.isfinite(self.phi)))
        self.make_grating()
        assert (np.all(np.isfinite(self.grating)))
        self.make_align_grid()
        assert (np.all(np.isfinite(self.align_grid)))

        def printout(t, x):
            if isinstance(x, np.ndarray):
                self.log.debug(f'refresh_pupil {self.name} {t} ' +
                               f'[{x.min():g}, {x.max():g}] {x.mean():g}')
            else:
                self.log.debug(f'refresh_pupil {self.name} ' + str(t) +
                               ' [0.0, 0.0] 0.0')

        printout('phi', self.phi)
        printout('phi2d', self.phi2d)
        printout('phi3d', self.phi3d)

        phase = (self.phi + self.mask2d_on * self.phi2d +
                 self.mask3d_on * self.phi3d + self.grating + self.align_grid)

        assert (np.all(phase[self.mask] == 0.))
        self.log.info(f'refresh_pupil {self.name} END')

        return phase

    def make_phi2d(self):
        # [-pi, pi] principal branch
        theta = (self.theta + np.pi) / (2 * np.pi)
        theta = np.power(theta, self.mask2d_mul)
        phi2d = self.mask2d_sign * theta * (2 * np.pi) - np.pi
        phi2d[self.mask] = 0
        self.phi2d = phi2d

    def make_phi3d(self):
        # [-pi, pi] principal branch
        phi3d = np.zeros_like(self.rr)
        phi3d[self.rr <= self.mask3d_radius] = self.mask3d_height * np.pi
        phi3d[self.rr >= 1] = 0
        # induce zero mean
        phi3d -= phi3d.mean()
        phi3d[self.rr >= 1] = 0
        self.phi3d = phi3d

    def make_align_grid(self):
        if self.align_grid_on:
            pitch = self.align_grid_pitch
            grid = np.zeros_like(self.rr)
            assert (len(self.rr.shape) == 2)
            for j in range(0, grid.shape[0], pitch):
                slice1 = grid[j:j + pitch, :]
                for i in range(grid.shape[1]):
                    for k in range(pitch):
                        ind = (k + ((j // pitch) % 2) * pitch)
                        slice1[:, ind::2 * pitch] = np.pi
            grid[self.mask] = 0
            self.align_grid = grid
        else:
            self.align_grid = 0

    def make_grating(self):
        xv = self.xv
        yv = self.yv
        coeffs = self.angle_xy
        grating = np.pi * (coeffs[0] * xv + coeffs[1] * yv)
        grating[self.mask] = 0
        self.grating = grating

    def make_phi(self):
        # [-pi, pi] principal branch
        a = np.dot(self.R, self.aberration)
        phi = np.pi + self.rzern.eval_grid(a)
        phi = np.ascontiguousarray(
            phi.reshape((self.holo.hologram_geometry[3],
                         self.holo.hologram_geometry[2]),
                        order='F'))
        phi[self.mask] = 0
        self.phi = phi

    def set_enabled(self, enabled, update=True):
        if enabled:
            self.enabled = 1
        else:
            self.enabled = 0
        self.xv = None
        if update:
            self.holo.refresh_hologram()

    def set_flipx(self, b):
        if b:
            self.flipx = 1
        else:
            self.flipx = 0
        self.xv = None
        self.holo.refresh_hologram()

    def set_flipy(self, b):
        if b:
            self.flipy = 1
        else:
            self.flipy = 0
        self.xv = None
        self.holo.refresh_hologram()

    def set_rotate(self, a):
        self.rotate = a
        self.xv = None
        self.holo.refresh_hologram()

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

    def set_mask2d_mul(self, s):
        self.mask2d_mul = s
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
            self.mask3d_radius = 0.6 * self.rho
        else:
            self.mask3d_radius = rho
        self.holo.refresh_hologram()

    def set_align_grid_on(self, on):
        self.align_grid_on = on
        self.holo.refresh_hologram()

    def set_align_grid_pitch(self, p):
        if p < 1:
            self.align_grid_pitch = 1
        else:
            self.align_grid_pitch = p
        self.holo.refresh_hologram()


class SLM(QDialog):
    "Hologram displayed in the SLM window."

    refreshHologramSignal = pyqtSignal()

    def __init__(self, pars={}):
        super().__init__(parent=None,
                         flags=Qt.FramelessWindowHint
                         | Qt.WindowStaysOnTopHint)

        self.setWindowTitle('SLM Hologram')

        self.hologram_geometry = [0, 0, 400, 200]
        self.pupils = []
        self.grating_coeffs = [0.0, 0.0]
        self.grating = None

        self.log = logging.getLogger(self.__class__.__name__)
        self.flat_file = None
        self.flat = None
        self.flat_on = 0

        self.arr = None
        self.qim = None
        self.wrap_value = 0xff

        if pars:
            self.dict2parameters(deepcopy(pars))

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
            'grating_coeffs': self.grating_coeffs,
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

        if 'grating_coeffs' in d:
            self.grating_coeffs[0] = d['grating_coeffs'][0]
            self.grating_coeffs[1] = d['grating_coeffs'][1]
        self.grating = None

        self.pupils.clear()
        for ps in d['pupils']:
            self.pupils.append(Pupil(self, ps))

    def load_parameters(self, d):
        self.dict2parameters(d)
        self.refresh_hologram()

    def save_parameters(self):
        return deepcopy(self.parameters2dict())

    def refresh_hologram(self):
        self.log.info('refresh_hologram START')

        # [0, 1]
        if self.flat_file is None:
            self.flat = np.zeros(shape=(self.hologram_geometry[3],
                                        self.hologram_geometry[2]))
        else:
            self.copy_flat_shape()

        self.setGeometry(*self.hologram_geometry)
        self.setFixedSize(self.hologram_geometry[2], self.hologram_geometry[3])

        if (self.arr is None or self.qim is None
                or self.arr.shape[0] != self.hologram_geometry[3]
                or self.arr.shape[1] != self.hologram_geometry[2]):
            self.log.info('refresh_hologram ALLOCATING arr & qim')
            self.arr = np.zeros(shape=(self.hologram_geometry[3],
                                       self.hologram_geometry[2]),
                                dtype=np.uint32)
            self.qim = QImage(self.arr.data, self.arr.shape[1],
                              self.arr.shape[0], QImage.Format_RGB32)

        phase = 0
        masks = np.zeros(
            (self.hologram_geometry[3], self.hologram_geometry[2]),
            dtype=np.bool)
        for p in self.pupils:
            phase += p.refresh_pupil()
            masks = np.logical_or(masks, np.logical_not(p.mask))
            assert (p.mask.shape == (self.hologram_geometry[3],
                                     self.hologram_geometry[2]))
            assert (p.mask.dtype == np.bool)
        masks = np.logical_not(masks)

        def printout(t, x):
            if isinstance(x, np.ndarray):
                self.log.info(f'refresh_hologram {t} ' +
                              f'[{x.min():g}, {x.max():g}] {x.mean():g}')
            else:
                self.log.info(f'refresh_hologram {t} ' + str(t) +
                              ' [0.0, 0.0] 0.0')

        if self.grating is None:
            self.make_grating()

        # [0, 1] waves
        background = self.flat_on * self.flat
        # add background grating in waves
        background[masks] += self.grating[masks] / (2 * np.pi)
        # [-pi, pi] principal branch rads (zero mean) -> waves
        phase /= (2 * np.pi)  # phase in waves
        # all in waves
        gray = background + phase
        printout('gray', gray)
        gray -= np.floor(gray.min())
        assert (gray.min() >= -1e-9)
        gray *= self.wrap_value
        printout('gray', gray)
        gray %= self.wrap_value
        printout('gray', gray)
        assert (gray.min() >= 0)
        assert (gray.max() <= 255)
        gray = np.flipud(gray.astype(np.uint8))
        self.gray = gray
        self.arr[:] = gray.astype(np.uint32) * 0x010101

        self.log.info(f'refresh_hologram END {str(time())}')
        self.refreshHologramSignal.emit()

    def copy_flat_shape(self):
        self.hologram_geometry[2] = self.flat.shape[1]
        self.hologram_geometry[3] = self.flat.shape[0]
        self.grating = None

    def set_flat(self, fname, refresh_hologram=True):
        if fname is None or fname == '':
            self.flat_file = None
            self.flat = 0.0
        else:
            try:
                self.flat_file = fname
                self.flat = np.flipud(
                    np.ascontiguousarray(plt.imread(fname), dtype=np.float) /
                    255)
                self.copy_flat_shape()
            except Exception:
                self.flat_file = None
                self.flat = 0.0
        self.grating = None
        if refresh_hologram:
            self.refresh_hologram()

    def set_hologram_geometry(self, geometry, refresh=True):
        if (self.flat_file is not None and isinstance(self.flat, np.ndarray)
                and len(self.flat.shape) == 2):
            self.hologram_geometry[:2] = geometry[:2]
            self.copy_flat_shape()
        elif geometry is not None:
            self.hologram_geometry = geometry
        self.grating = None

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

    def make_grating(self):
        Ny = self.hologram_geometry[3]
        Nx = self.hologram_geometry[2]
        coeffs = self.grating_coeffs

        if np.nonzero(coeffs)[0].size == 0:
            grating = np.zeros((Ny, Nx))
        else:
            dx = np.arange(0, Nx) / Nx
            dy = np.arange(0, Ny) / Ny
            xx, yy = np.meshgrid(dx, dy)
            grating = 2 * np.pi * (coeffs[0] * xx + coeffs[1] * yy)

        self.grating = grating

    def set_grating(self, val, ind):
        self.grating_coeffs[ind] = val
        self.grating = None
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

    def __str__(self):
        return (f'<slmtools.gui.{self.__class__.__name__} ' +
                f'pupils={str(len(self.pupils))}>')


class PhaseDisplay(QFrame):
    def __init__(self, n, pupil, parent=None):
        super().__init__(parent)

        self.siz1 = 40
        self.dirty = False
        self.rzern = None
        self.pupil = pupil

        self.setMaximumWidth(200)
        self.setMaximumHeight(200)

        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout = QGridLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.update_phase(n, None)
        self.im = self.ax.imshow(self.phi, origin='lower')
        self.ax.axis('off')
        self.cb = self.ax.figure.colorbar(self.im,
                                          ax=self.ax,
                                          orientation='horizontal')
        self.cb.locator = ticker.MaxNLocator(nbins=3)
        self.cb.update_ticks()

    def update_transforms(self):
        self.R = transform_pupil(self.rzern, self.pupil.rotate,
                                 self.pupil.flipx, self.pupil.flipy)
        self.dirty = 1

    def update_phase(self, n, z, redraw=False):
        if self.rzern is None or self.rzern.n != n:
            rzern = RZern(n)
            dd = np.linspace(-1, 1, self.siz1)
            xv, yv = np.meshgrid(dd, dd)
            rzern.make_cart_grid(xv, yv)
            self.rzern = rzern
            self.update_transforms()
        if z is None:
            z = np.zeros(self.rzern.nk)
        z = np.dot(self.R, z)
        self.phi = self.rzern.eval_grid(z).reshape((self.siz1, self.siz1),
                                                   order='F')
        inner = self.phi[np.isfinite(self.phi)]
        self.min1 = inner.min()
        self.max1 = inner.max()
        self.dirty = True
        if redraw:
            self.redraw()

    def redraw(self):
        if self.phi is None:
            return
        if self.dirty:
            self.im.set_data(self.phi)
            self.im.set_clim(self.min1, self.max1)
            self.canvas.draw()


class MatplotlibWindow(QFrame):
    def __init__(self,
                 slm,
                 slmwindow,
                 parent=None,
                 toolbar=True,
                 figsize=None):
        super().__init__(parent)

        self.shape = None
        self.im = None

        self.circ_ind = None
        self.circ = None
        self.circ_rho = None
        self.circ_xy = None
        self.circ_geometry = None

        # a figure instance to plot on
        if figsize is None:
            self.figure = Figure()
        else:
            self.figure = Figure(figsize)
        self.ax = self.figure.add_subplot(111)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

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
        self.slmwindow = slmwindow
        self.figure.subplots_adjust(left=0,
                                    bottom=0,
                                    right=1,
                                    top=1,
                                    wspace=None,
                                    hspace=None)

    def check_circ(self):
        if self.circ is None:
            return 1
        elif self.circ_ind != self.slmwindow.pupilsTab.currentIndex():
            return 1
        elif not np.allclose(self.circ_geometry, self.slm.hologram_geometry):
            return 1
        else:
            p = self.slm.pupils[self.circ_ind]
            if (p.rho != self.circ_rho or not np.allclose(p.xy, self.circ_xy)):
                return 1
            else:
                return 0

    def refresh_circle(self, index, draw=True):
        # ignore index for mpl call back
        if self.check_circ() and len(self.slm.pupils) > 1:
            self.circ_ind = self.slmwindow.pupilsTab.currentIndex()
            if self.circ:
                self.circ[0].remove()
                self.circ = None
            p = self.slm.pupils[self.circ_ind]
            ll = np.linspace(0, 2 * np.pi, 50)
            self.circ = self.ax.plot(p.rho * np.cos(ll) + p.xy[0] +
                                     self.slm.hologram_geometry[2] / 2,
                                     p.rho * np.sin(ll) - p.xy[1] +
                                     self.slm.hologram_geometry[3] / 2,
                                     color='r')
            self.circ_rho = p.rho
            self.circ_xy = np.array(p.xy, copy=1)
            self.circ_geometry = np.array(self.slm.hologram_geometry, copy=1)
            if draw:
                self.canvas.draw()

    def update_array(self):
        if self.slm.gray is None:
            return

        if self.im is None or self.slm.gray.shape != self.shape:
            if self.im:
                self.ax.clear()
                self.circ = None
            self.im = self.ax.imshow(self.slm.gray,
                                     cmap="gray",
                                     vmin=0,
                                     vmax=0xff)
            self.ax.axis("off")
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
            self.shape = self.slm.gray.shape
        if self.circ and len(self.slm.pupils) == 1:
            self.circ[0].remove()
            self.circ = None
        elif self.check_circ() and len(self.slm.pupils) != 1:
            self.refresh_circle(0, draw=False)

        self.im.set_data(self.slm.gray)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.canvas.draw()


class RelSlider:
    def __init__(self, val, cb):
        self.old_val = None
        self.fto100mul = 100
        self.cb = cb

        self.sba = QDoubleSpinBox()
        self.sba.setMinimum(-1000)
        self.sba.setMaximum(1000)
        self.sba.setDecimals(6)
        self.sba.setToolTip('Effective value')
        self.sba.setValue(val)
        self.sba_color(val)
        self.sba.setSingleStep(1.25e-3)

        self.qsr = QSlider(Qt.Horizontal)
        self.qsr.setMinimum(-100)
        self.qsr.setMaximum(100)
        self.qsr.setValue(0)
        self.qsr.setToolTip('Drag to apply relative delta')

        self.sbm = QDoubleSpinBox()
        self.sbm.setMinimum(0.01)
        self.sbm.setMaximum(1000)
        self.sbm.setSingleStep(1.25e-3)
        self.sbm.setToolTip('Maximum relative delta')
        self.sbm.setDecimals(2)
        self.sbm.setValue(4.0)

        def sba_cb():
            def f():
                self.block()
                val = self.sba.value()
                self.sba_color(val)
                self.cb(val)
                self.unblock()

            return f

        def qs1_cb():
            def f(t):
                self.block()

                if self.old_val is None:
                    self.qsr.setValue(0)
                    self.unblock()
                    return

                val = self.old_val + self.qsr.value() / 100 * self.sbm.value()
                self.sba.setValue(val)
                self.sba_color(val)
                self.cb(val)

                self.unblock()

            return f

        def qs1_end():
            def f():
                self.block()
                self.qsr.setValue(0)
                self.old_val = None
                self.unblock()

            return f

        def qs1_start():
            def f():
                self.block()
                self.old_val = self.get_value()
                self.unblock()

            return f

        self.sba_cb = sba_cb()
        self.qs1_cb = qs1_cb()
        self.qs1_start = qs1_start()
        self.qs1_end = qs1_end()

        self.sba.valueChanged.connect(self.sba_cb)
        self.qsr.valueChanged.connect(self.qs1_cb)
        self.qsr.sliderPressed.connect(self.qs1_start)
        self.qsr.sliderReleased.connect(self.qs1_end)

    def set_tooltip(self, t):
        self.sba.setToolTip(t)
        self.qsr.setToolTip(t)
        self.sbm.setToolTip(t)

    def sba_color(self, val):
        if abs(val) > 1e-4:
            self.sba.setStyleSheet("font-weight: bold;")
        else:
            self.sba.setStyleSheet("font-weight: normal;")
        # self.sba.update()

    def block(self):
        self.sba.blockSignals(True)
        self.qsr.blockSignals(True)
        self.sbm.blockSignals(True)

    def unblock(self):
        self.sba.blockSignals(False)
        self.qsr.blockSignals(False)
        self.sbm.blockSignals(False)

    def enable(self):
        self.sba.setEnabled(True)
        self.qsr.setEnabled(True)
        self.sbm.setEnabled(True)

    def disable(self):
        self.sba.setEnabled(False)
        self.qsr.setEnabled(False)
        self.sbm.setEnabled(False)

    def fto100(self, f):
        return int((f + self.m2) / (2 * self.m2) * self.fto100mul)

    def get_value(self):
        return self.sba.value()

    def set_value(self, v):
        self.sba_color(v)
        return self.sba.setValue(v)

    def add_to_layout(self, l1, ind1, ind2):
        l1.addWidget(self.sba, ind1, ind2)
        l1.addWidget(self.qsr, ind1, ind2 + 1)
        l1.addWidget(self.sbm, ind1, ind2 + 2)

    def remove_from_layout(self, l1):
        l1.removeWidget(self.sba)
        l1.removeWidget(self.qsr)
        l1.removeWidget(self.sbm)

        self.sba.setParent(None)
        self.qsr.setParent(None)
        self.sbm.setParent(None)

        self.sba.valueChanged.disconnect(self.sba_cb)
        self.qsr.valueChanged.disconnect(self.qs1_cb)
        self.qsr.sliderPressed.disconnect(self.qs1_start)
        self.qsr.sliderReleased.disconnect(self.qs1_end)

        self.sba_cb = None
        self.qs1_cb = None
        self.qs1_start = None
        self.qs1_end = None

        self.sb = None
        self.qsr = None


class PlotCoeffs(QDialog):
    def set_data(self, z):
        self.setWindowTitle('Zernike coefficients')
        frame = QFrame()
        fig = FigureCanvas(Figure(figsize=(7, 5)))
        layout = QGridLayout()
        frame.setLayout(layout)
        nav = NavigationToolbar2QT(fig, frame)
        layout.addWidget(nav, 0, 0)
        layout.addWidget(fig, 1, 0)
        fig.figure.subplots_adjust(left=.125,
                                   right=.9,
                                   bottom=.1,
                                   top=.9,
                                   wspace=0.45,
                                   hspace=0.45)
        self.fig = fig

        ax1 = fig.figure.add_subplot(1, 1, 1)
        ax1.plot(range(1, z.size + 1), z, marker='.')
        ax1.grid()
        ax1.set_xlabel('Noll')
        ax1.set_ylabel('[rad]')

        self.ax1 = ax1

        l1 = QGridLayout()
        self.setLayout(l1)
        l1.addWidget(frame)


class PupilPanel(QFrame):
    def __init__(self, pupil, ptabs, parent=None):
        """Subclass for a control GUI.
        Parameters:
            slm: SLM instance
            pars: dict, saved pars
            is_parent: bool. Useful in the case of doublepass to determine
                for instance which widget determines the overall geometry"""
        super().__init__(parent)
        self.parent = parent
        self.refresh_gui = {}
        self.pupil = pupil
        self.ptabs = ptabs

        self.make_pupil_tab()
        self.make_2d_tab()
        self.make_3d_tab()
        self.make_phase_tab()
        self.make_grating_tab()
        self.make_grid_tab()
        self.make_aberration_tab()

        upl = QGridLayout()
        upl.addWidget(self.group_phase, 0, 0, 3, 1)
        upl.addWidget(self.group_pupil, 0, 1)
        upl.addWidget(self.group_grating, 1, 1)
        upl.addWidget(self.group_2d, 2, 1)
        upl.addWidget(self.group_grid, 3, 0)
        upl.addWidget(self.group_3d, 3, 1)
        up = QFrame()
        up.setLayout(upl)

        top = QSplitter(Qt.Vertical)
        top.addWidget(up)
        top.addWidget(self.group_aberration)

        myl = QGridLayout()
        myl.addWidget(top)
        self.setLayout(myl)

        self.pindex = self.ptabs.count()
        self.ptabs.addTab(self, self.pupil.name)

    def helper_boolupdate(self, mycallback):
        def f(i):
            mycallback(i)

        return f

    def helper_boolupdate_transform(self, mycallback):
        def f(i):
            mycallback(i)
            self.phase_display.update_transforms()
            self.phase_display.update_phase(self.pupil.rzern.n,
                                            self.pupil.aberration,
                                            redraw=True)

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

        def help1(txt, mini, handler, curval, i, j=0):
            l1.addWidget(QLabel(txt), j, 2 * i)
            le = QLineEdit(str(curval))
            le.editingFinished.connect(handler(i, le))
            le.setMaximumWidth(50)
            val = MyQDoubleValidator()
            val.setFixup(curval)
            le.setValidator(val)
            if mini:
                val.setBottom(mini)
            l1.addWidget(le, j, 2 * i + 1)
            return le

        lex = help1('x0', None, handle_xy, self.pupil.xy[0], 0)
        ley = help1('y0', None, handle_xy, self.pupil.xy[1], 1)
        lerho = help1('radius', 10, handle_rho, self.pupil.rho, 2)

        cbenabled = QCheckBox('on')
        cbenabled.setChecked(self.pupil.enabled)
        cbenabled.toggled.connect(
            self.helper_boolupdate(self.pupil.set_enabled))

        lename = QLineEdit(self.pupil.name)
        l1.addWidget(cbenabled, 1, 0, 1, 2)
        l1.addWidget(lename, 1, 2, 1, 4)

        def fname():
            def f():
                self.pupil.name = lename.text()
                self.ptabs.setTabText(self.pindex, self.pupil.name)

            return f

        lename.editingFinished.connect(fname())

        def handle_rotate(i, le):
            def f():
                try:
                    fval = float(le.text())
                except Exception:
                    le.setText(str(self.pupil.rotate))
                    return
                self.pupil.set_rotate(fval)
                self.phase_display.update_transforms()
                self.phase_display.update_phase(self.pupil.rzern.n,
                                                self.pupil.aberration,
                                                redraw=True)
                le.setText(str(self.pupil.rotate))

            return f

        cbx = QCheckBox('flipx')
        cbx.setChecked(self.pupil.flipx)
        cbx.toggled.connect(
            self.helper_boolupdate_transform(self.pupil.set_flipx))
        cby = QCheckBox('flipy')
        cby.setChecked(self.pupil.flipy)
        cby.toggled.connect(
            self.helper_boolupdate_transform(self.pupil.set_flipy))
        l1.addWidget(cbx, 2, 0)
        l1.addWidget(cby, 2, 1)
        lerotate = help1('rotate', None, handle_rotate, self.pupil.rotate, 2,
                         2)

        def make_f():
            ctls = (lex, ley, lerho, cbx, cby, lerotate, cbenabled)

            def t1():
                return self.pupil.xy, self.pupil.rho

            def f():
                xy, rho = t1()
                for p in ctls:
                    p.blockSignals(True)
                lex.setText(str(xy[0]))
                ley.setText(str(xy[1]))
                lerho.setText(str(rho))
                cbx.setChecked(self.pupil.flipx)
                cby.setChecked(self.pupil.flipy)
                lerotate.setText(str(self.pupil.rotate))
                cbenabled.setChecked(self.pupil.enabled)
                for p in ctls:
                    p.blockSignals(False)

            return f

        self.refresh_gui['pupil'] = make_f()

        group.setLayout(l1)
        self.group_pupil = group

    def make_2d_tab(self):
        g = QGroupBox('2D STED')
        l1 = QGridLayout()
        c = QCheckBox('on')
        c.setChecked(self.pupil.mask2d_on)
        c.toggled.connect(self.helper_boolupdate(self.pupil.set_mask2d_on))
        l1.addWidget(c, 0, 0, 1, 3)

        def f():
            def f(r):
                self.pupil.set_mask2d_sign(r)

            return f

        lab1 = QLabel('m')
        l1.addWidget(lab1, 1, 0)
        s = RelSlider(self.pupil.mask2d_sign, f())
        s.add_to_layout(l1, 1, 1)
        lab1.setToolTip('Momentum')
        s.set_tooltip('Momentum')

        def f2():
            def f(r):
                self.pupil.set_mask2d_mul(r)

            return f

        lab2 = QLabel('α')
        l1.addWidget(lab2, 2, 0)
        s2 = RelSlider(self.pupil.mask2d_mul, f2())
        s2.add_to_layout(l1, 2, 1)
        tt2 = 'θ<sup>α</sup>'
        lab2.setToolTip(tt2)
        s2.set_tooltip(tt2)

        g.setLayout(l1)
        self.group_2d = g

        def f():
            def f():
                c.setChecked(self.pupil.mask2d_on)
                s.block()
                s.set_value(self.pupil.mask2d_sign)
                s2.set_value(self.pupil.mask2d_mul)
                s.unblock()

            return f

        self.refresh_gui['2d'] = f()

    def make_3d_tab(self):
        g = QGroupBox('3D STED')

        def update_radius(slider, what):
            def f(r):
                slider.setValue(int(r * 100))
                what(r)

            return f

        def update_spinbox(s):
            def f(t):
                s.setValue(t / 100)

            return f

        l1 = QGridLayout()
        c = QCheckBox('on')
        c.setChecked(self.pupil.mask3d_on)
        c.toggled.connect(self.helper_boolupdate(self.pupil.set_mask3d_on))
        l1.addWidget(c, 0, 0)
        slider1 = QSlider(Qt.Horizontal)
        slider1.setMinimum(0)
        slider1.setMaximum(100)
        slider1.setFocusPolicy(Qt.StrongFocus)
        slider1.setTickPosition(QSlider.TicksBothSides)
        slider1.setTickInterval(20)
        slider1.setSingleStep(1)
        slider1.setValue(int(100 * self.pupil.mask3d_radius))
        spinbox1 = QDoubleSpinBox()
        spinbox1.setRange(0.0, 1.0)
        spinbox1.setSingleStep(0.01)
        spinbox1.setValue(self.pupil.mask3d_radius)
        slider1.valueChanged.connect(update_spinbox(spinbox1))

        spinbox1.valueChanged.connect(
            update_radius(slider1, self.pupil.set_mask3d_radius))
        l1.addWidget(QLabel('radius'), 0, 1)
        l1.addWidget(spinbox1, 0, 2)
        l1.addWidget(slider1, 0, 3)

        slider2 = QSlider(Qt.Horizontal)
        slider2.setMinimum(0)
        slider2.setMaximum(200)
        slider2.setFocusPolicy(Qt.StrongFocus)
        slider2.setTickPosition(QSlider.TicksBothSides)
        slider2.setTickInterval(40)
        slider2.setSingleStep(1)
        slider2.setValue(int(100 * self.pupil.mask3d_height))
        spinbox2 = QDoubleSpinBox()
        spinbox2.setRange(0.0, 2.0)
        spinbox2.setSingleStep(0.01)
        spinbox2.setValue(self.pupil.mask3d_height)
        slider2.valueChanged.connect(update_spinbox(spinbox2))

        spinbox2.valueChanged.connect(
            update_radius(slider2, self.pupil.set_mask3d_height))
        l1.addWidget(QLabel('height'), 1, 1)
        l1.addWidget(spinbox2, 1, 2)
        l1.addWidget(slider2, 1, 3)
        g.setLayout(l1)

        self.group_3d = g

        def f():
            ctls = (slider1, slider2, spinbox1, spinbox2)

            def f():
                c.setChecked(self.pupil.mask3d_on)
                for p in ctls:
                    p.blockSignals(True)
                spinbox1.setValue(self.pupil.mask3d_radius)
                slider1.setValue(int(100 * self.pupil.mask3d_radius))
                spinbox2.setValue(self.pupil.mask3d_height)
                slider2.setValue(int(100 * self.pupil.mask3d_height))
                for p in ctls:
                    p.blockSignals(False)

            return f

        self.refresh_gui['3d'] = f()

    def make_grid_tab(self):
        g = QGroupBox('Grid')
        l1 = QGridLayout()
        c = QCheckBox('on')
        c.setChecked(self.pupil.align_grid_on)
        c.toggled.connect(self.helper_boolupdate(self.pupil.set_align_grid_on))
        l1.addWidget(c, 0, 0)

        le = QLineEdit(str(self.pupil.align_grid_pitch))
        val = MyQIntValidator()
        val.setBottom(1)
        val.setTop(100)
        val.setFixup(1)
        le.setValidator(val)

        def f():
            def f():
                try:
                    ival = int(le.text())
                    assert (ival > 0)
                    self.pupil.set_align_grid_pitch(ival)
                except Exception:
                    le.setText(str(self.pupil.align_grid_pitch))
                    return

            return f

        le.editingFinished.connect(f())
        l1.addWidget(le, 0, 1)
        g.setLayout(l1)

        self.group_grid = g

        def f():
            def f():
                c.setChecked(self.pupil.align_grid_on)
                le.setText(str(self.pupil.align_grid_pitch))

            return f

        self.refresh_gui['grid'] = f()

    def make_phase_tab(self):
        g = QGroupBox('Phase')
        phase_display = PhaseDisplay(self.pupil.rzern.n, self.pupil)
        l1 = QGridLayout()
        l1.addWidget(phase_display, 0, 0)
        g.setLayout(l1)

        self.group_phase = g
        self.phase_display = phase_display

    def make_aberration_tab(self):
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
        bplot = QPushButton('plot')
        toplay.addWidget(labzm, 0, 0)
        toplay.addWidget(lezm, 0, 1)
        toplay.addWidget(bplot, 0, 3)
        toplay.addWidget(reset, 0, 4)

        def plotf():
            def f():
                self.parent.mutex.lock()
                p = PlotCoeffs()
                p.set_data(self.pupil.aberration.ravel())
                p.exec_()
                self.parent.mutex.unlock()

            return f

        bplot.clicked.connect(plotf())

        scroll = QScrollArea()
        toplay.addWidget(scroll, 1, 0, 1, 5)
        scroll.setWidget(QWidget())
        scrollLayout = QGridLayout(scroll.widget())
        scroll.setWidgetResizable(True)
        self.zernike_rows = []

        def make_hand_slider(ind):
            def f(r):
                self.pupil.aberration[ind, 0] = r
                self.pupil.set_aberration(self.pupil.aberration)

                self.phase_display.update_phase(self.pupil.rzern.n,
                                                self.pupil.aberration,
                                                redraw=True)

            return f

        def make_hand_lab(le, i):
            def f():
                self.pupil.zernike_labels[str(i)] = le.text()

            return f

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

        def make_update_zernike_rows():
            def f(mynk=None):
                if mynk is None:
                    mynk = self.pupil.rzern.nk
                ntab = self.pupil.rzern.ntab
                mtab = self.pupil.rzern.mtab

                if len(self.zernike_rows) < mynk:
                    for i in range(len(self.zernike_rows), mynk):
                        lab = QLabel(
                            f'Z<sub>{i + 1}</sub> ' +
                            f'Z<sub>{ntab[i]}</sub><sup>{mtab[i]}</sup>')
                        slider = RelSlider(self.pupil.aberration[i, 0],
                                           make_hand_slider(i))

                        try:
                            zname = self.pupil.zernike_labels[str(i)]
                        except Exception:
                            zname = default_zernike_name(
                                i + 1, ntab[i], mtab[i])
                            self.pupil.zernike_labels[str(i)] = zname

                        lbn = QLineEdit(zname)
                        lbn.setMaximumWidth(120)
                        hand_lab = make_hand_lab(lbn, i)
                        lbn.editingFinished.connect(hand_lab)

                        scrollLayout.addWidget(lab, i, 0)
                        scrollLayout.addWidget(lbn, i, 1)
                        slider.add_to_layout(scrollLayout, i, 2)

                        self.zernike_rows.append((lab, slider, lbn, hand_lab))

                    assert (len(self.zernike_rows) == mynk)

                elif len(self.zernike_rows) > mynk:
                    for i in range(len(self.zernike_rows) - 1, mynk - 1, -1):
                        lab, slider, lbn, hand_lab = self.zernike_rows.pop()

                        scrollLayout.removeWidget(lab)
                        slider.remove_from_layout(scrollLayout)
                        scrollLayout.removeWidget(lbn)

                        lbn.editingFinished.disconnect(hand_lab)
                        lab.setParent(None)
                        lbn.setParent(None)

                    assert (len(self.zernike_rows) == mynk)

            return f

        self.update_zernike_rows = make_update_zernike_rows()

        def reset_fun():
            self.pupil.aberration[:] = 0
            self.pupil.set_aberration(self.pupil.aberration)
            self.phase_display.update_phase(self.pupil.rzern.n,
                                            self.pupil.aberration,
                                            redraw=True)
            self.update_zernike_rows(0)
            self.update_zernike_rows()

        def change_radial():
            try:
                ival = int(lezm.text())
                assert (ival > 0)
            except Exception:
                lezm.setText(str(len(self.zernike_rows)))
                return

            n = (ival + 1) * (ival + 2) // 2
            newab = np.zeros((n, 1))
            minn = min((n, self.pupil.rzern.nk))
            newab[:minn, 0] = self.pupil.aberration[:minn, 0]
            self.pupil.set_aberration(newab)

            self.update_zernike_rows()
            self.phase_display.update_phase(self.pupil.rzern.n,
                                            self.pupil.aberration,
                                            redraw=True)
            lezm.setText(str(self.pupil.rzern.n))

        self.phase_display.update_phase(self.pupil.rzern.n,
                                        self.pupil.aberration,
                                        redraw=True)

        reset.clicked.connect(reset_fun)
        lezm.editingFinished.connect(change_radial)

        self.group_aberration = top

        def f():
            def f():
                self.update_zernike_rows(0)
                self.update_zernike_rows()
                self.phase_display.update_phase(self.pupil.rzern.n,
                                                self.pupil.aberration,
                                                redraw=True)

            return f

        self.refresh_gui['aberration'] = f()
        self.update_zernike_rows(0)
        self.update_zernike_rows()

    def make_grating_tab(self):
        """Position tab is meant to help positionning the phase mask
        without using tip and tilt"""
        pos = QGroupBox('Blazed grating')
        poslay = QGridLayout()
        pos.setLayout(poslay)

        def make_cb(ind):
            def f(r):
                self.pupil.set_anglexy(r, ind)

            return f

        slider_x = RelSlider(self.pupil.angle_xy[0], make_cb(0))
        poslay.addWidget(QLabel('x'), 0, 0)
        slider_x.add_to_layout(poslay, 0, 1)

        slider_y = RelSlider(self.pupil.angle_xy[1], make_cb(1))
        poslay.addWidget(QLabel('y'), 1, 0)
        slider_y.add_to_layout(poslay, 1, 1)

        self.group_grating = pos

        def f():
            def f():
                for i, s in enumerate((slider_x, slider_y)):
                    s.block()
                    s.set_value(self.pupil.angle_xy[i])
                    s.unblock()

            return f

        self.refresh_gui['grating'] = f()

    def keyPressEvent(self, event):
        pass


def get_noll_indices(params):
    noll_min = params['min']
    noll_max = params['max']
    minclude = np.array(params['include'], dtype=np.int)
    mexclude = np.array(params['exclude'], dtype=np.int)

    mrange = np.arange(noll_min, noll_max + 1, dtype=np.int)
    zernike_indices1 = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))
    zernike_indices = []
    for k in minclude:
        if k in zernike_indices1 and k not in zernike_indices:
            zernike_indices.append(k)
    remaining = np.setdiff1d(zernike_indices1, np.unique(zernike_indices))
    for k in remaining:
        zernike_indices.append(k)
    assert (len(zernike_indices) == zernike_indices1.size)
    zernike_indices = np.array(zernike_indices, dtype=np.int)

    return zernike_indices


h5_prefix = 'slmtools/'


class Zernike2Control:
    @staticmethod
    def get_default_parameters():
        return {
            'include': [],
            'exclude': [1, 2, 3, 4],
            'min': 5,
            'max': 6,
        }

    @staticmethod
    def get_parameters_info():
        return {
            'include': (list, int, 'Zernike indices to include'),
            'exclude': (list, int, 'Zernike indices to include'),
            'min': (int, (1, None), 'Minimum Zernike index'),
            'max': (int, (1, None), 'Maximum Zernike index'),
        },

    def __init__(self):
        raise NotImplementedError()


class Zernike1Control:
    @staticmethod
    def get_default_parameters():
        return {
            'include': [],
            'exclude': [1, 2, 3, 4],
            'min': 5,
            'max': 6,
            'pupil_index': 0,
        }

    @staticmethod
    def get_parameters_info():
        return {
            'include': (list, int, 'Zernike indices to include', 1),
            'exclude': (list, int, 'Zernike indices to include', 1),
            'min': (int, (1, None), 'Minimum Zernike index', 1),
            'max': (int, (1, None), 'Maximum Zernike index', 1),
            'pupil_index': (int, (0, None), 'SLM pupil number', 0),
        }

    def __str__(self):
        return (f'<slmtools.gui.{self.__class__.__name__} ' +
                f'pupil={str(self.pars["pupil_index"])} ' +
                f'ndof={self.ndof} indices={self.indices}>')

    def __init__(self, slm, pars={}, h5f=None):
        self.log = logging.getLogger(self.__class__.__name__)
        dpars = self.get_default_parameters()
        pars = {**deepcopy(dpars), **deepcopy(pars)}
        self.pars = pars
        self.slm = slm
        self.pupil = slm.pupils[pars['pupil_index']]

        try:
            enabled = self.pars['enabled']
        except KeyError:
            enabled = 1

        if enabled:
            self.indices = get_noll_indices(pars)
        else:
            self.indices = np.array([], dtype=np.int)
        self.ndof = self.indices.size

        z0 = self.pupil.aberration.flatten()
        self.log.info(f'indices {self.indices} ndof {self.ndof}')
        if (self.ndof > 0
                and (self.indices - 1).max() >= self.pupil.aberration.size):
            z1 = z0
            z0 = np.zeros(self.indices.max())
            z0[:z1.size] = z1
            self.pupil.set_aberration(z1.reshape(-1, 1))
        self.z0 = z0

        self.h5f = h5f

        self.h5_save('slmtools', json.dumps(self.slm.save_parameters()))
        self.h5_save('indices', self.indices)
        self.h5_save('flat', self.slm.flat)
        self.h5_save('z0', self.z0)
        self.P = None

        # handle orthogonal pupil transform
        try:
            self.transform_pupil(self.pupil.rotate, self.pupil.flipx,
                                 self.pupil.flipy)
        except Exception:
            self.P = None

        self.ab = np.zeros((self.ndof, ))

        self.h5_make_empty('x', (self.ndof, ))
        self.h5_make_empty('z2', (self.z0.size, ))
        self.h5_save('name', self.__class__.__name__)
        self.h5_save('ab', self.ab)
        self.h5_save('P', np.eye(self.z0.size))
        self.h5_save('params', json.dumps(pars))

    def save_parameters(self, merge={}):
        d = {**merge, **self.pars}
        d['slm'] = self.slm.save_parameters()
        return d

    def h5_make_empty(self, name, shape, dtype=np.float):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + name
            if name in self.h5f:
                del self.h5f[name]
            self.h5f.create_dataset(name,
                                    shape + (0, ),
                                    maxshape=shape + (None, ),
                                    dtype=dtype)

    def h5_append(self, name, what):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + name
            self.h5f[name].resize(
                (self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + where
            if name in self.h5f:
                del self.h5f[name]
            self.h5f[name] = what

    def write(self, x):
        "Write Zernike coefficients"
        assert (x.size == self.ndof)

        z1 = np.zeros(self.pupil.aberration.size)

        # z controlled Zernike degrees of freedom
        z1[self.indices - 1] = x[:]

        # z1 transform all Zernike coefficients
        if self.P is not None:
            z1 = np.dot(self.P, z1)

        # add initial state
        z2 = self.z0 + z1

        # logging
        self.h5_append('x', x)
        self.h5_append('z2', z2)

        # write pupil
        self.pupil.set_aberration(z2.reshape(-1, 1))

    def set_random_ab(self, rms=1.0):
        raise NotImplementedError()

    def transform_pupil(self, alpha=0., flipx=False, flipy=False):
        rzern = self.calib.get_rzern()
        tot = transform_pupil(rzern, alpha, flipx, flipy)

        if tot.size == 1:
            return
        else:
            self.set_P(tot)

    def set_P(self, P):
        addr = h5_prefix + self.__class__.__name__ + '/P'
        if P is None:
            self.P = None

            if self.h5f:
                del self.h5f[addr]
                self.h5f[addr][:] = np.eye(self.nz)
        else:
            assert (P.ndim == 2)
            assert (P.shape[0] == P.shape[1])
            assert (np.allclose(np.dot(P, P.T), np.eye(P.shape[0])))
            if self.P is None:
                self.P = P.copy()
            else:
                np.dot(P, self.P.copy(), self.P)

            if self.h5f:
                del self.h5f[addr]
                self.h5f[addr][:] = self.P[:]


class PupilPositionControl:
    @staticmethod
    def get_default_parameters():
        return {
            'gain_x0': 1.0,
            'gain_y0': 1.0,
            'gain_rho': 1.0,
            'pupil_index': 0,
        }

    @staticmethod
    def get_parameters_info():
        return {
            'gain_x0': (float, (None, None),
                        'scale x0 degree of freedom (0 to disable)', 1),
            'gain_y0': (float, (None, None),
                        'scale x0 degree of freedom (0 to disable)', 1),
            'gain_rho': (float, (None, None),
                         'scale x0 degree of freedom (0 to disable)', 1),
            'pupil_index': (int, (0, None), 'SLM pupil number', 0),
        }

    def __str__(self):
        return (f'<slmtools.gui.{self.__class__.__name__} ' +
                f'pupil={str(self.pars["pupil_index"])} ' +
                f'ndof={self.ndof}>')

    def __init__(self, slm, pars={}, h5f=None):
        self.log = logging.getLogger(self.__class__.__name__)
        pars = {**deepcopy(self.get_default_parameters()), **deepcopy(pars)}
        self.pars = pars
        self.slm = slm
        self.pupil = slm.pupils[pars['pupil_index']]
        self.pos0 = np.array(
            [self.pupil.xy[0], self.pupil.xy[1], self.pupil.rho])

        self.gains = np.array([
            self.pars['gain_x0'], self.pars['gain_y0'], self.pars['gain_rho']
        ])

        try:
            enabled = self.pars['enabled']
        except KeyError:
            enabled = 1
        if enabled:
            self.ndof = (self.gains != 0.).sum()
        else:
            self.ndof = 0

        self.z0 = self.pupil.aberration.flatten()

        self.h5f = h5f

        self.h5_save('slmtools', json.dumps(self.slm.save_parameters()))
        self.h5_save('flat', self.slm.flat)
        self.h5_save('z0', np.zeros(self.ndof))
        self.h5_save('pos0', self.pos0)
        self.h5_save('gains', self.gains)
        self.P = None

        self.h5_make_empty('z2', (self.ndof, ))
        self.h5_make_empty('x', (self.ndof, ))
        self.h5_save('ab', np.zeros(self.ndof))
        self.h5_save('name', self.__class__.__name__)
        self.h5_save('params', json.dumps(pars))

    def save_parameters(self, merge={}):
        d = {**merge, **self.pars}
        d['slm'] = self.slm.save_parameters()
        return d

    def h5_make_empty(self, name, shape, dtype=np.float):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + name
            if name in self.h5f:
                del self.h5f[name]
            self.h5f.create_dataset(name,
                                    shape + (0, ),
                                    maxshape=shape + (None, ),
                                    dtype=dtype)

    def h5_append(self, name, what):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + name
            self.h5f[name].resize(
                (self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + where
            if name in self.h5f:
                del self.h5f[name]
            self.h5f[name] = what

    def write(self, x):
        "Write mask alignment"
        assert (x.size == self.ndof)

        delta = self.gains.copy()
        c = 0
        for i in range(3):
            if delta[i] != 0.0:
                delta[i] *= x[c]
                c += 1

        z2 = self.pos0 + delta

        # logging
        self.h5_append('x', x)
        self.h5_append('z2', z2)

        # write pupil
        self.pupil.set_xy(z2[:2])
        self.pupil.set_rho(z2[2])

    def set_random_ab(self, rms=1.0):
        raise NotImplementedError()


class SLMControls:
    @staticmethod
    def get_default_parameters():
        return {
            'Zernike1Control': Zernike1Control.get_default_parameters(),
            'PupilPositionControl':
            PupilPositionControl.get_default_parameters(),
            # 'Zernike2Control': Zernike2Control.get_default_parameters(),
        }

    @staticmethod
    def get_parameters_info():
        return {
            'Zernike1Control': Zernike1Control.get_parameters_info(),
            'PupilPositionControl': PupilPositionControl.get_parameters_info(),
            # 'Zernike2Control': Zernike2Control.get_parameters_info(),
        }

    @staticmethod
    def get_controls():
        return {
            'Zernike1Control': Zernike1Control,
            'PupilPositionControl': PupilPositionControl,
        }

    @staticmethod
    def new_control(slm, name, pars={}, h5f=None):
        options = SLMControls.get_controls()
        if name not in options.keys():
            raise ValueError(
                f'name must be one of {", ".join(options.keys())}')

        return options[name](slm, pars, h5f)


class OptionsPanel(QFrame):
    def setup(self, pars, name, defaultd, infod):
        self.lines = []
        self.pars = pars
        self.name = name
        self.defaultd = defaultd
        self.infod = infod

        layout = QGridLayout()
        self.setLayout(layout)

        combo = QComboBox()
        for k in defaultd.keys():
            combo.addItem(k)
        layout.addWidget(combo, 0, 0)
        combo.setCurrentIndex(0)
        self.combo = combo

        scroll = QScrollArea()
        scroll.setWidget(QWidget())
        scroll.setWidgetResizable(True)
        lay = QGridLayout(scroll.widget())
        self.scroll = scroll
        self.lay = lay

        layout.addWidget(scroll, 1, 0)
        addr_options = name + '_options'
        addr_selection = name + '_name'
        self.addr_options = addr_options
        self.addr_selection = addr_selection

        self.selection = combo.currentText()
        if addr_options not in self.pars:
            self.pars[addr_options] = defaultd
        if addr_selection not in self.pars:
            self.pars[addr_selection] = self.selection

        self.from_dict(self.selection, self.infod[self.selection],
                       self.pars[self.addr_options][self.selection])

        def f():
            def f(selection):
                self.clear_all()
                self.from_dict(selection, self.infod[selection],
                               self.pars[self.addr_options][selection])
                self.selection = selection

            return f

        combo.currentTextChanged.connect(f())

    def get_options(self):
        return (self.selection,
                dict(self.pars[self.addr_options][self.selection]))

    def from_dict(self, selection, infod, valuesd):
        def get_noll():
            indices = [
                str(s) for s in get_noll_indices(self.pars[self.addr_options]
                                                 [selection]).tolist()
            ]
            return ', '.join(indices)

        count = 0
        if selection == 'Zernike1Control':
            lab = QLabel('Noll indices')
            self.lay.addWidget(lab, count, 0)
            le = QLineEdit(get_noll())
            le.setReadOnly(True)
            self.lay.addWidget(le, count, 1)
            self.lines.append(((le, lab), None))

            le_noll = le
            count = 1
        else:
            le_noll = None

        for k, v in infod.items():
            if v[-1] == 0:
                continue

            lab = QLabel(k)

            type1 = v[0]
            bounds = v[1]
            desc = v[2]

            lab.setToolTip(desc)
            self.lay.addWidget(lab, count, 0)

            def fle(k, le, val, type1):
                def f():
                    newval = type1(le.text())
                    self.pars[self.addr_options][selection][k] = type1(
                        le.text())
                    val.setFixup(newval)
                    if le_noll:
                        le_noll.setText(get_noll())

                return f

            def ledisc(w, hand):
                def f():
                    w.editingFinished.disconnect(hand)

                return f

            curval = valuesd[k]
            if type1 in (int, float):
                le = QLineEdit(str(curval))
                le.setToolTip(desc)
                if type1 == int:
                    vv = MyQIntValidator()
                else:
                    vv = MyQDoubleValidator()
                vv.setFixup(curval)
                if bounds[0] is not None:
                    vv.setBottom(bounds[0])
                if bounds[1] is not None:
                    vv.setTop(bounds[1])
                le.setValidator(vv)
                hand = fle(k, le, vv, type1)
                le.editingFinished.connect(hand)
                disc = ledisc(le, hand)
            elif type1 == list:
                le = QLineEdit(', '.join([str(c) for c in curval]))
                le.setToolTip(desc)

                def make_validator(k, le, type1, bounds):
                    def f():
                        old = self.pars[self.addr_options][selection][k]
                        try:
                            tmp = [
                                bounds(s) for s in le.text().split(',')
                                if s != ''
                            ]
                        except Exception:
                            tmp = old
                        self.pars[self.addr_options][selection][k] = tmp
                        le.blockSignals(True)
                        le.setText(', '.join([str(c) for c in tmp]))
                        le.blockSignals(False)
                        if le_noll:
                            le_noll.setText(get_noll())

                    return f

                hand = make_validator(k, le, type1, bounds)
                le.editingFinished.connect(hand)
                disc = ledisc(le, hand)
            else:
                raise RuntimeError()

            self.lay.addWidget(le, count, 1)
            self.lines.append(((le, lab), disc))
            count += 1

    def clear_all(self):
        for l1 in self.lines:
            for w in l1[0]:
                self.lay.removeWidget(w)
                w.setParent(None)
            if l1[1]:
                l1[1]()
        self.lines.clear()


class SLMWindow(QMainWindow):
    "Dialog to control the SLM hologram."

    sig_acquire = pyqtSignal(tuple)
    sig_release = pyqtSignal(tuple)
    sig_lock = pyqtSignal()
    sig_unlock = pyqtSignal()

    def __init__(self, app, slm, pars={}):
        super().__init__(parent=None)
        self.app = app
        self.pupilPanels = []
        self.refresh_gui = {}
        self.can_close = True
        self.close_slm = True
        self.control_enabled = True

        self.slm = slm
        self.pars = pars
        self.mutex = QMutex()

        self.setWindowTitle('SLM Control ' + version.__version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        if 'controlwindow' in pars.keys():
            self.setGeometry(pars['controlwindow'][0],
                             pars['controlwindow'][1],
                             pars['controlwindow'][2],
                             pars['controlwindow'][3])

        self.pupilsTab = QTabWidget()
        for p in self.slm.pupils:
            pp = PupilPanel(p, self.pupilsTab, self)
            self.pupilPanels.append(pp)

        geom = self.make_geometry()
        disp = self.make_general_display()
        file1 = self.make_file_tab()
        flat = self.make_flat_tab()
        wrap = self.make_wrap_tab()
        pups = self.make_pupils_group()
        grating = self.make_grating_group()

        holo = QFrame()
        holo_lay = QGridLayout()
        holo.setLayout(holo_lay)
        holo_lay.addWidget(geom, 0, 0)
        holo_lay.addWidget(flat, 0, 1)
        holo_lay.addWidget(wrap, 0, 2)
        holo_lay.addWidget(grating, 1, 0, 1, 2)
        holo_lay.addWidget(pups, 1, 2)
        holo_lay.addWidget(file1, 2, 0, 1, 3)

        tabs = QTabWidget()
        self.tabs = tabs

        front = QSplitter(Qt.Vertical)
        front.addWidget(disp)
        front.addWidget(holo)
        self.make_control_options()
        tabs.addTab(front, 'SLM')
        tabs.addTab(self.control_options, 'control')

        horiz = QSplitter(Qt.Horizontal)
        horiz.addWidget(tabs)
        horiz.addWidget(self.pupilsTab)

        # lay = QGridLayout()
        # self.setLayout(lay)
        # lay.addWidget(horiz)
        self.setCentralWidget(horiz)

        def lock():
            self.mutex.lock()
            self.can_close = False
            for i in range(self.pupilsTab.count()):
                self.pupilsTab.widget(i).setEnabled(False)
            for i in range(self.tabs.count()):
                self.tabs.widget(i).setEnabled(False)

        def unlock():
            for i in range(self.pupilsTab.count()):
                self.pupilsTab.widget(i).setEnabled(True)
            for i in range(self.tabs.count()):
                self.tabs.widget(i).setEnabled(True)
            self.can_close = True
            self.mutex.unlock()

        def make_release_hand():
            def f(t):
                for pp in self.pupilPanels:
                    for _, v in pp.refresh_gui.items():
                        v()
                for _, v in self.refresh_gui.items():
                    v()
                unlock()

            return f

        def make_acquire_hand():
            def f(t):
                lock()

            return f

        self.sig_release.connect(make_release_hand())
        self.sig_acquire.connect(make_acquire_hand())

        def make_lock_hand():
            def f():
                lock()

            return f

        def make_unlock_hand():
            def f():
                unlock()

            return f

        self.sig_lock.connect(make_lock_hand())
        self.sig_unlock.connect(make_unlock_hand())

        self.slm.refresh_hologram()

    def __str__(self):
        return (f'<slmtools.gui.{self.__class__.__name__} ' +
                f'pupils={str(len(self.slm.pupils))}>')

    def make_control_options(self):
        control_options = OptionsPanel()
        control_options.setup(self.pars, 'control',
                              SLMControls.get_default_parameters(),
                              SLMControls.get_parameters_info())
        self.control_options = control_options

    @staticmethod
    def helper_boolupdate(mycallback, myupdate):
        def f(i):
            mycallback(i)
            myupdate()

        return f

    def make_general_display(self):
        mpwin = MatplotlibWindow(self.slm, self, figsize=(8, 6))
        self.slm.refreshHologramSignal.connect(mpwin.update_array)
        self.pupilsTab.currentChanged.connect(mpwin.refresh_circle)
        return mpwin

    def make_file_tab(self):
        """Rewriting the file tab to facilitate loading of 2-pupil files"""
        g = QGroupBox('File')
        bload = QPushButton('load')
        bsave = QPushButton('save')
        bconsole = QPushButton('console')
        l1 = QGridLayout()
        l1.addWidget(bsave, 0, 0)
        l1.addWidget(bload, 0, 1)
        l1.addWidget(bconsole, 0, 2)
        g.setLayout(l1)

        def helper_load():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName(
                    self,
                    'Load SLM parameters',
                    filter='JSON (*.json);;All Files (*)')
                if fdiag:
                    bk = self.save_parameters()
                    try:
                        with open(fdiag, 'r') as f:
                            self.load_parameters(json.load(f))
                    except Exception as ex:
                        self.load_parameters(bk)
                        em = QErrorMessage()
                        em.showMessage(
                            f'Failed to load {fdiag}: {str(ex)}<br><br>' +
                            traceback.format_exc().replace('\r', '').replace(
                                '\n', '<br>'))
                        em.exec_()

            return myf1

        def helper_save():
            def myf1():
                fdiag, _ = QFileDialog.getSaveFileName(
                    self,
                    'Save parameters',
                    directory=datetime.now().strftime(
                        '%Y%m%d_%H%M%S_slm.json'),
                    filter='JSON (*.json);;All Files (*)')
                if fdiag:
                    try:
                        with open(fdiag, 'w') as f:
                            json.dump(self.save_parameters(),
                                      f,
                                      sort_keys=True,
                                      indent=4)
                    except Exception as ex:
                        em = QErrorMessage()
                        em.showMessage(
                            f'Failed to write {fdiag}: {str(ex)}<br><br>' +
                            traceback.format_exc().replace('\r', '').replace(
                                '\n', '<br>'))
                        em.exec_()

            return myf1

        def helper_console():
            def end():
                def f(t):
                    self.sig_release.emit(())
                    self.unlock_gui()

                return f

            def start(t):
                self.lock_gui()
                console = Console(self)
                console.sig_close_console.connect(end())
                console.show()

            return start

        bload.clicked.connect(helper_load())
        bsave.clicked.connect(helper_save())
        bconsole.clicked.connect(helper_console())
        return g

    def load_parameters(self, d):
        # self.setGeometry(*d['controlwindow']['geometry'])
        self.slm.load_parameters(d['slm'])
        self.pupilsTab.clear()
        self.pupilPanels.clear()
        while self.pupilsTab.count():
            self.pupilsTab.removeTab(self.pupilsTab.count() - 1)
        for i in range(len(self.slm.pupils)):
            p = self.slm.pupils[i]
            pp = PupilPanel(p, self.pupilsTab, self)
            self.pupilPanels.append(pp)
        for pp in self.pupilPanels:
            for _, f in pp.refresh_gui.items():
                f()
        for _, f in self.refresh_gui.items():
            f()

    def save_parameters(self):
        curg = self.geometry()
        return {
            'controlwindow': {
                'geometry': [curg.x(),
                             curg.y(),
                             curg.width(),
                             curg.height()],
            },
            'slm': self.slm.save_parameters(),
        }

    def make_flat_tab(self):
        def helper_load_flat1():
            def myf1():
                fdiag, _ = QFileDialog.getOpenFileName(
                    self,
                    'Select a flat file',
                    filter='Images (*.bmp *.png);;All Files (*)')
                if fdiag:
                    self.slm.set_flat(fdiag)

            return myf1

        g = QGroupBox('Flattening')
        l1 = QGridLayout()
        cboxlf = QCheckBox('on')
        cboxlf.toggled.connect(
            self.helper_boolupdate(self.slm.set_flat_on, self.slm.update))
        cboxlf.setChecked(self.slm.flat_on)
        l1.addWidget(cboxlf, 0, 0)
        loadbut = QPushButton('load')
        loadbut.clicked.connect(helper_load_flat1())
        l1.addWidget(loadbut, 0, 1)
        g.setLayout(l1)

        def f():
            def f():
                cboxlf.setChecked(self.slm.flat_on)

            return f

        self.refresh_gui['flat'] = f()
        return g

    def make_geometry(self):
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
            l1.addWidget(QLabel(txt), 0, 2 * i)
            le = QLineEdit(str(self.slm.hologram_geometry[i]))
            le.editingFinished.connect(handle_geometry(i, le))
            le.setMaximumWidth(50)
            val = MyQIntValidator()
            val.setFixup(self.slm.hologram_geometry[i])
            le.setValidator(val)
            if mini:
                val.setBottom(mini)
            l1.addWidget(le, 0, 2 * i + 1)
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

        self.refresh_gui['geometry'] = f()
        group.setLayout(l1)
        return group

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

        def f():
            def f():
                lewrap.setText(str(self.slm.wrap_value))

            return f

        self.refresh_gui['wrap'] = f()
        return g

    def make_pupils_group(self):
        g = QGroupBox('Pupils')
        l1 = QGridLayout()

        bpls = QPushButton('+')
        bmin = QPushButton('-')
        btoggle = QPushButton('toggle')

        l1.addWidget(bmin, 0, 0)
        l1.addWidget(bpls, 0, 1)
        l1.addWidget(btoggle, 1, 0)

        def fp():
            def f():
                ind = self.pupilsTab.count()
                p = self.slm.add_pupil()
                pp = PupilPanel(p, self.pupilsTab, self)
                self.pupilPanels.append(pp)
                self.pupilsTab.setCurrentIndex(ind)

            return f

        def fm():
            def f():
                if len(self.slm.pupils) == 1:
                    return
                self.pupilsTab.removeTab(len(self.slm.pupils) - 1)
                self.pupilPanels.pop()
                self.slm.pop_pupil()

            return f

        def ft():
            def f():
                for p, pp in zip(self.slm.pupils, self.pupilPanels):
                    p.set_enabled(not p.enabled, update=False)
                    pp.refresh_gui['pupil']()
                    assert (pp.pupil == p)
                self.slm.refresh_hologram()

            return f

        bpls.clicked.connect(fp())
        bmin.clicked.connect(fm())
        btoggle.clicked.connect(ft())

        g.setLayout(l1)
        return g

    def make_grating_group(self):
        g = QGroupBox('Grating')
        l1 = QGridLayout()

        def make_cb(ind):
            def f(r):
                self.slm.set_grating(r, ind)

            return f

        slider_x = RelSlider(self.slm.grating_coeffs[0], make_cb(0))
        l1.addWidget(QLabel('x'), 0, 0)
        slider_x.add_to_layout(l1, 0, 1)

        slider_y = RelSlider(self.slm.grating_coeffs[1], make_cb(1))
        l1.addWidget(QLabel('y'), 1, 0)
        slider_y.add_to_layout(l1, 1, 1)

        def f():
            def f():
                for i, s in enumerate((slider_x, slider_y)):
                    s.block()
                    s.set_value(self.slm.grating_coeffs[i])
                    s.unblock()

            return f

        self.refresh_gui['grating'] = f()

        g.setLayout(l1)
        return g

    def closeEvent(self, event):
        if self.can_close:
            if self.close_slm:
                self.slm.close()
            event.accept()
        else:
            event.ignore()

    def acquire_control(self, h5f):
        self.sig_acquire.emit((h5f, ))

        try:
            cname, pars = self.control_options.get_options()
            pars['enabled'] = self.control_enabled
            pars['pupil_index'] = self.pupilsTab.currentIndex()
            c = SLMControls.new_control(self.slm, cname, pars, h5f)
        except Exception as ex:
            self.sig_release.emit((None, h5f))
            raise ex
        return c

    def release_control(self, control, h5f):
        self.sig_release.emit((control, h5f))

    def enable_control(self, b):
        self.control_enabled = b

    def lock_gui(self):
        self.sig_lock.emit()

    def unlock_gui(self):
        self.sig_unlock.emit()


def add_arguments(parser):
    parser.add_argument('--slm-parameters',
                        type=argparse.FileType('r'),
                        default=None,
                        metavar='JSON',
                        help='Load a previous configuration file')


def new_slm_window(app, args, pars={}):
    slm = SLM()
    slm.show()

    cwin = SLMWindow(app, slm)
    cwin.show()

    # argparse specified parameters can override pars
    if args.slm_parameters is not None:
        d = json.loads(args.slm_parameters.read())
        pars = {**pars, **d}
        args.slm_parameters = args.slm_parameters.name

    cwin.load_parameters(pars)

    return cwin


class Console(QDialog):
    sig_close_console = pyqtSignal(tuple)

    def __init__(self, slmwin):
        super().__init__(None)
        self.slmwin = slmwin

        self.setWindowTitle('SLM console ' + version.__version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.shell.banner2 = (
            "Use 'run -i' to run an external script and " +
            "'draw()' to update the hologram.\n")

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()
        kernel_client.namespace = self

        layout = QVBoxLayout(self)
        widget = RichJupyterWidget(parent=self)
        layout.addWidget(widget)
        widget.kernel_manager = kernel_manager
        widget.kernel_client = kernel_client
        self.show()

        def stop():
            def f():
                "Close the console"
                self.sig_close_console.emit((None, ))
                kernel_client.stop_channels()
                kernel_manager.shutdown_kernel()

            return f

        def draw():
            def f():
                "Update the SLM hologram"
                self.slmwin.app.processEvents()

            return f

        self.stop = stop()
        widget.exit_requested.connect(stop)

        kernel_manager.kernel.shell.push({
            'np': np,
            'plt': plt,
            'slmwin': self.slmwin,
            'slm': self.slmwin.slm,
            'draw': draw(),
            'exit': self.close,
            'quit': self.close,
        })

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='SLM control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no-file-log', action='store_true')
    parser.add_argument('--file-log', action='store_false', dest='no_file_log')
    parser.set_defaults(no_file_log=True)
    parser.add_argument('--load',
                        type=argparse.FileType('r'),
                        default=None,
                        metavar='JSON',
                        help='Load a previous configuration file')
    args = parser.parse_args(args[1:])

    if not args.no_file_log:
        fn = datetime.now().strftime('%Y%m%d-%H%M%S-' + str(os.getpid()) +
                                     '.log')
        logging.basicConfig(filename=fn, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    slm = SLM()
    slm.show()

    cwin = SLMWindow(app, slm)
    cwin.show()

    if args.load:
        d = json.loads(args.load.read())
        cwin.load_parameters(d)

    sys.exit(app.exec_())
