#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from slmtools import version

__author__ = 'J Antonello, A. Barbotin'
__copyright__ = 'Copyright 2018-2020 J Antonello and A. Barbotin'
__license__ = 'GPLv3+'
__email__ = 'jacopo@antonello.org, aurelien.barbotin@dtc.ox.ac.uk'
__status__ = 'Prototype'
__all__ = ['gui', 'version']
__date__ = version.__date__
__version__ = version.__version__
__commit__ = version.__commit__
__doc__ = """
Spatial light modulator in Python.

author:  {}
date:    {}
version: {}
commit:  {}
""".format(__author__, __date__, __version__, __commit__)

GRAY_DTYPE = np.uint8


def get_Ngrays(wrap):
    Ngrays = wrap + 1
    gray2phi = 2 * np.pi / Ngrays
    return Ngrays, gray2phi


def merge_hologram_bits(back, grating, phi, mask, wrap):

    assert (back.dtype == GRAY_DTYPE)
    assert (grating.dtype == np.float)
    assert (phi.dtype == np.float)
    assert (mask.dtype == np.bool)

    mph = np.logical_not(mask)
    mgr = mask

    Ngrays, gray2phi = get_Ngrays(wrap)

    gphi = np.round(phi[mph] / gray2phi).astype(GRAY_DTYPE)
    ggrt = np.round(grating[mgr] / gray2phi).astype(GRAY_DTYPE)

    holo = back.copy()
    holo[mph] = np.remainder(holo[mph] + gphi, Ngrays)
    holo[mgr] = np.remainder(holo[mgr] + ggrt, Ngrays)
    assert (holo.min() >= 0)
    assert (holo.max() <= wrap)
    assert (holo.dtype == GRAY_DTYPE)

    return holo
