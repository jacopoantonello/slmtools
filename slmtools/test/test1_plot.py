#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.ma import masked_array
from numpy.random import normal, uniform
from PyQt5.QtWidgets import QApplication
from skimage.restoration import unwrap_phase
from slmtools import (get_Ngrays, load_background, merge_hologram_bits,
                      save_background)
from slmtools.gui import Hologram
from slmtools.test import load_flat


def run_tests(show_plot):
    tmpflat = NamedTemporaryFile(delete=False, suffix='.png')
    tmpflat.close()
    flat = load_flat()
    save_background(tmpflat.name, flat)
    flat1 = load_background(tmpflat.name)
    assert (np.allclose(flat, flat1))
    if show_plot:
        print(f'{flat.shape=}')
        print(f'{flat1.shape=}')

    # check grayscale mappings
    wrap = int(np.round(uniform(1, 255)))
    if show_plot:
        print(f'{wrap=}')
    Ngrays, gray2phi = get_Ngrays(wrap)
    grays = np.arange(0, Ngrays)
    phs = gray2phi * np.arange(0, Ngrays)
    assert (np.allclose(Ngrays, grays.size))
    assert (np.allclose(Ngrays, phs.size))
    assert (np.allclose(phs[0], 0))
    assert (np.allclose(phs[-1] + (phs[1] - phs[0]), 2 * np.pi))

    # make a random hologram scene
    holo = Hologram()
    holo.set_wrap_value(wrap)
    pupil = holo.pupils[0]
    holo.set_flat(tmpflat.name, refresh_hologram=False)
    os.unlink(tmpflat.name)
    holo.set_flat_on(True)
    min1 = min((holo.hologram_geometry[2] / 2, holo.hologram_geometry[3] / 2))
    xy = np.array([
        uniform(-holo.hologram_geometry[2] / 2, holo.hologram_geometry[2] / 2),
        uniform(-holo.hologram_geometry[3] / 2, holo.hologram_geometry[3] / 2)
    ])
    rho = uniform(.2 * min1, .8 * min1)
    pupil.set_xy(xy)
    pupil.set_rho(rho)
    if show_plot:
        print(pupil.xy, pupil.rho)
        print(f'{pupil.xy=} {pupil.rho=} ')
    zc = normal(size=(28, ))
    zc /= norm(zc)
    pupil.set_aberration(zc)
    back, grating, phi, mask, wrap1 = holo.make_hologram_bits()
    hl = merge_hologram_bits(back, grating, phi, mask, wrap)
    assert (np.allclose(wrap, wrap1))
    mask_bg = mask
    mask_ap = np.logical_not(mask)
    del mask

    # check background
    back0 = np.remainder(flat.astype(np.float), Ngrays)
    back1 = hl.astype(np.float)

    if show_plot:
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(back0)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(back1)
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(back1 - back0)
        plt.colorbar()

    result_test1 = np.allclose(back0[mask_bg], back1[mask_bg])

    # check pupil phase
    ph0 = phi.copy()
    ph1 = (hl.astype(np.float) - back0.astype(np.float)) * gray2phi
    ph2 = np.array(unwrap_phase(masked_array(ph1, mask_bg)))

    ph0[mask_ap] -= ph0[mask_ap].mean()
    ph2[mask_ap] -= ph2[mask_ap].mean()
    err = ph0 - ph2

    ph0[mask_bg] = -np.inf
    ph1[mask_bg] = -np.inf
    ph2[mask_bg] = -np.inf
    err[mask_bg] = -np.inf

    def apply_remainder(p1):
        p2 = p1.copy()
        p2[mask_ap] = np.remainder(p2[mask_ap] / gray2phi, Ngrays)
        return p2

    if show_plot:
        plt.figure(2)

        nn = 3
        mm = 3

        plt.subplot(nn, mm, 1)
        plt.imshow(ph0)
        plt.colorbar()
        plt.subplot(nn, mm, 2)
        plt.imshow(apply_remainder(ph0))
        plt.colorbar()

        plt.subplot(nn, mm, 4)
        plt.imshow(ph1)
        plt.colorbar()
        plt.subplot(nn, mm, 5)
        plt.imshow(apply_remainder(ph1))
        plt.colorbar()

        plt.subplot(nn, mm, 7)
        plt.imshow(ph2)
        plt.colorbar()
        plt.subplot(nn, mm, 8)
        plt.imshow(apply_remainder(ph2))
        plt.colorbar()

        plt.subplot(nn, mm, 9)
        plt.imshow(err)
        plt.colorbar()

        plt.show()

    result_test2 = np.abs(err[mask_ap]).max() < 2 * np.pi / Ngrays

    return result_test1, result_test2


if __name__ == '__main__':
    app = QApplication(sys.argv)  # noqa
    r1, r2 = run_tests(True)
    assert (r1)
    assert (r2)
