#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.random import normal, uniform
from PyQt5.QtWidgets import QApplication
from slmtools import (get_Ngrays, load_background, merge_hologram_bits,
                      save_background)
from slmtools.gui import SLM
from slmtools.test import load_flat

app = QApplication(sys.argv)  # noqa

tmpflat = NamedTemporaryFile(delete=False, suffix='.png')
tmpflat.close()
flat = load_flat()
save_background(tmpflat.name, flat)
flat1 = load_background(tmpflat.name)
assert (np.allclose(flat, flat1))
print(f'{flat.shape=}')
print(f'{flat1.shape=}')

# check grayscale mappings
wrap = int(np.round(uniform(1, 255)))
wrap = 200  # DEBUG
print(f'{wrap=}')
Ngrays, gray2phi = get_Ngrays(wrap)
grays = np.arange(0, Ngrays)
phs = gray2phi * np.arange(0, Ngrays)
assert (np.allclose(Ngrays, grays.size))
assert (np.allclose(Ngrays, phs.size))
assert (np.allclose(phs[0], 0))
assert (np.allclose(phs[-1] + (phs[1] - phs[0]), 2 * np.pi))

# make a random hologram scene
holo = SLM()
holo.set_wrap_value(wrap)
pupil = holo.add_pupil()
holo.set_flat(tmpflat.name, refresh_hologram=False)
holo.set_flat_on(True)
min1 = min((holo.hologram_geometry[2] / 2, holo.hologram_geometry[3] / 2))
xy = np.array([
    uniform(-holo.hologram_geometry[2] / 2, -holo.hologram_geometry[2] / 2),
    uniform(-holo.hologram_geometry[3] / 2, -holo.hologram_geometry[3] / 2)
])
rho = uniform(.2 * min1, .8 * min1)
xy *= 0  # DEBUG
rho = 100  # DEBUG
pupil.set_xy(xy)
pupil.set_rho(rho)
print(pupil.xy, pupil.rho)
print(f'{pupil.xy=} {pupil.rho=} ')
zc = normal(size=(28, ))
zc /= norm(zc)
zc *= 0  # DEBUG
pupil.set_aberration(zc)
back, grating, phi, mask, wrap1 = holo.make_hologram_bits()
hl = merge_hologram_bits(back, grating, phi, mask, wrap)
assert (np.allclose(wrap, wrap1))

# check background
back0 = np.remainder(flat.astype(np.float), Ngrays)
back1 = hl.astype(np.float)
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
plt.show()
assert (np.allclose(back0[mask], back1[mask]))

os.unlink(tmpflat.name)
