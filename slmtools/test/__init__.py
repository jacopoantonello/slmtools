#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import dirname, join

from scipy.io import loadmat


def load_flat():
    return loadmat(join(dirname(__file__), 'flat.mat'))['flat']
