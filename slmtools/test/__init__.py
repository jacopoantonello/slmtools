#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import dirname, join

from scipy.io import loadmat


def get_flat_path():
    return join(dirname(__file__), 'flat.mat')


def load_flat():
    return loadmat(get_flat_path())['flat']
