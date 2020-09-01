#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import dirname, join

from slmtools import load_background


def get_flat_path():
    return join(dirname(__file__), 'flat.png')


def load_flat():
    return load_background(get_flat_path())
