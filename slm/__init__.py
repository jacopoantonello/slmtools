#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from slm import version

__author__ = 'J Antonello, A. Barbotin'
__copyright__ = 'Copyright 2019 J Antonello and A. Barbotin'
__license__ = 'GPLv3+'
__email__ = 'jacopo@antonello.org, aurelien.barbotin@dtc.ox.ac.uk'
__status__ = 'Prototype'
__all__ = ['slm', 'version']
__doc__ = """
Spatial light modulator in Python.

author:  {}
date:    {}
version: {}
commit:  {}
""".format(
    __author__,
    version.__date__,
    version.__version__,
    version.__commit__)
