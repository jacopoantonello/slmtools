#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
