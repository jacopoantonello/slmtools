#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import unittest

from PyQt5.QtWidgets import QApplication
from slmtools.test.test1_plot import run_tests

app = QApplication(sys.argv)  # noqa


class TestHologram(unittest.TestCase):
    def test1(self):
        r1, r2 = run_tests(False)
        self.assertTrue(r1)
        self.assertTrue(r2)


if __name__ == '__main__':
    unittest.main()
