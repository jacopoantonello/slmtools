#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from os import path
from subprocess import check_output

from setuptools import find_packages, setup

# Get the long description from the relevant file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def update_version():
    try:
        toks = check_output('git describe --tags --long --dirty',
                            universal_newlines=True,
                            shell=True).strip().split('-')
        version = toks[0].strip('v') + '.' + toks[1]
        last = check_output('git log -n 1',
                            universal_newlines=True,
                            shell=True)
        date = re.search(r'^Date:\s+([^\s].*)$', last, re.MULTILINE).group(1)
        commit = re.search(r'^commit\s+([^\s]{40})', last,
                           re.MULTILINE).group(1)

        with open(path.join('slmtools', 'version.py'), 'w', newline='\n') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('# -*- coding: utf-8 -*-\n\n')
            f.write(f"__version__ = '{version}'\n")
            f.write(f"__date__ = '{date}'\n")
            f.write(f"__commit__ = '{commit}'")
    except Exception as e:
        print(f'Cannot update version: {str(e)}', file=sys.stderr)
        print('Is Git installed?', file=sys.stderr)


update_version()


def lookup_version():
    with open(os.path.join('slmtools', 'version.py'), 'r') as f:
        m = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    return m.group(1)


setup(
    name='slmtools',
    version=lookup_version(),
    description='Spatial light modulator in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jacopoantonello/slmtools',
    maintainer='The SLM Project Developers',
    license='GPLv3+',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Physics',
        ('License :: OSI Approved :: ' +
         'GNU General Public License v3 or later (GPLv3+)'),
        'Programming Language :: Python :: 3',
    ],
    install_requires=['numpy', 'matplotlib', 'PyQt5', 'zernike', 'h5py'],
    packages=find_packages(exclude=['tests*', 'examples*']),
    python_requires='>=3.7',
)
