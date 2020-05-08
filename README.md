# slmtools

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOE.393363-blue)](https://doi.org/10.1364/OE.393363)

Python code to use a spatial light modulator (SLM).

![](./media/screenshot.png)

## Main Features

* support for the vortex and top-hat phase masks in stimulated emission
  depletion (STED) microscopy
* aberration control via Zernike polynomials
* support for multiple pupils over the same SLM window

## Installation

The **easiest** way to install `slmtools` is to first install [Anaconda for Python
3](https://www.anaconda.com/download). After that, open an `Anaconda Prompt`,
type `pip install --user slmtools` and hit enter to install `slmtools`.

To start up the GUI, open `Anaconda Prompt`, type `python -m slmtools.gui` and
hit enter.

## Installation from the GitHub repository

To install `slmtools` in Windows, follow the steps below.

- You should first install the following software requirements:
    - [Anaconda for Python 3](https://www.anaconda.com/download). This includes
      Python as well as some necessary scientific libraries.
    - [Git](https://git-scm.com/download/win). This is necessary for the
      automatic version numbering of this package. Also, make sure you choose
      *Git from the command line and also 3rd-party software* in *Adjusting
      your PATH environment*.
- *Clone* this repository using Git. From any folder in File Explorer,
  right-click and hit *Git Bash here*. Paste `git clone
  https://github.com/jacopoantonello/slmtools` and hit enter. Do not use GitHub's
  *Download ZIP* button above, as the installation script will not work in that
  case.
- Finally, double-click on `install.bat`.

To run the GUI, double-click on `run.bat`.
