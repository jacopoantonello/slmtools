SLM
===
Spatial light modulator in Python.

Main Features
-------------
* 2D and 3D STED masks
* double pass
* Zernike aberration control

Develop installation
--------------------
    $ python setup.py develop --user

Static installation
------------------
For Linux use

    $ python setup.py install --user

For Windows use

    $ python setup.py bdist_wheel
    $ pip install dist\*.whl

How to run
----------
Open a terminal and navigate to a folder outside the `slm` source folder

    $ python -m slm.slm
    
Or in double click in `run_gui.bat` in Windows.
