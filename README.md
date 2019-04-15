# SLM
Spatial light modulator in Python.

# Main Features
* 2D and 3D STED masks
* double pass
* Zernike aberration control

# Required packages
* [zernike](https://github.com/jacopoantonello/zernike/)

# Develop installation
```bash
python setup.py develop --user
```

# Static installation
For Linux use

```bash
python setup.py install --user
```

For Windows use

```bash
python setup.py bdist_wheel
pip install dist\*.whl
```

# How to run
Open a terminal and navigate to a folder outside the `slm` source folder

```bash
python -m slm.slm
```
    
Or in double click in `run_gui.bat` in Windows.
