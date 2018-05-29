#!/bin/sh

# python3 test_wavelet3D.py 1 2 0 5

# CONDAT
python3 test_wavelet3D.py 1 5 1 200
python3 test_wavelet3D.py 1 5 0 200

# FISTA
python3 test_wavelet3D.py 0 5 1 200
python3 test_wavelet3D.py 0 5 0 200
