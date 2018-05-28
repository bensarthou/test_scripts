#!/bin/sh

# CONDAT
python3 test_wavelet3D.py 1 10 1 200
python3 test_wavelet3D.py 1 10 0 200

# FISTA
python3 test_wavelet3D.py 0 10 1 200
python3 test_wavelet3D.py 0 10 0 200
