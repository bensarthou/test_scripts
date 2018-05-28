# Package import
from utils import imshow3D
from utils import normalize_samples
from utils import convert_locations_to_mask
from utils import convert_mask_to_locations_3D
from linear import pyWavelet3
from utils import flatten_wave
from utils import unflatten_wave


# Third party import
import numpy as np
import scipy.fftpack as pfft
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pywt
import time



# Load input data
filename = '/volatile/bsarthou/' \
            'meas_MID14_gre_800um_iso_128x128x128_FID24.mat'
Iref = loadmat(filename)['ref']

# imshow3D(Iref, display=True)

samples = loadmat('/volatile/bsarthou/'
                  'samples_sparkling_3D_N128_502x1536x8_FID4971.mat')['samples']
samples = normalize_samples(samples)
cartesian_samples = convert_locations_to_mask(samples, [128, 128, 128])
# imshow3D(cartesian_samples, display=True)

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
kspace_mask = pfft.ifftshift(cartesian_samples)
kspace_data = pfft.fftn(Iref) * kspace_mask

# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations_3D(kspace_mask)

# Zero order solution
image_rec0 = pfft.ifftn(kspace_data)
# imshow3D(np.abs(image_rec0), display=True)

linear_op = pyWavelet3(wavelet_name="sym4",
                       nb_scale=4)

R = 10
old_coeffs = None

print('LOOP LINEAR')

for r in range(R):

    trf = pywt.Wavelet("sym4")

    coeffs = linear_op.op(image_rec0)
    # time.sleep(10)
    if old_coeffs is not None:
        print('DIFF COEFFS: ', coeffs - old_coeffs)
    old_coeffs = coeffs

    recons = linear_op.adj_op(coeffs)
    # imshow3D(np.abs(recons), display=True)
    print('DIFF RECONS: ', np.abs((image_rec0-recons).sum()))


print('LOOP PYWT')

trf = pywt.Wavelet("sym4")
save_coeffs_pywt = None
for r in range(R):

    coeffs_dict_1 = pywt.wavedecn(image_rec0,
                                  trf,
                                  level=3)
    # np.save('/volatile/bsarthou/datas/flatten/coeffs_dict_1.npy', coeffs_dict_1)
    coeffs_pywt, coeffs_shape = flatten_wave(coeffs_dict_1)
    coeffs_dict = unflatten_wave(coeffs_pywt, coeffs_shape)
    recons_pywt = pywt.waverecn(coeffs=coeffs_dict, wavelet=trf)
    # imshow3D(np.abs(recons_pywt), display=True)
    print('DIFF RECONS PYWT: ', np.abs((image_rec0-recons_pywt).sum()))
