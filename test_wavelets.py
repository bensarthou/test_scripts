import pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.fourier import NFFT2
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as pfft


# Loading input data
image = get_sample_data("mri-slice-nifti")
image.data += np.random.randn(*image.shape) * 20.
mask = get_sample_data("mri-mask")
# image.show()
# mask.show()

# Get the locations of the kspace samples and the associated observations
kspace_loc = convert_mask_to_locations(mask.data)
fourier_op = NFFT2(samples=kspace_loc, shape=image.shape)
kspace_obs = fourier_op.op(image.data)

data = fourier_op.adj_op(kspace_obs)

image = pysap.Image(data=data)
image.show()


wt_list_not_ok = [
           'MallatWaveletTransform79Filters',
           'HaarWaveletTransform',
           'WaveletTransformViaLiftingScheme',
]
wavelet_name = wt_list_not_ok[0]
wavelet_name = 'PyramidalLinearWaveletTransform'
nb_scales = 4

print('With bindings')

linear_op = Wavelet2(
    nb_scale=nb_scales,
    wavelet_name=wavelet_name)

alpha1 = linear_op.op(data)

print('With binaries')
linear_op2 = Wavelet2(
    nb_scale=nb_scales,
    wavelet_name=wavelet_name)

linear_op2.transform.use_wrapping = True

alpha2 = linear_op2.op(data)

print(alpha1)
print(alpha2)

plt.figure()
plt.subplot(121)
plt.plot(np.real(alpha1), np.imag(alpha1), '.k')
plt.subplot(122)
plt.plot(np.real(alpha2), np.imag(alpha2), '.k')
plt.show()
