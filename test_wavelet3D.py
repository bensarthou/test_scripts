"""
Neuroimaging cartesian reconstruction
=====================================

Credit: S Lannuzel, L Elgueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.reconstruct_3D.utils import convert_locations_to_mask
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.parallel_mri.gradient import Grad_pMRI

from pysap.plugins.mri.reconstruct_3D.linear import pyWavelet3
from pysap.plugins.mri.reconstruct.linear import Wavelet2
from pysap.plugins.mri.reconstruct_3D.fourier import NFFT3
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_fista
from pysap.plugins.mri.parallel_mri.reconstruct import sparse_rec_condatvu


# Third party import
import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from modopt.math.metrics import ssim, snr, psnr, nrmse

c, r = int(sys.argv[1]), int(sys.argv[2])
p, m = int(sys.argv[3]), int(sys.argv[4])

if c == 1:
    CONDAT = True  # choose which algo to use
else:
    CONDAT = False
nb_runs = r  # nb of call to the opti algorithm
if p == 1:
    PWT = True  # Use of Pywavelet transforms, or Sparse3D bindings
else:
    PWT = False
max_iter = m  # Nb max of iterations in reconstruction

# Load input data
filename = '/volatile/temp_bs/meas_MID14_gre_800um_iso_128x128x128_FID24.mat'
Iref = loadmat(filename)['ref']

# imshow3D(Iref, display=True)

samples = loadmat('/volatile/temp_bs/'
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
kspace_loc = convert_mask_to_locations_3D(cartesian_samples)
fourier_op_gen = NFFT3(samples=kspace_loc, shape=Iref.shape)
kspace_data = fourier_op_gen.op(Iref)

# Zero order solution
image_rec0 = fourier_op_gen.adj_op(kspace_data)
imshow3D(np.abs(image_rec0), display=False)


tab_time = np.zeros((1, nb_runs))
tab_metrics = np.zeros((4, nb_runs))
list_cost = []

if CONDAT is False:

    for run in range(nb_runs):
        start = time.clock()

        if PWT:
            linear_op = pyWavelet3(wavelet_name="bior6.8",
                                   nb_scale=3)
        else:
            linear_op = Wavelet2(nb_scale=3, wavelet_name='Mallat3DWaveletTransform79Filters')

        fourier_op = NFFT3(samples=kspace_loc, shape=Iref.shape)
        gradient_op = Grad_pMRI(data=kspace_data,
                                fourier_op=fourier_op,
                                linear_op=linear_op)

        x_final, transform, cost = sparse_rec_fista(
            gradient_op=gradient_op,
            linear_op=linear_op,
            mu=0,
            lambda_init=1.0,
            max_nb_of_iter=max_iter,
            atol=1e-4,
            verbose=1,
            get_cost=True)

        end = time.clock()

        tab_time[0, run] = end - start

        tab_metrics[0, run] = ssim(np.abs(Iref), np.abs(x_final), mask=None)
        tab_metrics[1, run] = snr(np.abs(Iref), np.abs(x_final), mask=None)
        tab_metrics[2, run] = psnr(np.abs(Iref), np.abs(x_final), mask=None)
        tab_metrics[3, run] = nrmse(np.abs(Iref), np.abs(x_final), mask=None)

        list_cost.append(cost)

else:

    for run in range(nb_runs):
        start = time.clock()

        if PWT:
            linear_op = pyWavelet3(wavelet_name="bior6.8",
                                   nb_scale=3)
        else:
            linear_op = Wavelet2(nb_scale=3, wavelet_name='ATrou3D')

        fourier_op = NFFT3(samples=kspace_loc, shape=Iref.shape)

        gradient_op_cd = Grad_pMRI(data=kspace_data,
                                   fourier_op=fourier_op)
        x_final, transformn, cost = sparse_rec_condatvu(
            gradient_op=gradient_op_cd,
            linear_op=linear_op,
            std_est=None,
            std_est_method="dual",
            std_thr=2.,
            mu=1e-5,
            tau=None,
            sigma=None,
            relaxation_factor=1.0,
            nb_of_reweights=0,
            max_nb_of_iter=max_iter,
            add_positivity=False,
            atol=1e-4,
            verbose=1,
            get_cost=True)
        end = time.clock()

        # plt.figure()
        # plt.plot(cost)
        # plt.show()

        tab_time[0, run] = end - start

        tab_metrics[0, run] = ssim(np.abs(Iref), np.abs(x_final), mask=None)
        tab_metrics[1, run] = snr(np.abs(Iref), np.abs(x_final), mask=None)
        tab_metrics[2, run] = psnr(np.abs(Iref), np.abs(x_final), mask=None)
        tab_metrics[3, run] = nrmse(np.abs(Iref), np.abs(x_final), mask=None)

        list_cost.append(cost)


print('TIME')
print(tab_time)
print('METRICS')
print(tab_metrics)


np.save('/volatile/temp_bs/save_wavelet3_mallat_'
        + str(c)+'_'+str(r)+'_'+str(p)+'_'+str(m) +
        '.npy', {'time': tab_time, 'metrics': tab_metrics, 'cost': list_cost})
