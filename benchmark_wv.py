# Package import
from pysap.plugins.mri.reconstruct_3D.utils import imshow3D
from pysap.plugins.mri.reconstruct_3D.utils import normalize_samples
from pysap.plugins.mri.reconstruct_3D.utils import convert_mask_to_locations_3D
from pysap.plugins.mri.parallel_mri.gradient import Gradient_pMRI

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
# pwt_name = 'bior3.3'
# sparse3d_name = 'BiOrthogonalTransform3D'
# isap_filter_id = 13
# sp3d_title = 'BiOr44'
nb_scale = 3
R = 10

list_pwt = ['bior6.8', 'db4', 'bior2.4', 'bior3.3']
list_sp3d = [{'name': 'BiOr79', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 1},
             {'name': 'Db4', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 2},
             {'name': 'Bior35', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 12},
             {'name': 'Bior44', 'sp3d_name': 'BiOrthogonalTransform3D',
              'filter_id': 13}]

# Load input data
filename = '/volatile/bsarthou/datas/' \
            'meas_MID14_gre_800um_iso_128x128x128_FID24.mat'
Iref = loadmat(filename)['ref']

for (i, pwt_name) in enumerate(list_pwt):
    for (j, dic) in enumerate(list_sp3d):
        print('Pywavelet:', pwt_name)
        time_pwt_trans, time_pwt_recons, error_pwt = [], [], []
        for k in range(R):
            linear_op = pyWavelet3(wavelet_name=pwt_name,
                                   nb_scale=nb_scale)
            start = time.clock()
            coeffs = linear_op.op(Iref)
            mid = time.clock()
            recons1 = linear_op.adj_op(coeffs)
            end = time.clock()
            time_pwt_trans.append(mid - start)
            time_pwt_recons.append(end - mid)
            error_pwt.append(np.abs((Iref-recons1).mean()))
            # print('Trans time:', mid - start, 'Recons time:', end - mid)

        print('{}: Trans, recons, error=({},{}, {})'.format(
                                                      pwt_name,
                                                      np.mean(time_pwt_trans),
                                                      np.mean(time_pwt_recons),
                                                      np.mean(error_pwt)))

        wt_type = dic['sp3d_name']
        sp3d_name = dic['name']
        kwargs = dict()
        kwargs["type_of_filters"] = dic['filter_id']
        time_sp3d_trans, time_sp3d_recons, error_sp3d = [], [], []

        print('Sparse3D:', sp3d_name)
        for k in range(R):
            linear_op = Wavelet2(nb_scale=nb_scale-1,
                                 wavelet_name=wt_type,
                                 **kwargs)
            start = time.clock()
            coeffs = linear_op.op(Iref)
            mid = time.clock()
            recons2 = linear_op.adj_op(coeffs)
            end = time.clock()
            time_sp3d_trans.append(mid - start)
            time_sp3d_recons.append(end - mid)
            error_sp3d.append(np.abs((Iref-recons2).mean()))

            # print('Trans time:', mid - start, 'Recons time:', end - mid)
        print('{}: Trans, recons, error=({},{}, {})'.format(
                                                     sp3d_name,
                                                     np.mean(time_sp3d_trans),
                                                     np.mean(time_sp3d_recons),
                                                     np.mean(error_sp3d)))
        results = {'pwt_name': pwt_name, 'sp3d_name': sp3d_name,
                   't_pwt': [np.mean(time_pwt_trans),
                             np.mean(time_pwt_recons)],
                   't_sp3d': [np.mean(time_sp3d_trans),
                              np.mean(time_sp3d_recons)],
                   't_diff': [np.mean(time_pwt_trans) -
                              np.mean(time_sp3d_trans),
                              np.mean(time_pwt_recons) -
                              np.mean(time_sp3d_recons)],
                   'error_pwt': np.mean(error_pwt),
                   'error_sp3d': np.mean(error_sp3d),
                   'error_cross': np.abs(recons1 - recons2).mean()}
        np.save('/volatile/bsarthou/datas/wv_benchmark/loop_'+pwt_name+'_VS_' +
                sp3d_name+'.npy', results)
        # print('DIFF', np.abs(recons1 - recons2).mean())
