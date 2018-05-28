# Package import
# import pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.utils import flatten, unflatten
import numpy as np
import pysparse
from skimage import data

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

from pysap.plugins.mri.reconstruct.linear import Wavelet2

# type_mr_2D_3D = [14, 1]  # MALLAT
type_mr_2D_3D = [29, 2]  # LIFTING
# type_mr_2D_3D = [1, 3]  # ATROU

type_mr_2D, type_mr_3D = type_mr_2D_3D[0], type_mr_2D_3D[1]

nb_scales = 3


data = data.astronaut().reshape((128, 128, 48))
print('DATA SHAPE:', data.shape)
trf2D = pysparse.MRTransform(type_of_multiresolution_transform=type_mr_2D,
                             number_of_scales=nb_scales, verbose=1)
trf2D.opath = '~/src/datas/save_bindings/test_2D.mr'

analysis_data_2D, csh = trf2D.transform(data[:, :, 0].astype(np.double),
                                        save=False)
print('NB BANDS 2D:', len(analysis_data_2D))
print('SHAPE FOR EACH BAND', [s.shape for s in analysis_data_2D])
print('-----------------------------------')
trf2D.analysis_data = analysis_data_2D

data_recons_2D = trf2D.reconstruct(analysis_data_2D)
print('RECONS SHAPE:', data_recons_2D.shape)
print('DIFF RECONS', (data_recons_2D - data[:, :, 0]).sum())


print('\n')
print('--------------------------------\n')
print('--------------- 3D -------------\n')
print('--------------------------------\n')


trf3D = pysparse.MRTransform3D(type_of_multiresolution_transform=type_mr_3D,
                               type_of_lifting_transform=3,
                               number_of_scales=nb_scales,
                               iter=3,
                               type_of_filters=1,
                               use_l2_norm=False,
                               nb_procs=0,
                               verbose=1)

analysis_data, nb_band_per_scale = trf3D.transform(data.astype(np.double),
                                                   save=False)


print('-----------------------------------')

print('NB BANDS 3D:', len(analysis_data))
print('SHAPE FOR EACH BAND', [s.shape for s in analysis_data])
print('BANDS SHAPE:', nb_band_per_scale)
print('-----------------------------------')


# np.save('/volatile/bsarthou/cube_trans_mallat_bind.npy',
#         np.array(analysis_data))

# coeffs, coeffs_shape = flatten(analysis_data)
# print('SHAPE:', coeffs_shape)
# print(coeffs)
# print(len(coeffs))
# synth_data = unflatten(coeffs, coeffs_shape)
# print('UNFLATTEN SHAPE:', len(synth_data))
# print('DIFF FLATTEN:', np.sum([np.sum(x-y) for x, y
#                                in zip(analysis_data, synth_data)]))

# print('-----------------------------------')
# print(analysis_data)
# print(synth_data)
print('-----------------------------------')


data_recons = trf3D.reconstruct(analysis_data)
print('RECONS SHAPE:', data_recons.shape)
print('DIFF', (data_recons - data).sum())
print('MSE 10:', sqrt(mse(data_recons[:, :, 10], data[:, :, 10])))

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(data[:, :, 10], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(data_recons[:, :, 10], cmap='gray')
plt.suptitle('Pysparse3D')
plt.show()
# plt.pause(0.001)
# input("Press [enter] to continue.")

data_binding = data_recons

print('\n')
print('---------------------------------------\n')
print('--------------- Linear Op -------------\n')
print('---------------------------------------\n')

wavelet_name = 'Wavelet3DTransformViaLiftingScheme'

linear_op = Wavelet2(
        nb_scale=nb_scales,
        wavelet_name=wavelet_name)

coeffs = linear_op.op(data)
image = linear_op.adj_op(coeffs)

print('RECONS SHAPE:', image.shape)
print('DIFF', (image - data).sum())
print('MSE 10:', sqrt(mse(image[:, :, 10], data[:, :, 10])))

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(data[:, :, 10], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image[:, :, 10], cmap='gray')
plt.suptitle('Wavelet3D')
plt.show()

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(data_binding[:, :, 10], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image[:, :, 10], cmap='gray')
plt.suptitle('Comparaison: ' + str(sqrt(mse(data_binding[:, :, 10],
                                            image[:, :, 10]))))
plt.show()
