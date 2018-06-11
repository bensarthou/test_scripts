from scipy.io import loadmat
import pandas as pd
import numpy as np
from modopt.math.metrics import ssim, snr, psnr, nrmse
from wrapper_pysap import recons_pysap
from pysap.data import get_sample_data
from pysap.plugins.mri.reconstruct.utils import convert_mask_to_locations


#############################################################################
# Loading the train and test image datasets
# -----------------------------------------
#
# We will train our models on a set of images, and also test reconstruction on
# another one. Output of this section must be two list of np.arrays, 2D or 3D
print('LOADING IMAGES')
images_directory = '/volatile/bsarthou/datas/FLASH3D/'
train_set = pd.HDFStore(images_directory + 'training_data.h5')['df']
test_set = pd.HDFStore(images_directory + 'testing_data.h5')['df']
X_train = np.dstack(train_set['data'])
X_test = np.dstack(test_set['data'])

pad = np.zeros((30, 384, 400))

X_train = np.concatenate((pad, X_train, pad), axis=0)

# # Reshape for squared images (256*256)
# X_train = X_train[((X_train.shape[0]//2)-128):((X_train.shape[0]//2)+128),
#                   ((X_train.shape[1]//2)-128):((X_train.shape[1]//2)+128), :]
#
# X_test = X_test[((X_test.shape[0]//2)-128):((X_test.shape[0]//2)+128),
#                 ((X_test.shape[1]//2)-128):((X_test.shape[1]//2)+128), :]

X_train, X_test = X_train.transpose(), X_test.transpose()

shape_img = (X_train[0].shape)

print('Number of images in the db:', X_train.shape[0])
X_train = X_train[:40, :, :]


#############################################################################
# Loading the sampling patterns we want to test
# ---------------------------------------------
#
# This algorithm explores a discrete sampling space,in which all binary mask
# will be loaded in masks list.
print('LOADING SAMPLES')
# samples_dirpath = '/home/bs255482/.local/share/pysap/'
# samples_paths = ['samples_radial_x15_34x3072.mat',
#                  'samples_sparkling_x15_34x3072.mat',
#                  'samples_radial_x8_64x3072.mat',
#                  'samples_sparkling_x8_64x3072.mat']

samples_dirpath = '/volatile/bsarthou/datas/sparkling/samples_sparkling_/' + \
                  'N384_radsym_nc24/ns3072/'
samples_paths = []
base_dir = '2018-Jun-06_N384_nc24_ns3072_Dt0.01ms_OS1_decim1_'
for decay in [1, 1.75, 2.5, 3.25, 4]:
    for tau in [0.5, 0.7, 0.9, 1, 1.2]:
        path = base_dir + 'decay{}_tau{}/'.format(decay, tau) + \
               'Samples_SPARKLING_N384_R2_nc24x3072_Dt10us.mat'
        samples_paths.append(path)

print(samples_paths)
masks = [loadmat(samples_dirpath + path) for path in samples_paths]
masks = [mask['samples'] for mask in masks]

# Normalisation between [-0.5, 0.5[
masks = [mask/(2*np.abs(mask).max()) for mask in masks]

#############################################################################
# Choose your metric
# ---------------------------------------------
#
# Metric will be used to find the best sampling pattern for a fixed
# reconstruction pattern.

metric = nrmse

#############################################################################
# Reconstruction algorithm
# ---------------------------------------------
#
# We need to fix a CS reconstruction algorithm which will help determine which
# sampling pattern is better. The wrapper around the reconstruction algorithm
# must be as such: recons(image, samples, opt),with image a 2D or 3D np.array,
# samples the locations in [-0.5, 0.5[, and opt a dictionnary with the others
# parameters of the algorithm.

g = recons_pysap
kwargs = {'wavelet_name': "BsplineWaveletTransformATrousAlgorithm",
          'nb_scales': 4,
          'std_est': None,
          'std_est_method': None,
          'std_thr': 2.,
          'mu': 1e-9,
          'tau': None,
          'sigma': None,
          'relaxation_factor': 1.0,
          'nb_of_reweights': 2,
          'max_nb_of_iter': 20,
          'add_positivity': False,
          'atol': 1e-4,
          'non_cartesian': True,
          'uniform_data_shape': shape_img,
          'verbose': 0}

#############################################################################
# Learning-based compressive MRI
# ---------------------------------------------
#
# The algorithm developped by [1] will search the best sampling pattern among
# masks by computing mean of reconstruction metric for a pattern over all
# images of the database

gamma = np.zeros((1, len(masks)))

for (i, mask) in enumerate(masks):
    gamma_temp = 0
    for j in range(X_train.shape[0]):
        image = X_train[j]
        img_recons = g(image, mask, **kwargs)
        m = metric(image, img_recons)
        print('ID Sampling, ID Image, recons metric: ({}, {},'
              '{:.4f})\n'.format(i, j, m))
        gamma_temp = gamma_temp + m

    gamma[i] = gamma_temp/m

i0 = np.argmax(gamma)

print('Best mask is ', samples_paths[i0])

# #############################################################################
# # test
# # -----
# mask = get_sample_data("mri-mask")
# kspace_loc = convert_mask_to_locations(mask.data)
#
# image = X_train[0]
# # print(image)
# # print(image.min(), image.max())
# # exit(0)
# kwargs = {'wavelet_name': "BsplineWaveletTransformATrousAlgorithm",
#           'nb_scales': 4,
#           'std_est': None,
#           'std_est_method': None,
#           'std_thr': 2.,
#           'mu': 1e-9,
#           'tau': None,
#           'sigma': None,
#           'relaxation_factor': 1.0,
#           'nb_of_reweights': 2,
#           'max_nb_of_iter': 20,
#           'add_positivity': False,
#           'atol': 1e-4,
#           'non_cartesian': True,
#           'uniform_data_shape': image.shape,
#           'verbose': 1}
#
# img_rec = g(image, masks[1], **kwargs)
