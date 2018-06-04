import sys
import numpy as np
import matplotlib.pyplot as plt
import time

dirpath = '/volatile/bsarthou/datas/'

print('-----------------------------------\n')
print('---Comparison Pywt vs. Sparse3D----')
print('-----------------------------------\n')


dct = {}
name_pwt = 'bior3.3'
name_sp3d = 'Haar'
dct['pwt'] = np.load(dirpath+'save_wavelet3_'+name_pwt+'_1_5_1_200.npy').item()
dct['sparse3d'] = np.load(dirpath +
                          'save_wavelet3_'+name_sp3d+'_1_5_0_200.npy').item()

print('TIME PERFS')

time_pwt = np.mean(dct['pwt']['time'])
time_s3d = np.mean(dct['sparse3d']['time'])

print('METRICS PERFS')

metric_pwt = np.mean(dct['pwt']['metrics'], axis=1)
metric_s3d = np.mean(dct['sparse3d']['metrics'], axis=1)

# print('SSIM: (Bior6:' + str(metric_pwt[0]) + '),(ATrou3D:' + str(metric_s3d[0])
#       + '), (diff:' + str(metric_pwt[0] - metric_s3d[0]) + ')\n')
#
# print('SNR: (Bior6:' + str(metric_pwt[1]) + '),(ATrou3D:' + str(metric_s3d[1])
#       + '), (diff:' + str(metric_pwt[1] - metric_s3d[1]) + ')\n')
#
# print('PSNR: (Bior6:' + str(metric_pwt[2]) + '),(ATrou3D:' + str(metric_s3d[2])
#       + '), (diff:' + str(metric_pwt[2] - metric_s3d[2]) + ')\n')
#
# print('NRMSE:(Bior6:' + str(metric_pwt[3]) + '),(ATrou3D:' + str(metric_s3d[3])
#       + '), (diff:' + str(metric_pwt[3] - metric_s3d[3]) + ')\n')


print('COST FUNC')

cost1 = list(dct['pwt']['cost'][0])
cost2 = list(dct['sparse3d']['cost'][0])
min_size = min(len(cost1), len(cost2))

plt.figure()
plt.suptitle('CONDAT: Pywavelet {} vs. Sparse3D {}\n'.format(
                                                        name_pwt, name_sp3d))
plt.subplot(2, 1, 1)
plt.title('time (in s): {:.1f}/{:.1f}/{:.1f}'.format(time_pwt, time_s3d,

                                                     time_pwt - time_s3d))
plt.plot(cost1, 'r')
plt.plot(cost2, 'b')
plt.subplot(2, 1, 2)
plt.title('SSIM:SNR:PSNR:NRMSE = ({:.3f}/{:.3f}/{:.3E}),({:.3f}/{:.3f}/{:.3E})'
          ',({:.3f}/{:.3f}/{:.3E}),'
          ' ({:.3f}/{:.3f}/{:.3E})'.format(metric_pwt[0], metric_s3d[0],
                                           metric_pwt[0] - metric_s3d[0],
                                           metric_pwt[1], metric_s3d[1],
                                           metric_pwt[1] - metric_s3d[1],
                                           metric_pwt[2], metric_s3d[2],
                                           metric_pwt[2] - metric_s3d[2],
                                           metric_pwt[3], metric_s3d[3],
                                           metric_pwt[3] - metric_s3d[3]))

plt.plot(list(range(min(len(cost1), len(cost2)))),
         [cost1[i] - cost2[i] for i in range(min_size)])
plt.show()

print('---------------------------------------\n')
print('---Find best among list of wavelets----')
print('---------------------------------------\n')

# list_wt = ['bior2.4', 'bior3.3', 'bior6.8', 'db4', 'haar']
list_wt = ['BiOr35', 'BiOr44', 'Mallat79', 'Daubechies4', 'Haar', 'Atrou3D']
print('List of wavelets: ', list_wt)

best_time_name = ''
best_time = None
# chosen_metrics = ('SSIM', 0)
# chosen_metrics = ('SNR', 1)
chosen_metrics = ('PSNR', 2)
# chosen_metrics = ('NRMSE', 3)

best_metric_name = ''
best_metric = None

for (i, wt) in enumerate(list_wt):
    dct = np.load(dirpath+'save_wavelet3_'+wt+'_1_5_0_200.npy').item()
    time = np.mean(dct['time'])

    if(best_time is None or time < best_time):
        best_time = time
        best_time_name = wt

    metric_tab = np.mean(dct['metrics'], axis=1)
    metric = metric_tab[chosen_metrics[1]]

    if(best_metric is None or metric > best_metric):
        best_metric = metric
        best_metric_name = wt

print('Best {}:{} for {} transform'.format(chosen_metrics[0],
                                           best_metric, best_metric_name))

print('Best time:{} for {} transform'.format(best_time, best_time_name))
