import sys
import numpy as np
import matplotlib.pyplot as plt
import time

dirpath = '/volatile/bsarthou/datas/'

dct = {}

dct['pwt'] = np.load(dirpath+'save_wavelet3_1_5_1_200.npy').item()
dct['sparse3d'] = np.load(dirpath+'save_wavelet3_1_5_0_200.npy').item()

print('TIME PERFS')

time_pwt = np.mean(dct['pwt']['time'])
time_s3d = np.mean(dct['sparse3d']['time'])

print('time (in s): (Sym4:' + str(time_pwt) + '),(ATrou3D:' + str(time_s3d)
      + '), (diff:' + str(np.abs(time_pwt - time_s3d)) + ')\n')


print('METRICS PERFS')

metric_pwt = np.mean(dct['pwt']['metrics'], axis=1)
metric_s3d = np.mean(dct['sparse3d']['metrics'], axis=1)

print('SSIM: (Bior6:' + str(metric_pwt[0]) + '),(ATrou3D:' + str(metric_s3d[0])
      + '), (diff:' + str(metric_pwt[0] - metric_s3d[0]) + ')\n')

print('SNR: (Bior6:' + str(metric_pwt[1]) + '),(ATrou3D:' + str(metric_s3d[1])
      + '), (diff:' + str(metric_pwt[1] - metric_s3d[1]) + ')\n')

print('PSNR: (Bior6:' + str(metric_pwt[2]) + '),(ATrou3D:' + str(metric_s3d[2])
      + '), (diff:' + str(metric_pwt[2] - metric_s3d[2]) + ')\n')

print('NRMSE:(Bior6:' + str(metric_pwt[3]) + '),(ATrou3D:' + str(metric_s3d[3])
      + '), (diff:' + str(metric_pwt[3] - metric_s3d[3]) + ')\n')


print('COST FUNC')

cost1 = list(dct['pwt']['cost'][0])
cost2 = list(dct['sparse3d']['cost'][0])
min_size = min(len(cost1), len(cost2))

plt.figure()
plt.suptitle('Bior6 Pywavelet vs. Sparse3D ATrou3D\n')
plt.subplot(2, 1, 1)
plt.title('time (in s): {:.1f}/{:.1f}/{:.1f}'.format(time_pwt, time_s3d,

                                                     time_pwt - time_s3d))
plt.plot(cost1, 'r.')
plt.plot(cost2, 'b.')
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
