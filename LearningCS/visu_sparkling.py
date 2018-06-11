import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

print('LOADING DATA')
directory = '/volatile/bsarthou/datas/sparkling/stat_sparkling_'
path_res = ['1e-08_201869_031.npy', '1e-07_201869_30.npy',
            '1e-06_201869_1115.npy', '1e-05_201869_1936.npy']

list_metric_names = ['snr', 'psnr', 'ssim', 'nrmse']


def load_results(path):
    list_dic = np.load(path)
    list_tau, list_decay = [], []
    res_metrics = [[] for i in range(len(list_metric_names))]

    # Load metrics for each set of sparkling parameters
    for dico in list_dic:
        list_tau.append(dico['tau'])
        list_decay.append(dico['decay'])
        for j in range(len(list_metric_names)):
            res_metrics[j].append(dico['mean_'+list_metric_names[j]])
    return list_tau, list_decay, res_metrics


taus, decays, metricss = [], [], []

for i in range(len(path_res)):
    tau, decay, metrics = load_results(directory + path_res[i])
    taus.append(tau)
    decays.append(decays)
    metricss.append(metrics)
    print('{} out of {} loaded'.format(i, len(path_res)), end="\r")

print('DATA LOADED')

fig = plt.figure()
fig.suptitle('1e-8; 1e-7; 1e-6; 1e-5')
for j in range(len(metricss[0])):
    axJ = fig.add_subplot(220+(j+1), projection='3d')
    axJ.set_title(list_metric_names[j])
    axJ.set_xlabel('tau')
    axJ.set_ylabel('decay')
    for i in range(len(path_res)):
        axJ.plot_trisurf(taus[i], decays[i], metricss[i][j], cmap='magma')

# ax1 = fig.add_subplot(221, projection='3d')
# ax2 = fig.add_subplot(222, projection='3d')
# ax3 = fig.add_subplot(223, projection='3d')
# ax4 = fig.add_subplot(224, projection='3d')
#
# ax1.plot_trisurf(list_tau, list_decay, list_snr, cmap='magma')
# ax1.set_title('snr')
# ax1.set_xlabel('tau')
# ax1.set_ylabel('decay')
#
# ax2.plot_trisurf(list_tau, list_decay, list_psnr, cmap='prism')
# ax2.set_title('psnr')
# ax2.set_xlabel('tau')
# ax2.set_ylabel('decay')
#
# ax3.plot_trisurf(list_tau, list_decay, list_ssim, cmap='rainbow')
# ax3.set_title('ssim')
# ax3.set_xlabel('tau')
# ax3.set_ylabel('decay')
#
# ax4.plot_trisurf(list_tau, list_decay, list_nrmse, cmap='seismic')
# ax4.set_title('nrmse')
# ax4.set_xlabel('tau')
# ax4.set_ylabel('decay')

plt.show()
