import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker


def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)


def load_results(path, image=False, id_metric=2):
    list_dic = np.load(path)
    list_tau, list_decay = [], []
    res_metrics = [[] for i in range(len(list_metric_names))]

    # Load metrics for each set of sparkling parameters
    best_image = {'dic': {}, 'id_metric': id_metric, 'best_metric': 0.0,
                  'tau': 0.0, 'decay': 0.0}

    for dico in list_dic:
        list_tau.append(dico['tau'])
        list_decay.append(dico['decay'])

        # Check for the best reconstructed image in the list of dicts,
        # and its associated params.
        if (image and (dico['mean_'+list_metric_names[id_metric]] >
                       best_image['best_metric'])):

            best_image['best_metric'] = dico['mean_' +
                                             list_metric_names[id_metric]]
            best_image['tau'] = dico['tau']
            best_image['decay'] = dico['decay']
            best_image['data'] = dico['img_recons']

        for j in range(len(list_metric_names)):
            res_metrics[j].append(dico['mean_'+list_metric_names[j]])

    if not image:
        return list_tau, list_decay, res_metrics
    else:
        return list_tau, list_decay, res_metrics, best_image


print('LOADING DATA')
directory = '/volatile/bsarthou/datas/sparkling/stat_sparkling_'
#
# path_res = ['1e-08_201869_031.npy', '1e-07_201869_30.npy',
#             '1e-06_201869_1115.npy', '1e-05_201869_1936.npy',
#             '0.0001_2018610_417.npy', '0.1_2018613_950.npy',
#             '0.0_2018613_135.npy', '10.0_2018614_142.npy',
#             '100.0_2018614_1015.npy']

# path_res = ['0.0_2018613_135.npy']
path_res = ['0.0001_2018610_417.npy', '0.1_2018613_950.npy',
            '0.0_2018613_135.npy', '10.0_2018614_142.npy',
            '100.0_2018614_1015.npy']

titles = [float(p.split('_')[0]) for p in path_res]
# WARNING must be same size as path_res
colormap = ['magma', 'seismic', 'rainbow', 'gray', 'plasma', 'magma', 'plasma',
            'seismic', 'rainbow']
colormap2 = ['k', 'b', 'r', 'g', 'y', 'm', 'k', 'b', 'r']
list_metric_names = ['snr', 'psnr', 'ssim', 'nrmse']

taus, decays, metricss = [], [], []

for i in range(len(path_res)):
    tau, decay, metrics = load_results(directory + path_res[i])
    taus.append(tau)
    decays.append(decay)
    metricss.append(metrics)
    print('{} out of {} loaded'.format(i+1, len(path_res)), end="\r")

print('\n DATA LOADED')

fig = plt.figure()
fig.suptitle(titles)
for j in range(len(metricss[0])):
    axJ = fig.add_subplot(220+(j+1), projection='3d')
    axJ.set_title(list_metric_names[j])
    axJ.set_xlabel('tau')
    axJ.set_ylabel('decay')
    for i in range(len(path_res)):
        surf = axJ.plot_trisurf(taus[i], decays[i], metricss[i][j],
                                color=colormap2[i],
                                label='lambda {}'.format(titles[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    leg = axJ.legend()
    LH = leg.legendHandles
    LH[i].set_color(colormap2[i])

# plt.show()

print('Differences of metrics')

nb_parameters = len(metricss)
id_metrics = 2

fig = plt.figure()
fig.suptitle('Absolute difference of ' + list_metric_names[id_metrics])

for i in range(nb_parameters):
    for j in range(nb_parameters):

        if(i == 0 and j == 0):
            axJ = fig.add_subplot(nb_parameters, nb_parameters,
                                  (i+1)+j*nb_parameters,
                                  projection='3d')
            axJ.set_zlim(-5e-3, 5e-3)
        else:
            axJ = fig.add_subplot(nb_parameters, nb_parameters,
                                  (i+1)+j*nb_parameters,
                                  projection='3d',
                                  sharez=old_axJ)

        axJ.set_title('{} vs {}'.format(titles[i], titles[j]), **{'size': 8})
        axJ.set_xlabel('tau')
        axJ.set_ylabel('decay')
        axJ.plot_trisurf(taus[i], decays[i],
                         np.asarray(metricss[i][id_metrics]) -
                         np.asarray(metricss[j][id_metrics]),
                         cmap=colormap[i])
        old_axJ = axJ


fig2 = plt.figure()
fig2.suptitle('Relative difference of ' + list_metric_names[id_metrics])

for i in range(nb_parameters):
    for j in range(nb_parameters):
        if j != i:
            axJ = fig2.add_subplot(nb_parameters, nb_parameters,
                                   (i+1)+j*nb_parameters,
                                   projection='3d')
            axJ.zaxis.set_major_formatter(mticker.FuncFormatter(
                                            log_tick_formatter))
            diff_ij = np.asarray(metricss[i][id_metrics])\
                - np.asarray(metricss[j][id_metrics])
            axJ.set_title('{} vs {}: {:.2E}'.format(titles[i], titles[j],
                                                    np.mean(diff_ij)),
                          **{'size': 8})
            axJ.set_xlabel('tau')
            axJ.set_ylabel('decay')
            axJ.plot_trisurf(taus[i], decays[i],
                             np.log10(np.abs(diff_ij)),
                             cmap='magma')

# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()

print('MEAN DIFFERENCE METRIC')
mean_metric = np.asarray([[np.abs(np.mean(np.asarray(metricss[i][id_metrics]) -
                                          np.asarray(metricss[j][id_metrics])))
                           for j in range(nb_parameters)]
                          for i in range(nb_parameters)])

fig3 = plt.figure()
fig3.suptitle('Difference of mean {} between lambdas'.format(
                                                list_metric_names[id_metrics]))
ax = fig3.add_subplot(111, projection='3d')
ax.set_xlabel('lambda')
ax.set_ylabel('lambda')
new_titles = [(np.log10(k) if k != 0 else 0.0) for k in titles]
X, Y = np.meshgrid(new_titles, new_titles)
ax.plot_wireframe(X, Y, mean_metric)

# ax.plot_trisurf(titles, titles,
#                 mean_metric.flatten(),
#                 cmap='magma')

# plt.show()

print('IMAGES')
# Idea: For each lambda, find best tau, decay for a metric and plot the
# reconstructed image

best_images = []
for i in range(len(path_res)):
    r1, r2, r3, best_image = load_results(directory + path_res[i], image=True)
    best_images.append(best_image)
    print('{} out of {} loaded'.format(i+1, len(path_res)), end="\r")

fig4 = plt.figure()
fig4.suptitle('Best reconstructed image for several lambdas')

for i in range(len(path_res)):
    best_image = best_images[i]
    axI = fig4.add_subplot(len(path_res), 1, i+1)
    axI.imshow(np.abs(best_image['data']), cmap='gray')
    # axI.title('Lambda: {} for (tau, decay): ({},{}). {}={}'.format(
    #             titles[i],
    #             best_image['tau'],
    #             best_image['decay'],
    #             list_metric_names[best_image['id_metric']],
    #             best_image['best_metric']))

plt.show()
