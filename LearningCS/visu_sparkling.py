import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import datetime

ABSOLUTE_METRIC_DIFF = False
RELATIVE_METRIC_DIFF = False
MEAN_DIFF_METRIC = False
IMAGES = False
T_D_FIXED_FOR_L = True

id_metrics = 2  # SSIM
# id_metrics = 0  # SNR

nb_decay, nb_tau = 5, 5  # TODO: Remove those hardcoded values
nb_samples = 40


def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)


def load_results(path, image=False, id_metric=2):
    list_dic = np.load(path)
    list_tau, list_decay = [], []
    res_metrics = [[] for i in range(len(list_metric_names))]
    std_metrics = [[] for i in range(len(list_metric_names))]
    min_metrics = [[] for i in range(len(list_metric_names))]
    max_metrics = [[] for i in range(len(list_metric_names))]

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

        stat_bool = True
        for j in range(len(list_metric_names)):
            res_metrics[j].append(dico['mean_'+list_metric_names[j]])
            try:
                std_metrics[j].append(dico['std_'+list_metric_names[j]])
                min_metrics[j].append(dico['min_'+list_metric_names[j]])
                max_metrics[j].append(dico['max_'+list_metric_names[j]])
            except:
                stat_bool = False
                pass

    if not stat_bool:
        print('No metrics statistics availables')
        res_metrics = [res_metrics]
    else:
        res_metrics = [res_metrics, std_metrics, min_metrics, max_metrics]

    if not image:
        return list_tau, list_decay, res_metrics
    else:
        return list_tau, list_decay, res_metrics, best_image


#############################################################################
# LOAD DATA
# ---------
#
# Load all the dictionnaries listed in path_res and extract list of taus,
# decays, metrics for plots

print('LOADING DATA')
directory = '/volatile/bsarthou/datas/sparkling/stat_sparkling_'
#
# path_res = ['1e-08_201869_031.npy', '1e-07_201869_30.npy',
#             '1e-06_201869_1115.npy', '1e-05_201869_1936.npy',
#             '0.0001_2018610_417.npy', '0.1_2018613_950.npy',
#             '0.0_2018613_135.npy', '10.0_2018614_142.npy',
#             '100.0_2018614_1015.npy']

# path_res = ['0.0001_2018610_417.npy', '0.1_2018613_950.npy',
#             '0.0_2018613_135.npy', '10.0_2018614_142.npy',
#             '100.0_2018614_1015.npy']
#
# # One test
# path_res = ['100.0_2018615_1423.npy']

# # From 100 to 2000
# path_res = ['100.0_2018616_154.npy',
#             '200.0_2018616_945.npy',
#             '250.0_2018616_1655.npy',
#             '300.0_2018616_2344.npy',
#             '500.0_2018617_69.npy',
#             '1000.0_2018617_1336.npy',
#             '2000.0_2018617_2154.npy']
#
# # Best lambdas, only 3
# path_res = ['100.0_2018616_154.npy',
#             '200.0_2018616_945.npy',
#             '250.0_2018616_1655.npy']
#
# # All available
path_res = ['1e-08_201869_031.npy',
            '1e-07_201869_30.npy',
            '1e-06_201869_1115.npy',
            '1e-05_201869_1936.npy',
            '0.0001_2018610_417.npy',
            '0.1_2018613_950.npy',
            '0.0_2018613_135.npy',
            '10.0_2018614_142.npy',
            '100.0_2018616_154.npy',
            '200.0_2018616_945.npy',
            '250.0_2018616_1655.npy',
            '300.0_2018616_2344.npy',
            '500.0_2018617_69.npy',
            '1000.0_2018617_1336.npy',
            '2000.0_2018617_2154.npy']

# # Example for slides
# path_res = ['1e-08_201869_031.npy',
#             '0.0_2018613_135.npy',
#             '100.0_2018616_154.npy',
#             '1000.0_2018617_1336.npy']


nb_lambdas = len(path_res)

titles = [float(p.split('_')[0]) for p in path_res]
# WARNING must be same size as path_res
colormap = ['magma', 'seismic', 'rainbow', 'gray', 'plasma']
colormap2 = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
list_metric_names = ['snr', 'psnr', 'ssim', 'nrmse']

taus, decays = [], []
metricss_mean, metricss_std, metricss_min, metricss_max = [], [], [], []

reg_table = np.zeros((nb_lambdas*nb_decay*nb_tau, 4))

for i in range(nb_lambdas):
    tau, decay, metrics = load_results(directory + path_res[i])
    taus.append(tau)
    decays.append(decay)
    metricss_mean.append(metrics[0])

    temp_table = np.zeros((nb_decay*nb_tau, 4))
    temp_table[:, 0] = titles[i]*np.ones((nb_decay*nb_tau,))
    temp_table[:, 1] = tau
    temp_table[:, 2] = decay
    temp_table[:, 3] = metrics[0][id_metrics]

    reg_table[((nb_tau*nb_decay)*i):((nb_tau*nb_decay)*(i+1)), :] = temp_table

    if(len(metrics) == 4):
        metricss_std.append(metrics[1])
        metricss_min.append(metrics[2])
        metricss_max.append(metrics[3])
    else:
        metricss_std.append([[0 for x in metrics[0][j]]
                             for j in range(len(metrics[0]))])
        metricss_min.append(metrics[0])
        metricss_max.append(metrics[0])

    print('{} out of {} loaded'.format(i+1, nb_lambdas), end="\r")

date = datetime.datetime.now()
namefile = 'Reg_FLASH3D_{}{}{}_{}{}.npy'.format(date.year,
                                                date.month,
                                                date.day,  date.hour,
                                                date.minute)

np.save(directory+namefile, reg_table)
# print(reg_table)
print('\n DATA LOADED')
# print(titles)
# print(len(taus))
# print(taus)
# print(len(decays))
# print(decays)
# print(len(metricss_mean))
# print(metricss_mean)


#############################################################################
# PLOT LAMBDA SURFACES
# --------------------
#
# For each lambda (in each dictionary), plot the value of all metrics according
# to tau and decay sparkling parameters. Each surface is a specific lambda

fig = plt.figure()
fig.suptitle('lambdas: ' + str(titles))
for j in range(len(metricss_mean[0])):
    axJ = fig.add_subplot(220+(j+1), projection='3d')
    axJ.set_title(list_metric_names[j])
    axJ.set_xlabel('tau')
    axJ.set_ylabel('decay')
    for i in range(nb_lambdas):
        surf = axJ.plot_trisurf(taus[i], decays[i], metricss_mean[i][j],
                                color=colormap2[i % (len(colormap2))],
                                label='lambda {}'.format(titles[i]))
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
    leg = axJ.legend()
    LH = leg.legendHandles
    LH[i].set_color(colormap2[i % (len(colormap2))])

# plt.show()

#############################################################################
# DIFFERENCE BETWEEN LAMBDAS
# -------------------------
#
# For a specific metric (id specified by id_metrics param), plot the cross
# difference between lambdas of the metric. The subplot (i, j) represents the
# surface difference between lambda_i and lambda_j. Two plots are given, one
# with fixed z axis [-5e-3, 5e-3], the other one with logarithmic axis
# dependent on the subplot


nb_parameters = len(metricss_mean)

if ABSOLUTE_METRIC_DIFF:
    print('Differences of metrics')
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

            axJ.set_title('{} vs {}'.format(titles[i], titles[j]),
                          **{'size': 8})
            axJ.set_xlabel('tau')
            axJ.set_ylabel('decay')
            axJ.plot_trisurf(taus[i], decays[i],
                             np.asarray(metricss_mean[i][id_metrics]) -
                             np.asarray(metricss_mean[j][id_metrics]),
                             cmap=colormap[i % (len(colormap))])
            old_axJ = axJ

if RELATIVE_METRIC_DIFF:
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
                diff_ij = np.asarray(metricss_mean[i][id_metrics])\
                    - np.asarray(metricss_mean[j][id_metrics])
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

#############################################################################
# LAMBDA INFLUENCE ON A METRIC
# ----------------------------
#
# For a specific metric (id specified by id_metrics param), plot the cross
# difference between lambdas of the metric. This time, the value taken for a
# specific lambda is the mean across all (tau, decay) values. In the wireplot,
# point(i,j) represent the absolute difference between mean(SSIM(tau, decay))
# of lambda_i and lambda_j. The graph is obviously symetric

mean_metric = np.asarray([[np.abs(np.mean(np.asarray(
                                    metricss_mean[i][id_metrics]) -
                                          np.asarray(
                                    metricss_mean[j][id_metrics])))
                           for j in range(nb_parameters)]
                          for i in range(nb_parameters)])
if MEAN_DIFF_METRIC:
    print('MEAN DIFFERENCE METRIC')
    fig3 = plt.figure()
    fig3.suptitle('Difference of mean {} between lambdas'.format(
                                        list_metric_names[id_metrics]))
    ax = fig3.add_subplot(111, projection='3d')
    ax.set_xlabel('lambda')
    ax.set_ylabel('lambda')
    new_titles = [(np.log10(k) if k != 0 else 0.0) for k in titles]
    X, Y = np.meshgrid(new_titles, new_titles)
    ax.plot_wireframe(X, Y, mean_metric)

    # plt.show()

#############################################################################
# LAMBDA INFLUENCE ON A METRIC
# ----------------------------
#
# For each lambda, plot the "best reconstructed image" with its associated
# parameter tau and decay
if IMAGES:
    print('IMAGES CHECK')
    best_images = []
    for i in range(nb_lambdas):
        r1, r2, r3, best_image = load_results(directory + path_res[i],
                                              image=True)
        best_images.append(best_image)
        print('{} out of {} loaded'.format(i+1, nb_lambdas), end="\r")

    fig4 = plt.figure()
    fig4.suptitle('Best reconstructed image for several lambdas')

    for i in range(nb_lambdas):
        best_image = best_images[i]
        axI = fig4.add_subplot(np.ceil(nb_lambdas/2.), 2, i+1)
        axI.imshow(np.abs(best_image['data']), cmap='gray')
        axI.set_title('Lambda: {} for (tau, decay): ({},{}). {}={}'.format(
                    titles[i],
                    best_image['tau'],
                    best_image['decay'],
                    list_metric_names[best_image['id_metric']],
                    best_image['best_metric']))
# plt.show()
#############################################################################
# LAMBDA, TAU OR LAMDA, DECAY FIXED
# ---------------------------------
#
# For each lambda, plota metric along tau or decay, with decay or tau
# associated fixed (it's a 2D projection of the surfaces seen before)
if T_D_FIXED_FOR_L:
    print('LAMBDA, TAU OR LAMDA, DECAY FIXED')
    fig5 = plt.figure()
    fig5.suptitle('tau, decay, lambda fixed')

    ax_tau = fig5.add_subplot(1, 2, 1)
    ax_decay = fig5.add_subplot(1, 2, 2, sharey=ax_tau)

    id_fixed_tau = 1
    id_fixed_decay = 1

    ax_tau.set_title('{} along decay,'
                     ' tau fixed at {}'.format(list_metric_names[id_metrics],
                                               taus[0][id_fixed_tau]))

    ax_decay.set_title('{} along tau,'
                       ' decay fixed at {}'.format(list_metric_names[
                        id_metrics], decays[0][nb_lambdas*id_fixed_decay]))

    for i in range(nb_lambdas):
        tab_metrics_mean = np.asarray(
            metricss_mean[i][id_metrics]).reshape((nb_decay, nb_tau))

        ax_tau.plot(decays[i][::nb_tau], tab_metrics_mean[:, id_fixed_tau],
                    color=colormap2[i % (len(colormap2))], label=titles[i])

        ax_decay.plot(taus[i][:nb_decay], tab_metrics_mean[id_fixed_decay, :],
                      color=colormap2[i % (len(colormap2))], label=titles[i])

        if (metricss_std != []):
            tab_metrics_std = np.asarray(
                metricss_std[i][id_metrics]).reshape((nb_decay, nb_tau))
            tab_metrics_min = np.asarray(
                metricss_min[i][id_metrics]).reshape((nb_decay, nb_tau))
            tab_metrics_max = np.asarray(
                metricss_max[i][id_metrics]).reshape((nb_decay, nb_tau))

            ax_tau.plot(decays[i][::nb_tau],
                        tab_metrics_mean[:, id_fixed_tau]
                        - tab_metrics_std[:, id_fixed_tau]/np.sqrt(nb_samples),
                        color=colormap2[i % (len(colormap2))],
                        label=str(titles[i])+' std-',
                        linestyle='--')
            ax_tau.plot(decays[i][::nb_tau],
                        tab_metrics_mean[:, id_fixed_tau]
                        + tab_metrics_std[:, id_fixed_tau]/np.sqrt(nb_samples),
                        color=colormap2[i % (len(colormap2))],
                        label=str(titles[i])+' std+',
                        linestyle='--')
            ax_tau.plot(decays[i][::nb_tau], tab_metrics_min[:, id_fixed_tau],
                        color=colormap2[i % (len(colormap2))],
                        label=str(titles[i])+' min',
                        linestyle=':')
            ax_tau.plot(decays[i][::nb_tau], tab_metrics_max[:, id_fixed_tau],
                        color=colormap2[i % (len(colormap2))],
                        label=str(titles[i])+' max',
                        linestyle=':')

            ax_decay.plot(taus[i][:nb_decay],
                          tab_metrics_mean[id_fixed_decay, :]
                          - tab_metrics_std[id_fixed_decay,
                                            :]/np.sqrt(nb_samples),
                          color=colormap2[i % (len(colormap2))],
                          label=str(titles[i])+' std-',
                          linestyle='--')
            ax_decay.plot(taus[i][:nb_decay],
                          tab_metrics_mean[id_fixed_decay, :]
                          + tab_metrics_std[id_fixed_decay,
                                            :]/np.sqrt(nb_samples),
                          color=colormap2[i % (len(colormap2))],
                          label=str(titles[i])+' std+',
                          linestyle='--')
            ax_decay.plot(taus[i][:nb_decay],
                          tab_metrics_min[id_fixed_decay, :],
                          color=colormap2[i % (len(colormap2))],
                          label=str(titles[i])+' min',
                          linestyle=':')
            ax_decay.plot(taus[i][:nb_decay],
                          tab_metrics_max[id_fixed_decay, :],
                          color=colormap2[i % (len(colormap2))],
                          label=str(titles[i])+' max',
                          linestyle=':')

    ax_tau.legend()
    ax_decay.legend()
    ax_tau.set_xlabel('decay')
    ax_tau.set_ylabel(list_metric_names[id_metrics])
    ax_decay.set_xlabel('tau')
    ax_decay.set_ylabel(list_metric_names[id_metrics])


#############################################################################
# TAU, DECAY FIXED
# ----------------
#
# For each lambda, plota metric along tau or decay, with decay or tau
# associated fixed (it's a 2D projection of the surfaces seen before)
# if L_FIXED_FOR_T_D:

plt.show()
