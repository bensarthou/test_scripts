from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.svm import SVR

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colormap = ['r', 'g', 'b', 'y', 'm', 'k', 'c']
marker = ['o', '*', '^']

colormarkers = [(a, b) for a in marker for b in colormap]


def plot_scatter(X, y, suptitle):

    lambdas = np.unique(X[:, 0])

    fig = plt.figure()
    fig.suptitle(suptitle)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('tau')
    ax.set_ylabel('decay')

    for (i, l) in enumerate(lambdas):
        ind = (X[:, 0] == l)

        tau = X[ind, 1]
        decay = X[ind, 2]
        metric = y[ind]

        # print(tau.shape, decay.shape, metric.shape)
        # X, Y = np.meshgrid(tau, decay)
        ax.scatter(tau, decay, metric,
                   color=colormarkers[i][1],
                   marker=colormarkers[i][0])

    plt.show()


X_tot = np.load('/volatile/bsarthou/datas/sparkling/'
                'stat_sparkling_Reg_FLASH3D_2018618_1643.npy')

X_tot, y_tot = X_tot[:, :3], X_tot[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X_tot, y_tot,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)
# generate a model of polynomial features
poly = PolynomialFeatures(degree=2)

# transform the x data for proper fitting (for single variable type it returns,
# [1,x,x**2])
X_train_ = poly.fit_transform(X_train)

# transform the prediction to fit the model type
X_test_ = poly.fit_transform(X_test)

# Here we can remove polynomial orders we don't want
# for instance I'm removing the `x` component
# X_ = np.delete(X_, (1), axis=1)
# X_test_ = np.delete(X_test_, (1) ,axis=1)

# generate the regression object
clf = linear_model.LinearRegression()
# clf = SVR(kernel='linear', C=1e3, verbose=0)
# clf = linear_model.Ridge()
# X_train_, X_test_ = X_train, X_test
# preform the actual regression
clf.fit(X_train_, y_train)

# print("X_ = ", X_)
# print("X_test_ = ", X_test_)
# print("Prediction = ", clf.predict(predict_))
print('Score = ', r2_score(y_test, clf.predict(X_test_)))

# plot_scatter(X_train, y_train, 'train')
# plot_scatter(X_test, y_test, 'test')
# plot_scatter(X_test, clf.predict(X_test_), 'predict')
