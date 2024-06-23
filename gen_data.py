"""
Step 1:
Generate simulation data
"""

import os
import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import norm, uniform
from scipy.io import savemat

"""
installed all the libraries above
"""

""" initialization """
simu = 5  # number of simulation datasets
l_grid = 50   # size for coordinates
nk = 500  # sample size for study k data (k=1,2)
n0 = 1000  # sample size for testing data
m = 2  # number of measurements
p = 3  # number of observed effects
phase = 'uneq'  # case setting up: "uneq" or "eq"
np.random.seed(123)

# generate coordinate data
coord_data = uniform.rvs(size=l_grid)
coord_data.sort()
coord_mat = np.dot(coord_data.reshape(l_grid, 1), np.ones(shape=(1, l_grid)))
# coord_dict = np.absolute(coord_mat-np.transpose(coord_mat))

# set up the observed effect size
beta11 = np.reshape(3*coord_data**2, newshape=(1, l_grid))
beta12 = np.reshape(2*(1-coord_data)**2, newshape=(1, l_grid))
beta13 = np.reshape(4*coord_data*(1-coord_data), newshape=(1, l_grid))
beta1 = np.vstack((beta11, beta12, beta13))

beta21 = np.reshape(5*(coord_data-0.5)**2, newshape=(1, l_grid))
beta22 = np.reshape(3*coord_data**0.5, newshape=(1, l_grid))
beta23 = np.reshape(4*coord_data*(1-coord_data), newshape=(1, l_grid))
beta2 = np.vstack((beta21, beta22, beta23))

beta0 = np.zeros(shape=(p, l_grid, 2))
beta0[:, :, 0] = beta1
beta0[:, :, 1] = beta2

# set up the basis functions & fPCA
eta11 = np.reshape(np.sqrt(2)*np.sin(2*np.pi*coord_data), newshape=(1, l_grid))
eta12 = np.reshape(np.sqrt(2)*np.cos(2*np.pi*coord_data), newshape=(1, l_grid))
eta21 = eta12
eta22 = eta11
lambda11 = 0.5
lambda12 = 0.1
lambda21 = 0.5
lambda22 = 0.1
sigma2 = 0.1

# set up the random effect
rho = 0.5
s2_gamma = 0.025
if phase == 'eq':
    s2_gamma_1 = s2_gamma
    s2_gamma_2 = s2_gamma
else:
    s2_gamma_1 = 2/3*s2_gamma
    s2_gamma_2 = 4/3*s2_gamma
cov_gamma_1 = s2_gamma_1 * rho ** np.absolute(coord_mat-np.transpose(coord_mat))
cov_gamma_2 = s2_gamma_2 * rho ** np.absolute(coord_mat - np.transpose(coord_mat))

for k in range(simu):
    # create folder
    idx = k+1
    data_folder = './simu_dat/%s/%d' % (phase, idx)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    beta_file_name = '%s/beta0.mat' % data_folder
    savemat(beta_file_name, mdict={'beta0': beta0})
    coord_file_name = '%s/coord_data.mat' % data_folder
    savemat(coord_file_name, mdict={'coord_data': coord_data})

    # generate covariate data for study 1
    x1 = np.hstack((np.ones(shape=(nk, 1)), multivariate_normal([0, 0], [[1, 0.25], [0.25, 1]], (nk,))))
    x1_file_name = '%s/x1.mat' % data_folder
    savemat(x1_file_name, mdict={'x1': x1})
    # generate response data for study 1
    y1 = np.zeros(shape=(nk, l_grid, m))
    z1 = x1[:, 1:3]
    gamma1 = multivariate_normal(np.zeros(shape=(l_grid,)), cov_gamma_1, (2,))
    gamma2 = multivariate_normal(np.zeros(shape=(l_grid,)), cov_gamma_2, (2,))
    xi11 = lambda11 ** 0.5 * norm.rvs(size=(nk, 1))
    xi12 = lambda12 ** 0.5 * norm.rvs(size=(nk, 1))
    eta_1 = np.dot(xi11, eta11) + np.dot(xi12, eta12)
    err_1 = sigma2 ** 0.5 * norm.rvs(size=(nk, l_grid))
    y1[:, :, 0] = np.dot(x1, beta1) + np.dot(z1, np.vstack((gamma1[0, :], gamma2[0, :]))) + eta_1 + err_1
    xi21 = lambda21 ** 0.5 * norm.rvs(size=(nk, 1))
    xi22 = lambda22 ** 0.5 * norm.rvs(size=(nk, 1))
    eta_2 = np.dot(xi21, eta21) + np.dot(xi22, eta22)
    err_2 = sigma2 ** 0.5 * norm.rvs(size=(nk, l_grid))
    y1[:, :, 1] = np.dot(x1, beta2) + np.dot(z1, np.vstack((gamma1[1, :], gamma2[1, :]))) + eta_2 + err_2
    y_file_name = '%s/y1.mat' % data_folder
    savemat(y_file_name, mdict={'y1': y1})

    # generate covariate data for study 2
    x2 = np.hstack((np.ones(shape=(nk, 1)), multivariate_normal([0, 0], [[1, -0.25], [-0.25, 1]], (nk,))))
    x2_file_name = '%s/x2.mat' % data_folder
    savemat(x2_file_name, mdict={'x2': x2})
    # generate response data for study 2
    y2 = np.zeros(shape=(nk, l_grid, m))
    z2 = x2[:, 1:3]
    gamma1 = multivariate_normal(np.zeros(shape=(l_grid,)), cov_gamma_1, (2,))
    gamma2 = multivariate_normal(np.zeros(shape=(l_grid,)), cov_gamma_2, (2,))
    xi11 = lambda11 ** 0.5 * norm.rvs(size=(nk, 1))
    xi12 = lambda12 ** 0.5 * norm.rvs(size=(nk, 1))
    eta_1 = np.dot(xi11, eta11) + np.dot(xi12, eta12)
    err_1 = sigma2 ** 0.5 * norm.rvs(size=(nk, l_grid))
    y2[:, :, 0] = np.dot(x2, beta1) + np.dot(z2, np.vstack((gamma1[0, :], gamma2[0, :]))) + eta_1 + err_1
    xi21 = lambda21 ** 0.5 * norm.rvs(size=(nk, 1))
    xi22 = lambda22 ** 0.5 * norm.rvs(size=(nk, 1))
    eta_2 = np.dot(xi21, eta21) + np.dot(xi22, eta22)
    err_2 = sigma2 ** 0.5 * norm.rvs(size=(nk, l_grid))
    y2[:, :, 1] = np.dot(x2, beta2) + np.dot(z2, np.vstack((gamma1[1, :], gamma2[1, :]))) + eta_2 + err_2
    y_file_name = '%s/y2.mat' % data_folder
    savemat(y_file_name, mdict={'y2': y2})

    # generate covariate data for testing data
    x01 = np.hstack((np.ones(shape=(nk, 1)), multivariate_normal([0, 0], [[1, 0.25], [0.25, 1]], (nk,))))
    x02 = np.hstack((np.ones(shape=(nk, 1)), multivariate_normal([0, 0], [[1, -0.25], [-0.25, 1]], (nk,))))
    x0 = np.vstack((x01, x02))
    x0_file_name = '%s/x0.mat' % data_folder
    savemat(x0_file_name, mdict={'x0': x0})
    # generate response data for testing data
    y0 = np.zeros(shape=(n0, l_grid, m))
    z0 = x0[:, 1:3]
    gamma1 = multivariate_normal(np.zeros(shape=(l_grid,)), cov_gamma_1, (2,))
    gamma2 = multivariate_normal(np.zeros(shape=(l_grid,)), cov_gamma_2, (2,))
    xi11 = lambda11 ** 0.5 * norm.rvs(size=(n0, 1))
    xi12 = lambda12 ** 0.5 * norm.rvs(size=(n0, 1))
    eta_1 = np.dot(xi11, eta11) + np.dot(xi12, eta12)
    err_1 = sigma2 ** 0.5 * norm.rvs(size=(n0, l_grid))
    y0[:, :, 0] = np.dot(x0, beta1) + np.dot(z0, np.vstack((gamma1[0, :], gamma2[0, :]))) + eta_1 + err_1
    xi21 = lambda21 ** 0.5 * norm.rvs(size=(n0, 1))
    xi22 = lambda22 ** 0.5 * norm.rvs(size=(n0, 1))
    eta_2 = np.dot(xi21, eta21) + np.dot(xi22, eta22)
    err_2 = sigma2 ** 0.5 * norm.rvs(size=(n0, l_grid))
    y0[:, :, 1] = np.dot(x0, beta2) + np.dot(z0, np.vstack((gamma1[1, :], gamma2[1, :]))) + eta_2 + err_2
    y_file_name = '%s/y0.mat' % data_folder
    savemat(y_file_name, mdict={'y0': y0})
