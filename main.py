"""
Step 2:
Calculate two learners and MSPE on simulation data
"""

import numpy as np
import time
from scipy.io import loadmat, savemat
from mvcm import mvcm_beta

"""
installed all the libraries above
"""

""" initialization """
simu = 5  # number of simulation datasets
l_grid = 50  # size for coordinates
nk = 500  # sample size for study k data (k=1,2)
n0 = 1000  # sample size for testing data
m = 2  # number of measurements
p = 3  # number of observed effects
h = 0.07  # pre-specified bandwidth if needed
phase = 'uneq'  # case setting up: "uneq" or "eq"
result_folder = './simu_result/%s' % phase
np.random.seed(123)


MSPE = np.zeros((simu, 2))  # 2 columns: MSPE_m MSPE_e

for k in range(simu):
    idx = k + 1
    """+++++++++++++++++++++++++++++++++++"""
    start_0 = time.time()
    print('run the analysis on the dataset %d' % idx)
    data_folder = './simu_dat/%s/%d' % (phase, idx)
    beta_file_name = '%s/beta0.mat' % data_folder
    beta0 = loadmat(beta_file_name)['beta0']
    coord_file_name = '%s/coord_data.mat' % data_folder
    coord_data = loadmat(coord_file_name)['coord_data']
    coord = np.reshape(coord_data, (l_grid, 1))
    x1_file_name = '%s/x1.mat' % data_folder
    x1 = loadmat(x1_file_name)['x1']
    y1_file_name = '%s/y1.mat' % data_folder
    y1 = loadmat(y1_file_name)['y1']
    x2_file_name = '%s/x2.mat' % data_folder
    x2 = loadmat(x2_file_name)['x2']
    y2_file_name = '%s/y2.mat' % data_folder
    y2 = loadmat(y2_file_name)['y2']

    # merging learner
    print('train the merging learner')
    x_m = np.vstack((x1, x2))
    y_m = np.zeros((n0, l_grid, m))
    y_m[:, :, 0] = np.vstack((y1[:, :, 0], y2[:, :, 0]))
    y_m[:, :, 1] = np.vstack((y1[:, :, 1], y2[:, :, 1]))
    # bw = np.reshape([h, h], (1, 2))
    beta_m, bw_o = mvcm_beta(x_m, y_m, coord)
    beta_m_file_name = '%s/beta_m.mat' % data_folder
    savemat(beta_m_file_name, mdict={'beta_m': beta_m})

    # ensembling learner
    # beta = np.zeros((p, l_grid, m, 2))
    print('train the ensembling learner on study 1')
    beta1, _ = mvcm_beta(x1, y1, coord, bw0=bw_o)
    print('train the ensembling learner on study 2')
    beta2, _ = mvcm_beta(x2, y2, coord, bw0=bw_o)
    w1 = 0.5   # ensembling weights
    beta_e = w1*beta1 + (1-w1)*beta2
    beta_e_file_name = '%s/beta_e.mat' % data_folder
    savemat(beta_e_file_name, mdict={'beta_e': beta_e})

    # loading testing data
    x0_file_name = '%s/x0.mat' % data_folder
    x0 = loadmat(x0_file_name)['x0']
    y0_file_name = '%s/y0.mat' % data_folder
    y0 = loadmat(y0_file_name)['y0']
    MSPE[k, 0] = np.sum((y0[:, :, 0] - np.dot(x0, beta_m[:, :, 0])) ** 2) + np.sum((y0[:, :, 1] - np.dot(x0, beta_m[:, :, 1])) ** 2)  # MSPE_m
    MSPE[k, 1] = np.sum((y0[:, :, 0] - np.dot(x0, beta_e[:, :, 0])) ** 2) + np.sum((y0[:, :, 1] - np.dot(x0, beta_e[:, :, 1])) ** 2)  # MSPE_e

    end_0 = time.time()

    print("------------------------------- \n MSPE based on two learners are finished !\n ---------------------------")

    print("Elapsed time is " + str(end_0 - start_0))

np.savetxt(result_folder + 'MSPE.txt', MSPE)
print(np.sum(MSPE[:, 0] > MSPE[:, 1]))  # how many (out of 1000) replicates indicating the ensembling learner better
d = np.log(MSPE[:, 1] / MSPE[:, 0])  # log(MSPE_e / MSPE_m): + merge good, - ensemble good
print(np.round([np.mean(d), np.max(d), np.min(d)], 5))


