"""
Local linear kernel smoothing on beta in MVCM.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-18
"""

import numpy as np
from numpy.linalg import inv
import statsmodels.nonparametric.api as nparam

"""
installed all the libraries above
"""


def mvcm_beta(x, y, coord_data, bw0='cv_ls'):
    """
        Local linear kernel smoothing on beta in MVCM.

        :param
            x (matrix): covariate data (design matrix, n*p)
            y (matrix): imaging response data (response matrix, n*l*m)
            coord_data (matrix): common coordinate matrix (l*d)
            bw0 (vector): pre-defined optimal bandwidth
        :return
            beta (matrix): estimated functional coefficient (p*l*m)
            bw_o (matrix): optimal bandwidth (d*m)

    """

    # Set up
    n, l, m = y.shape
    d = coord_data.shape[1]
    p = x.shape[1]

    sm_y = y * 0
    res_y = y * 0
    beta = np.zeros(shape=(p, l, m))
    bw_o = np.zeros(shape=(d, m))

    c_mat = np.dot(inv(np.dot(x.T, x) + np.eye(p) * 0.00001), x.T)

    if d == 1:
        var_tp = 'c'
    elif d == 2:
        var_tp = 'cc'
    else:
        var_tp = 'ccc'

    for mii in range(m):
        y_avg = np.mean(y[:, :, mii], axis=0)
        if isinstance(bw0, str) == 0:
            bw_opt = bw0[:, mii]
        else:
            model_bw = nparam.KernelReg(endog=[y_avg], exog=[coord_data], var_type=var_tp, bw='cv_ls')
            bw_opt = model_bw.bw
        print("The optimal bandwidth for the " + str(mii+1) + "-th functional measurement is")
        print(bw_opt)
        bw_o[:, mii] = bw_opt
        for nii in range(n):
            y_ii = np.reshape(y[nii, :, mii], newshape=y_avg.shape)
            model_y = nparam.KernelReg(endog=[y_ii], exog=[coord_data], var_type=var_tp, bw=bw_opt)
            sm_y[nii, :, mii] = model_y.fit()[0]
        beta[:, :, mii] = np.dot(c_mat, sm_y[:, :, mii])

    return beta, bw_o
