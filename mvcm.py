# mkdir /pine/scr/y/s/yshan/p3/data/sim/dat/res_uneq_h3
# cd /pine/scr/y/s/yshan/p3/code
import numpy as np
from numpy.linalg import inv
from mvcm_beta_20210131 import mvcm_beta
p = '/pine/scr/y/s/yshan/p3/data/sim/dat/'
n = 100
n_rep = 5000
coord = np.loadtxt(p+'coord.txt')[:,np.newaxis] # (50,)
hi = 3
h = 0.07

x0 = np.loadtxt(p+'x/x0.txt')          # (400, 3)
x = np.zeros((n*4, 3))
for k in range(4):
  x[k*n:(k+1)*n,:] = np.loadtxt(p+'x/x'+str(k+1)+'.txt')         # (100, 3)

case = 'uneq'

for v in range(10):
  print(v)
  MSPE = np.zeros((n_rep,2)) # 2 columns: MSPE_m MSPE_e
  if v==0:
    py = p+'y_v0/k'
  else:
    py = p+'y_'+case+'_h'+str(hi)+'/v'+str(v)+'_k'
  
  for r in range(n_rep):
    beta = np.zeros((4,3,50))
    y = np.zeros((n*4, 50))
    for k in range(4):
      y[k*n:(k+1)*n,:] = np.loadtxt(py+str(k+1)+'.'+str(r+1)) # (100, 50)
      xk = x[k*n:(k+1)*n,:]
      yk = y[k*n:(k+1)*n,:]
      beta[k,:,:] = mvcm_beta(xk, yk, coord,h)
    
    beta_m = mvcm_beta(x,y,coord,h)
    beta_e = np.sum(beta, axis=0)/4
    y0 = np.loadtxt(py+'0.'+str(r+1)) # (400, 50)
    MSPE[r,0] = np.sum((y0-np.dot(x0, beta_m))**2) # MSPE_m
    MSPE[r,1] = np.sum((y0-np.dot(x0, beta_e))**2) # MSPE_e
  
  np.savetxt(p+'res_'+case+'_h'+str(hi)+'/v'+str(v)+'.MSPE_mVe', MSPE)
  print(np.sum(MSPE[:,0] < MSPE[:,1])) # how many (out of 1000) replicates give merging better
  d = np.log(MSPE[:,1] / MSPE[:,0])    # log(MSPE_e / MSPE_m): + merge good, - ensemble good
  print(np.round([np.mean(d), np.max(d), np.min(d)],5))


# 0
# 3607
# [ 0.00298  0.03979 -0.01399]
# 1
# 3112
# [ 0.00197  0.04278 -0.02073]
# 2
# 2834
# [ 0.00121  0.03724 -0.04811]
# 3
# 2701
# [ 0.00039  0.04108 -0.05728]
# 4
# 2609
# [-0.00017  0.05538 -0.08439]
# 5
# 2554
# [-0.00055  0.05564 -0.098  ]
# 6
# 2483
# [-0.00139  0.07148 -0.09447]
# 7
# 2415
# [-0.00251  0.06676 -0.13943]
# 8
# 2384
# [-0.00264  0.07806 -0.16092]
# 9
# 2375
# [-0.00281  0.09627 -0.19126]
