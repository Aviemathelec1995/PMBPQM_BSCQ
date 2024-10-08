import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
from numpy.linalg import matrix_rank
import copy
import numba
from numba import int64, float64, jit, njit, vectorize
import matplotlib.pyplot as pl

import bscq_ldpc_threshold as bs


#define parameters
theta=0.59
p=0
no_samples=10000
dv=5
dc=6
depth=100

H=bs.helstrom_channel_density(theta,p,no_samples,dv,dc,depth)


if min(H)==0:
  print('parameters lie below threshold')
else:
  for i in range(depth-1):
    print(H[i],'iter:',i+1)