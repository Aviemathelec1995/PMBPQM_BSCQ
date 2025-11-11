import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
from numpy.linalg import matrix_rank
import copy
import numba
from numba import int64, float64, jit, njit, vectorize
import matplotlib

import matplotlib.pyplot as pl
import argparse as ap

import csv

import bscq_ldpc_threshold as bs

def theta_to_alpha(theta):
    return (1+np.cos(theta))/2

def threshold_alpha(no_samples,k,D_vec,depth,tol):
    th_alpha=[]
    for i in range(len(D_vec)):
        t=bs.binary_search_t(0,0,no_samples,k,D_vec[i],depth,tol)
        th_alpha.append(theta_to_alpha(t))
    return np.array(th_alpha)

def main():
  print(f'Thresholds for regular LDPC codes with varianle node degree {k}, and check node degrees {D_vec}')
  print(f'Number of samples={no_samples}')
  print(f'Depth={depth}')
#   print(f'Number of points for the plot={no_points}')
  print(f'Error tolerance for threshold values={tol}')
  th_alpha=threshold_alpha(no_samples,k,D_vec,depth,tol)
  for i in range(len(D_vec)):
      print(f'Threshold alpha for ({k},{D_vec[i]}) code is {th_alpha[i]}')
  csv_filename = f"threshold_alpha_k{k}_Dvec.csv"
  with open(csv_filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    # header
    writer.writerow(["k", "D", "alpha_threshold"])
    # data rows
    for D_i, alpha_i in zip(D_vec, th_alpha):
       writer.writerow([k, D_i, alpha_i])
    print(f"Threshold alpha values saved to {csv_filename}")

if __name__== "__main__":
  parser = ap.ArgumentParser('Thresholds for regular LDPC codes over BSCQ channels')
  parser.add_argument('--verbose', '-v', help='Display text output', action="store_true")
  parser.add_argument('-ns', dest='no_samples', type=int, default=1000, help='Number of samples for DE')
  parser.add_argument('-k', dest='k', type=int, default=3, help='Bitnode degree')
  parser.add_argument('-D_vec', dest='D_vec', type=int,nargs='+', default=[4,5,6], help='Checknode degree')
  parser.add_argument('-M', dest='depth', type=int, default=60, help='Depth of the tree')
#   parser.add_argument('-np', dest='no_points', type=int, default=5, help='Number of points for the plot')
  parser.add_argument('-err', dest='tol', type=float, default=0.005, help='Error tolerance for threshold values')
  # parse arguments
  args = parser.parse_args()
  locals().update(vars(args))
  main()

