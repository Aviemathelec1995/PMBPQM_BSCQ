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
from tqdm import tqdm

import bscq_ldpc_threshold as bs
def plot1(p,t,ph,th):
  pl.plot(p,t,linestyle='-',marker='>',markersize=7.5,label='PMBPQM')
  pl.plot(ph,th,linestyle='-',marker='D',markersize=7.5,label='Holevo Bound')
  pl.xticks(fontsize=18)
  pl.yticks(fontsize=18)
  pl.grid(which='major',linestyle='--',linewidth=1)
  pl.grid(which='minor', linestyle='--', linewidth='0.5')
  pl.minorticks_on()
  pl.xlabel(r'$p$' ,fontsize=23)
  pl.ylabel(r'$\theta$',fontsize=23)
  pl.legend(fontsize=10)
  pl.savefig("ldpc_bscq_threshold_plot.png",bbox_inches='tight')
  pl.show()

def plot2(p,t,ph,th):
  t1=np.pi/2-np.array(t)
  t1=np.flip(t1)
  p=np.flip(np.array(p))

  th1=np.pi/2-np.array(th)
  th1=np.flip(th1)
  ph=np.flip(np.array(ph))

  pl.plot(t1,p,linestyle='-',marker='>',markersize=7.5,label='PMBPQM')
  pl.plot(th1,ph,linestyle='-',marker='>',markersize=7.5,label='Holevo Bound')
  pl.xticks(fontsize=18)
  pl.yticks(fontsize=18)
  pl.grid(which='major',linestyle='--',linewidth=1)
  pl.grid(which='minor', linestyle='--', linewidth='0.5')
  pl.minorticks_on()
  pl.xlabel(r'$\frac{\pi}{2}-\theta$' ,fontsize=23)
  pl.ylabel(r'$p$',fontsize=23)
  pl.legend(fontsize=10)
  pl.savefig("ldpc_bscq_threshold_plot.png",bbox_inches='tight')
  pl.show()
  
def main():
  print(f'Thresholds for ({int(dv)},{int(dc)}) regular LDPC codes over BSCQ channels')
  print(f'Number of samples={no_samples}')
  print(f'Depth={depth}')
  print(f'Number of points for the plot={no_points}')
  print(f'Error tolerance for threshold values={tol}')
  p,t,ph,th=bs.gen_threshold(no_samples,dv,dc,depth,no_points,tol)
  # print('\u03B8 values',t)
  # print('corresponding p values',p)
  prefixes = [
    "p values:",
    "\u03B8 values:",
    "Holevo Bound p values:",
    "Holevo Bound \u03B8 values:",
  ]
  arrays=[p,t,ph,th]
  with open(f"{dv}_{dc}_regular_LDPC_threshold.csv", 'w') as f:
    for text, arr in zip(prefixes, arrays):
      arr_np = np.asarray(arr)
      line = f"{text} {np.array2string(arr_np, separator=', ')}\n"
      f.write(line)
      
  choice = input("Please select an option (1: For p vs \u03B8 plot, 2: For \u03C0/2-\u03B8 vs p plot): ")
  if choice== '1':
    print(f"Generating p vs \u03B8 threshold plot for ({int(dv)},{int(dc)}) regular LDPC code ")
    plot1(p,t,ph,th)
  elif choice == '2':
    print(f"Generating \u03C0/2-\u03B8 vs p plot for ({int(dv)},{int(dc)}) regular LDPC code")
    plot2(p,t,ph,th)
  else:
    print("Prinitng p and \u03B8 values. Please select 1 or 2 to generate plots.")
    print('p values:',p)
    print('\u03B8 values',t)
    print('Holevo Bound p values',ph)
    print('Holevo Bound \u03B8 values',th)

    

if __name__== "__main__":
  parser = ap.ArgumentParser('Thresholds for regular LDPC codes over BSCQ channels')
  parser.add_argument('--verbose', '-v', help='Display text output', action="store_true")
  parser.add_argument('-ns', dest='no_samples', type=int, default=1000, help='Number of samples for DE')
  parser.add_argument('-dv', dest='dv', type=int, default=3, help='Bitnode degree')
  parser.add_argument('-dc', dest='dc', type=int, default=4, help='Checknode degree')
  parser.add_argument('-M', dest='depth', type=int, default=60, help='Depth of the tree')
  parser.add_argument('-np', dest='no_points', type=int, default=5, help='Number of points for the plot')
  parser.add_argument('-err', dest='tol', type=float, default=0.005, help='Error tolerance for threshold values')
  # parse arguments
  args = parser.parse_args()
  locals().update(vars(args))
  main()
