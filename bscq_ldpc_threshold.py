import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm,logm
from numpy.linalg import matrix_rank
import copy
import numba
from numba import int64, float64, jit, njit, vectorize, boolean
import matplotlib.pyplot as pl
from tqdm import tqdm
import argparse as ap
import time
import warnings

from scipy.linalg._matfuncs_inv_ssq import LogmExactlySingularWarning 

def pauli(x):
  '''
  function to compute pauli matrices
  outputs Identity, Pauli X,Y and Z 
  '''
  if x==0:
    return np.array([[1,0],[0,1]])
  if x==1:
    return np.array([[0,1],[1,0]])
  if x==2:
    return np.array([[0,-1j],[+1j,0]])
  if x==3:
    return np.array([[1,0],[0,-1]])
  else:
    print('invalid')

# modified rho based on delta, gamma, and input x rather than theta and p

def rhom(delta,g,x):
  '''
  function to compute qubit density matrices
  input: 
       d(float): delta parameter of the BSCQ channel
       g(float): gamma parameter of the BSCQ channel
      x(0 or 1): 1 indicates applying Pauli X on the density matrix

  output:
       (float[:,:]): density matrix

  '''
  return pauli(x)@np.array([[delta, g],[g, 1-delta]])@pauli(x)


def helstrom(density_mat1,unitary):
  '''
  compute eigen values and eigen vectors of rho0- U@rho0@U for Helstrom measurement assuming uniform input
  Arguments:
             density_mat1(float[:,:]): density matrix
             unitary(float[:,:]): unitary
  Returns:
             l(float[:]): eigenvalues
             v(float[:,:]): eigenvectors
  '''
  r1=density_mat1
  u=unitary
  r2=u@r1@np.conjugate(np.transpose(u))
  l,v=LA.eig(r1-r2)
  return l,v


def helstrom_success(rho1,unitary):
  '''
  compute success probability for helstrom measurement assuming uniform input
  input:
      density_mat1= density matrix rho0
      unitary = unitary U
  output: 
      probability of success
  '''
  l,vec=helstrom(rho1,unitary)
  u=unitary
  rho2=u@rho1@np.conjugate(np.transpose(u))
  v_pos_eig = np.array(vec[:,l>0])
  v_pos_eigh= np.conjugate(np.transpose(v_pos_eig))
  p1=np.trace(v_pos_eigh @ rho1 @ v_pos_eig)
  p2=1-np.trace(v_pos_eigh@ rho2 @ v_pos_eig)
  return 0.5*(p1+p2)


def helstrom_success_vec(X):
  '''
  compute success probability for helstrom measurement for vector containing delta, gamma 
  parameters for qubit density matrices with Pauli X symmetry 
  input:
      vector X
      X[0] contains delta values
      X[1] contains gamma values
  output: 
      vector containing helstrom success probability of each pair BSCQ
      channel with delta_{i},gamma_{i} pairs 
  '''
  o=[]
  l=len(X[0])
  for i in range(l):
    d,g=X[0][i],X[1][i]
    r=rhom(d,g,0)
    h=helstrom_success(r,pauli(1))
    o.append(h)
  return o
  #codes using Numba

@njit
def psc_theta_p(t:float64,p:float64)->(float64[:,:]):
  '''
  function to compute density matrices in theta, p parameter domain
  input: 
      t= theta
      probability p
  '''
  q0=np.array([1,0])
  q1=np.array([0,1])
  tq0=np.cos(t/2)*q0+np.sin(t/2)*q1
  tq1=np.cos(-t/2)*q0+np.sin(-t/2)*q1
  return (1-p)*np.outer(tq0,tq0)+p*np.outer(tq1,tq1)

@njit
def psc_thp_dg(t:float64,p:float64)->(float64[:,:]):
  '''
  function to convert density matrices from theta, p parameter domain to 
  delta, gamma parameter domain

  input:
      t= theta
      probability p
  
  output:
      parameter delta=d
             gamma=g

  '''
  H=np.array([[1,1],[1,-1]])/np.sqrt(2)
  rho=H@psc_theta_p(t,p)@np.transpose(H)
  #print(rho)
  d=min(rho[0][0],rho[1][1])
  g=rho[1][0]

  return d,g

@njit
def rhom_jit(delta:float64,g:float64) -> (float64[:,:]):
    '''
    function to compute qubit density matrices in delta, gamma domain in numba format
    '''
    return np.array([[delta, g],[g, 1-delta]],dtype=float64)

# "bitnode" take input states parameterized by (d1, g1) and (d2, g2) and returns the post-measurement state parameters and probabilities at a bitnode
@njit
def bitnode_jit(d1:float64,d2:float64,g1:float64,g2:float64) ->(float64[:,:],float64[:]):
  '''
  Code to compute parameters of post measurement states after applying bitnode operation
         Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel

          Returns:
                  (list[float[:],float[:]]): delta and gamma parameters for the post measurement states
                  (float[:]): probability of the post measurement states
  '''
  # the state given root value z=0 is rho0 = np.kron(rhom(d1,g1,0),rhom(d2,g2,0))
  # the state given root value z=1 is rho1 = np.kron(rhom(d1,g1,1),rhom(d2,g2,1))

  # we find the paired measurements by taking the eigenvectors of the difference matrix rhofull
  x=np.array([[0,1],[1,0]],dtype=float64)

  rhofull = np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))-np.kron(x@rhom_jit(d1,g1)@x,x@rhom_jit(d2,g2)@x)

  evals, evecs = np.linalg.eigh(rhofull)
  # fix eigenvector v0
  v0 = evecs[:,0]

  # symmetry operator Un = np.kron(pauli(1),pauli(1))
  Un = np.kron(x,x)
  # check if the second eigenvector evecs[:,1] is orthogonal to Un@evecs[:,1]
  x1 = evecs[:,1]@(Un@evecs[:,1])

 
  if np.abs(x1)<10e-10:
    v1 = evecs[:,1]
  # if not orthogonal, combine evecs[:,1], evecs[:,2] to create v1 s.t. v1@(Un@v1)= 0
  if np.abs(x1)>=10e-10:
    vec1, vec2 = evecs[:,1], evecs[:,2]
    b11, b12, b21, b22 = np.dot(vec1, (Un@vec1).conj()), np.dot(vec2, (Un@vec1).conj()), np.dot(vec1, (Un@vec2).conj()), np.dot(vec2, (Un@vec2).conj())

    alpha = (-b12-b21-np.sqrt((b12+b21)**2-4*b11*b22))/(2*b22)
    v1 = vec1+alpha*vec2
    v1 = v1/np.sqrt(v1@v1)
    ##

  # the paired measurement is then given by {|v0><v0| + Un|v0><v0|Un, |v1><v1| + Un|v1><v1|Un}
  ## find new state parameters (d1a, g1a) for measurement outcome 0
  # find probability p0 of observing measurement  outcome 0
  p0 = v0@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v0+v0@Un@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@Un@v0
  d1a, g1a = v0@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v0/(p0+10e-21), v0@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@(Un@v0)/(p0+10e-21)
  ## find new state parameters (d1b, g1b) for measurement outcome 1
  # find probability p1 of observing measurement  outcome 1
  p1 = v1@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v1+v1@Un@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@Un@v1
  d2a, g2a = v1@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v1/(p1+10e-21), v1@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@(Un@v1)/(p1+10e-21)
 
  #print([p0, p1])
  d1a=min([d1a,1-d1a])
  d2a=min([d2a,1-d2a])
  return np.array([[d1a, g1a], [d2a, g2a]]), np.array([p0, p1])

# "checknode" take input states parameterized by (d1, g1) and (d2, g2) and returns the post-measurement state parameters and probabilities at a checknode
@njit
def checknode_jit(d1:float64,d2:float64,g1:float64,g2:float64) ->(float64[:,:],float64[:]):
  '''
    Code to compute parameters of post measurement states after applying checknode operation
         Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel

          Returns:
                  (list[float[:],float[:]]): delta and gamma parameters for the post measurement states
                  (float[:]): probability of the post measurement states

  '''
  x=np.array([[0,1],[1,0]],dtype=float64)
  I=np.array([[1,0],[0,1]],dtype=float64)
  # rho0, rho1 correspond to the states at a check node when z=0 (z=1) respectively
  rho0, rho1 = 1/2*(np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2)) + np.kron(x@rhom_jit(d1,g1)@x,x@rhom_jit(d2,g2)@x)), 1/2*(np.kron(rhom_jit(d1,g1),x@rhom_jit(d2,g2)@x) + np.kron(x@rhom_jit(d1,g1)@x,rhom_jit(d2,g2)))
  # for check node combining, the optimal choice of eigenvectors appears to always be generated by v0 and v1
  v0 = 1/np.sqrt(2)*np.array([1,0,0,1])
  v1 = 1/np.sqrt(2)*np.array([-1, 0, 0, 1])
  # symmetry operator for a check node
  Un = np.kron(x,I)
  ## find new state parameters (d1a, g1a) for measurement outcome 0
  # find probability p0 of observing measurement  outcome 0
  p0 = v0@rho0@v0+v0@Un@rho0@Un@v0
  d1a, g1a = v0@rho0@v0/(p0+10e-21), v0@rho0@(Un@v0)/(p0+10e-21)
  ## find new gamma, delta for second outcome
  p1 = v1@rho0@v1+v1@Un@rho0@Un@v1
  d2a, g2a = v1@rho0@v1/(p1+10e-21), v1@rho0@(Un@v1)/(p1+10e-21)
  # return new gamma, delta pairs as well as respective probabilities tra and tr2a
  d1a=min([d1a,1-d1a])
  d2a=min([d2a,1-d2a])
  return np.array([[d1a, g1a], [d2a, g2a]]), np.array([p0, p1])
@njit
def bitnode_jit2(d1:float64,d2:float64,g1:float64,g2:float64,pr=None)->(float64[:,:]):
  '''
   Code to sample post measurement state from bitnode operation
         Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel
                  pr(0 or 1): output specific post measurement state

          Returns:
                  (float64[:]): delta,gamma parameters associated with one of the post measurement density matrix
                  
  '''

  rho,pb=bitnode_jit(d1,d2,g1,g2)
  if pr==None:
    s=pb[0]
  else:
    s=pr
  o=int(np.random.random()>s)
  #choice([0,1],p=[s,1-s])
  if o==0:
    return rho[0]
  else:
    return rho[1]

@njit
def checknode_jit1(d1:float64,d2:float64,g1:float64,g2:float64,pr=None)->(float64[:,:]):
  '''
   Code to sample post measurement state from checknode operation
         Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel
                  pr(0 or 1): output specific post measurement state

          Returns:
                  (float64[:]): delta,gamma parameters associated with one of the post measurement density matrix
  '''

  rho,pc=checknode_jit(d1,d2,g1,g2)
  if pr==None:
    s=pc[0]
  else:
    s=pr
  o=int(np.random.random()>s)
  if o==0:
    return rho[0]
  else:
    return rho[1]

@njit
def bitnode_vec_jit(d1:float64[:],d2:float64[:],g1:float64[:],g2:float64[:],pr_vec=None,perm=None)->(float64[:],float64[:]):
  '''
   Code to sample post measurement state from bitnode operation from delta, gamma parameters
   for a set of different BSCQ channels
         Arguments:
                  d1(float64[:]): parameter delta for the first set of BSCQ channels
                  d2(float64[:]): parameter delta for the second set of BSCQ channel
                  g1(float64[:]) : parameter gamma for the first set of BSCQ channels
                  g2(float64[:]): parameter gamma for the second  set of BSCQ channels
                  pr_vec(binary [:]): output specific post measurement state
                  perm(int[:]): permuation on second set to combine bitnode operation
                               if none, we apply random permutation


          Returns:
                  d(float64[:]): delta parameters for combined channels
                  g(float64[:])  gamma parameters for combined channels
                  
  '''
  l=np.shape(d1)[0]
  if perm==None:
    p=np.random.permutation(l)
  else:
    p=perm
  d=np.zeros(l)
  g=np.zeros(l)
  if pr_vec==None:
    for i in range(l):
      d[i],g[i]=bitnode_jit2(d1[i],d2[p[i]],g1[i],g2[p[i]])
  else:
    for i in range(l):
      d[i],g[i]=bitnode_jit2(d1[i],d2[p[i]],g1[i],g2[p[i]],pr_vec[i])
  return d,g
@njit
def checknode_vec_jit(d1:float64[:],d2:float64[:],g1:float64[:],g2:float64[:],pr_vec=None,perm=None)->(float64[:],float64[:]):
  '''
   Code to sample post measurement state from checknode operation from delta, gamma parameters
   for a set of different BSCQ channels
         Arguments:
                  d1(float64[:]): parameter delta for the first set of BSCQ channels
                  d2(float64[:]): parameter delta for the second set of BSCQ channel
                  g1(float64[:]) : parameter gamma for the first set of BSCQ channels
                  g2(float64[:]): parameter gamma for the second  set of BSCQ channels
                  pr_vec(binary [:]): output specific post measurement state
                  perm(int[:]): permuation on second set to combine checknode operation
                                if none, we apply random permutation

          Returns:
                  d(float64[:]): delta parameters for combined channels
                  g(float64[:])  gamma parameters for combined channels
                  
  '''
  l=np.shape(d1)[0]
  if perm==None:
    p=np.random.permutation(l)
  else:
    p=perm
  d=np.zeros(l)
  g=np.zeros(l)
  if pr_vec==None:
    for i in range(l):
      d[i],g[i]=checknode_jit1(d1[i],d2[p[i]],g1[i],g2[p[i]])
  else:
    for i in range(l):
      d[i],g[i]=checknode_jit1(d1[i],d2[p[i]],g1[i],g2[p[i]],pr_vec[i])
  return d,g


# @njit
# def bitnode_power_jit(d:float64[:],g:float64[:],k:int64,pr_vec=None,perm=None)->(float64[:],float64[:]):
#   '''
#    Code to sample post measurement state from bitnode operation from delta, gamma parameters
#    for a set of  BSCQ channels with given bitnode degree associated with LDPC or LDGM code
#          Arguments:
#                   d(float64[:]): parameter delta for a set of BSCQ channels
                  
#                   g(float64[:]) : parameter gamma for a set of BSCQ channels
                  
#                   k(int64): number of time bitnode to be applied on the channels
                  
#                   pr_vec(binary [:]): output specific post measurement state
#                   perm(int[:]): permuation on second set to combine bitnode operation
#                                if none, we apply random permutation


#           Returns:
#                   d(float64[:]): delta parameters for combined channels
#                   g(float64[:])  gamma parameters for combined channels
                  
#   '''
#   if k==1:
#     return d,g
#   else:
#     d1,g1=bitnode_vec_jit(d,d,g,g,pr_vec,perm)
#     if k>2:
#       for i in range(k-2):
#         d1,g1=bitnode_vec_jit(d,d1,g,g1,pr_vec,perm)
#     return d1,g1
  
@njit
def bitnode_power_jit(d:float64[:],g:float64[:],k:int64,pr_vec=None,perm=None)->(float64[:],float64[:]):
    '''
    Code to sample post-measurement state from bitnode operation with delta, gamma parameters
    for a set of BSCQ channels associated with an LDPC or LDGM code, using exponentiation by squaring.
    
    Arguments:
        d (float64[:]): Delta parameter for a set of BSCQ channels
        g (float64[:]): Gamma parameter for a set of BSCQ channels
        k (int64): Number of times bitnode to be applied to the channels
        pr_vec (binary[:], optional): Output-specific post-measurement state
        perm (int[:], optional): Permutation on the second set to combine bitnode operation; 
                                  if None, a random permutation is applied

    Returns:
        (float64[:], float64[:]): Updated delta and gamma parameters for combined channels
    '''
    
#    
    # Initialize the result as the identity in terms of the bitnode operation
    d_result, g_result = d.copy(), g.copy()
    
    # Base matrices (starting with initial d and g)
    d_base, g_base = d.copy(), g.copy()

    # if k==1:
    #    return d_result,g_result
    
    k=k-1
    while k > 0:
        if k % 2 == 1:  # If k is odd, multiply the result by the current base
            d_result, g_result = bitnode_vec_jit(d_result, d_base, g_result, g_base, pr_vec, perm)
        
        # Square the base
        d_base, g_base = bitnode_vec_jit(d_base, d_base, g_base, g_base, pr_vec, perm)
        
        # Halve k
        k //= 2
    
    return d_result, g_result
# @njit
# def checknode_power_jit(d:float64[:],g:float64[:],k:int64,pr_vec=None,perm=None)->(float64[:],float64[:]):
#   '''
#    Code to sample post measurement state from checknode operation from delta, gamma parameters
#    for a set of  BSCQ channels with given checknode degree associated with LDPC or LDGM code
#          Arguments:
#                   d(float64[:]): parameter delta for a set of BSCQ channels
                  
#                   g(float64[:]) : parameter gamma for a set of BSCQ channels
                  
#                   k(int64): number of time checknode to be applied on the channels
                  
#                   pr_vec(binary [:]): output specific post measurement state
#                   perm(int[:]): permuation on second set to combine checknode operation
#                                if none, we apply random permutation


#           Returns:
#                   d(float64[:]): delta parameters for combined channels
#                   g(float64[:])  gamma parameters for combined channels
                  
#   '''
#   if k==1:
#     return d,g
#   else:
#     d1,g1=checknode_vec_jit(d,d,g,g,pr_vec,perm)
#     if k>2:
#       for i in range(k-2):
#         d1,g1=checknode_vec_jit(d,d1,g,g1,pr_vec,perm)
#     return d1,g1


@njit
def checknode_power_jit(d:float64[:],g:float64[:],k:int64,pr_vec=None,perm=None)->(float64[:],float64[:]):
    '''
    Code to sample post-measurement state from checknode operation with delta, gamma parameters
    for a set of BSCQ channels associated with an LDPC or LDGM code, using exponentiation by squaring.
    
    Arguments:
        d (float64[:]): Delta parameter for a set of BSCQ channels
        g (float64[:]): Gamma parameter for a set of BSCQ channels
        k (int64): Number of times checknode to be applied to the channels
        pr_vec (bool_[:], optional): Output-specific post-measurement state
        perm (int64[:], optional): Permutation on the second set to combine checknode operation; 
                                  if None, a random permutation is applied

    Returns:
        (float64[:], float64[:]): Updated delta and gamma parameters for combined channels
    '''
    # Convert empty arrays to None if not provided to handle optional arguments
   
    # Identity state for exponentiation by squaring
    d_result, g_result = d.copy(), g.copy()
    d_base, g_base = d.copy(), g.copy()
    
    
    k=k-1
    while k > 0:
        if k % 2 == 1:  # If k is odd, multiply the result by the current base
            d_result, g_result = checknode_vec_jit(d_result, d_base, g_result, g_base, pr_vec, perm)
        
        # Square the base
        d_base, g_base = checknode_vec_jit(d_base, d_base, g_base, g_base, pr_vec, perm)
        
        # Halve k
        k //= 2
    
    return d_result, g_result

@njit
def channel_density_jit(d:float64[:],g:float64[:],dv:int64,dc:int64,n:int64,code='LDPC')->(float64[:,:,:]):
  '''
   code to compute delta, gamma parameters associated for a given regular LDPC or LDGM code
   with bitnode degree dv and checknode degree dv
   input:
      d(float[:]): delta parameters for identical BSCQ channels
                   typically of the form= delta*np.ones(number of channels)
      g(float[:]): gamma parameters for identical BSCQ channels
                   typically of the form= gamma*np.ones(number of channels)
      dv: bitnode degree
      dc: checknode degree
      n: depth of the tree
      code = LDPC(default)  or LDGM
    
   output:
      x (float[:,:,:]):x[i][0] contains delta parameters at depth i
                       x[i][1] contains gamma parameters at depth i
  '''
  x=np.zeros(shape=(n,2,len(d)),dtype=float64)
  x[0][0],x[0][1]=d,g
  if code=='LDPC':
    for i in range(n-1):
      d1,g1=x[i][0],x[i][1]
      dcheck,gcheck=checknode_power_jit(d1,g1,dc-1)
      db,gb=bitnode_power_jit(dcheck,gcheck,dv-1)
      df,gf=bitnode_vec_jit(d,db,g,gb)
      x[i+1][0],x[i+1][1]=df,gf
  elif code=='LDGM':
    for i in range(n-1):
      d1,g1=x[i][0],x[i][1]
      dcheck,gcheck=bitnode_power_jit(d1,g1,dc-1)
      db,gb=checknode_power_jit(dcheck,gcheck,dv-1)
      df,gf=checknode_vec_jit(d,db,g,gb)
      x[i+1][0],x[i+1][1]=df,gf



  return x

def helstrom_channel_density(t,p,no_samples,dv,dc,depth,code='LDPC'):
  '''
  code to compute average Helstrom error probability associated for a given regular LDPC or LDGM code
  with bitnode degree dv and checknode degree dv at each depth at theta, p domain
  input:
    t: theta parameter of the channel
    p: p parameter of the channel
    no_samples: number of samples
    dv:bitnode degree
    dc: checknode degree
    depth: depth of the tree
    code = LDPC(default)  or LDGM

  output:
    h([float:]): Helstrom error probability at each level

  
  '''
  dp,gp=psc_thp_dg(t,p)
  d=np.ones(no_samples)*dp
  g=np.ones(no_samples)*gp
  O=channel_density_jit(d,g,dv,dc,depth)
  h=np.zeros(depth-1)
  for i in range(depth-1):
    h[i]=1-np.average(helstrom_success_vec(O[i+1]))

  return h

def calculate_max_iterations(t_left, t_right, tol):
  L = t_right - t_left  # Initial length of the interval
  return int(np.ceil(np.log2(L / tol)))

# Binary search to find t for a given p
def binary_search_t(p,t_min,no_samples,dv,dc,depth,tol=0.02,code='LDPC'):
    t_left, t_right = t_min, np.pi/2
    max_iter = calculate_max_iterations(t_left, t_right, tol)  # Determine max iterations
    iteration = 0

    while t_right - t_left > tol and iteration < max_iter:
        t_mid = (t_left + t_right) / 2
        h=helstrom_channel_density(t_mid,p,no_samples,dv,dc,depth)
        #print(t_mid)
        if min(h) > 1e-4:
            t_left = t_mid  # Adjust the search range

        else:
            t_right = t_mid
        iteration += 1

    return (t_left + t_right) / 2

# Binary search to find p for a given t
def binary_search_p(t,p_max,no_samples,dv,dc,depth,tol=0.005,code='LDPC'):
    p_left, p_right = 0, p_max
    max_iter = calculate_max_iterations(p_left, p_right, tol)  # Determine max iterations
    iteration = 0

    while p_right - p_left > tol and iteration < max_iter:
        p_mid = (p_left + p_right) / 2
        h=helstrom_channel_density(t,p_mid,no_samples,dv,dc,depth)
        #print(p_mid)
        if min(h) > 1e-4:
            p_right = p_mid  # Adjust the search range

        else:
            p_left = p_mid
        iteration += 1

    return (p_left + p_right) / 2


def von_neumann(rho):
  '''
  function to compute von-Neumann entropy (in bits)
  input:
      
      rho: indicates applying Pauli X on the density matrix

  output:
      von-Neumann entropy
       

  '''
  #rho_log=logm(rho,False)[0]/np.log(2)
  #return -np.trace(rho@rho_log)
  with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LogmExactlySingularWarning)
        rho_log=logm(rho,False)[0]/np.log(2)
        return -np.trace(rho@rho_log)


def holevo_bound(rho0,rho1):
  '''
   function to compute holevo bound of BSCQ channel
   input:
    rho1: first density matrix
    rho2: second density matrix
    
   output:
     holevo bound
   '''
   
  return von_neumann(0.5*rho0+0.5*rho1)-0.5*von_neumann(rho0)-0.5*von_neumann(rho1)

def binary_search_holevo_t(p,dv,dc,tol=0.005):
    t_left, t_right = 0, np.pi/2
    max_iter = calculate_max_iterations(t_left, t_right, tol)  # Determine max iterations
    iteration = 0
    rate=1-(dv/dc)

    while t_right - t_left > tol and iteration < max_iter:
        t_mid = (t_left + t_right) / 2
        rho0=psc_theta_p(t_mid,p)
        rho1=pauli(3)@rho0@pauli(3)
        h=holevo_bound(rho0,rho1)
        

        #print(t_mid)
        if  h > rate:
            t_right = t_mid  # Adjust the search range

        else:
            t_left = t_mid
        iteration += 1

    return (t_left + t_right) / 2



def binary_search_holevo_p(t,dv,dc,tol=0.005):
    p_left, p_right = 0, 0.5
    max_iter = calculate_max_iterations(p_left, p_right, tol)  # Determine max iterations
    iteration = 0
    rate=1-(dv/dc)

    while p_right - p_left > tol and iteration < max_iter:
        p_mid = (p_left + p_right) / 2
        rho0=psc_theta_p(t,p_mid)
        rho1=pauli(3)@rho0@pauli(3)
        h=holevo_bound(rho0,rho1)
        #print(p_mid)
        if h > rate:
            p_left= p_mid  # Adjust the search range

        else:
            p_right = p_mid
        iteration += 1

    return (p_left + p_right) / 2


def gen_threshold(no_samples,dv,dc,depth,no_points,tol,code='LDPC'):
     
     start_time = time.time()
     t0=binary_search_t(0,0,no_samples,dv,dc,depth,tol)
     end_time = time.time()
     print('(\u03B8\u2080,0)=',(t0,0))
     print("Execution time to compute first point:", end_time - start_time, "seconds")
     p0=binary_search_p(np.pi/2,0.5,no_samples,dv,dc,depth,tol)
     print('(\u03C0/2,p\u2080)=',(np.pi/2,p0))
     p_min=p0/(no_points-1)
     p_max=p0-p0/(no_points-1)
     p_val=np.linspace(p_min,p_max,no_points-2)
     t_val=[t0]
     GREEN = '\033[92m'
     RESET = '\033[0m'
     for i in tqdm(range(len(p_val)),desc="searching for thresholds (\u03B8\u2096,p\u2096)",bar_format=f'{GREEN}{{l_bar}}{{bar}}{{r_bar}}{RESET}'):
       time.sleep(0.05)
       tk=binary_search_t(p_val[i],t_val[i],no_samples,dv,dc,depth,tol)
       print(f'{tk,p_val[i]}')
       t_val.append(tk)
     p_val=np.concatenate([[0],p_val])
     p_val=np.concatenate([p_val,[p0]])
     t_val.append(np.pi/2)

     print(f'Computing Holevo Bound for {(dv,dc)} regular LDPC code')
     th0=binary_search_holevo_t(0,dv,dc,tol)
     print('(\u03B8\u2080,0)=',(th0,0))

     ph0=binary_search_holevo_p(np.pi/2,dv,dc,tol)
     print('(\u03C0/2,p\u2080)=',(np.pi/2,ph0))
     p_min=ph0/(no_points-1)
     p_max=ph0-ph0/(no_points-1)
     p_val1=np.linspace(p_min,p_max,no_points-2)
     t_val1=[th0]

     for i in tqdm(range(len(p_val1)),desc="searching for thresholds (\u03B8\u2096,p\u2096) for Holevo Bound",bar_format=f'{GREEN}{{l_bar}}{{bar}}{{r_bar}}{RESET}'):
       time.sleep(0.05)
       tk=binary_search_holevo_t(p_val1[i],dv,dc,tol)
       print(f'{tk,p_val1[i]}')
       t_val1.append(tk)
     p_val1=np.concatenate([[0],p_val1])
     p_val1=np.concatenate([p_val1,[ph0]])
     t_val1.append(np.pi/2)

     return p_val,t_val,p_val1,t_val1

def main():
  print(f'Thresholds for ({dv},{dc}) regular LDPC codes over BSCQ channels')
  p,t,ph,th=gen_threshold(no_samples,dv,dc,depth,no_points,tol)
  print('\u03B8 values',t)
  print('corresponding p values',p)
  print('Holevo bound:',[ph,th])
if __name__== "__main__":
  parser = ap.ArgumentParser('Thresholds for regular LDPC codes over BSCQ channels')
  parser.add_argument('--verbose', '-v', help='Display text output', action="store_true")
  parser.add_argument('-ns', dest='no_samples', type=int, default=1000, help='Number of samples for DE')
  parser.add_argument('-dv', dest='dv', type=int, default=3, help='Bitnode degree')
  parser.add_argument('-dc', dest='dc', type=int, default=4, help='Checknode degree')
  parser.add_argument('-M', dest='depth', type=int, default=60, help='Depth of the tree')
  parser.add_argument('-np', dest='no_points', type=int, default=6, help='Number of points for the plot')
  parser.add_argument('-err', dest='tol', type=float, default=0.005, help='Error tolerance for threshold values')
  # parse arguments
  args = parser.parse_args()
  locals().update(vars(args))
  main()
