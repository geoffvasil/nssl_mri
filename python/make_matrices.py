import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD

from dedalus.core import operators, evaluator
import h5py
import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mri_matrices import MRI_Eigenproblem

# Dedalus Ball Parameters
target_m = 0
if (target_m==0):
    Mmax = 1
else:
    Mmax = 3*target_m
Lmax = 384
Nmax = 64
L_dealias = 3/2 #Dealias by x harmonics
N_dealias = 3/2 #Dealias by y orders

Lfull = int(L_dealias*Lmax)
Nfull = int(N_dealias*Nmax)

#Dedalus Timestepper & Parameters
dtype = np.complex128 #Data type

#Problem Parameters
ri = 0.89
ro = 0.99    #Outer radius
Pm = 1
Re = 10e0**6.0
Rm = Pm*Re #Magnetic Reynolds number
Rei = 1/Re
Rmi = 1/Rm
M = -5.5

#Define radii for coordinate system creators
myM = 10**M
myRe = Re
mynn = Nmax
myell = Lmax
neigs = 32
guess = 0
label = '0'
solver = MRI_Eigenproblem(target_m,myell,mynn,myRe,myM,neigs,guess,MPI.COMM_SELF,label=label)
A = (solver.eigenvalue_subproblem.L_min @ solver.eigenvalue_subproblem.pre_right)
A.data.astype(np.clongdouble)
B = -(solver.eigenvalue_subproblem.M_min @ solver.eigenvalue_subproblem.pre_right)
B.data.astype(np.clongdouble)
            
ainf = sp.linalg.norm(A,ord=np.inf)
A = A/ainf
B = B/ainf
            
sp.save_npz('A.npz', A)
sp.save_npz('B.npz', B)
sp.save_npz('PR.npz',solver.eigenvalue_subproblem.pre_right)
