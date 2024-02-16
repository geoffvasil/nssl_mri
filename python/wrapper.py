import os
import sys
from scipy.interpolate import interpn
from mpi4py import MPI
import numpy as np
from mri_evp import MRI_Eigenproblem
CW = MPI.COMM_WORLD

#search for already finished cases.
Re = 6.0
myRe = 10**Re

neigs = 32
myrank = CW.rank
mynn = 64
ellvals = [256,384]

M = -5.5
Mstr="{:.2f}".format(M)
myM = 10e0**M

if (myrank<21):
    guesses = np.linspace(-0.03,0.03,21)
    for ii in range(4):
        for myell in ellvals:
            guess = guesses[myrank] + 0.005j*(ii+1)
            print(M,myell,guess)
            solver = MRI_Eigenproblem(0,myell,mynn,myRe,myM,neigs,guess,MPI.COMM_SELF,label='{:.4f}j_{:.4f}'.format(guess.imag,guess.real))
else:
    guesses = 1j*np.linspace(0.03,0.14,12)
    for ii in range(4):
        for myell in ellvals:
            guess = guesses[4*(myrank-21) + ii]
            print(M,myell,guess)
            solver = MRI_Eigenproblem(0,myell,mynn,myRe,myM,neigs,guess,MPI.COMM_SELF,label='{:.4f}j_{:.4f}'.format(guess.imag,guess.real))
