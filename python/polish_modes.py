import os
import time
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
radii = (ri,ro)

mesh = None

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, mesh=mesh, dtype=dtype, comm=MPI.COMM_SELF)
b = d3.ShellBasis(c, (Mmax,Lmax,Nmax), radii=radii, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))

# Fields
myuu = d.VectorField(c, bases=b, name='uu')
mybb = d.VectorField(c, bases=b, name='bb')
parity = d.Field(bases=b, name='parity')

ells = [256,384]
kk = ells[0]
Mstr="{:.2f}".format(M)
kkstr="{:d}".format(kk)
prefix = "ModelS_Fit_M" + Mstr + "_ellmax"+kkstr+"_"
p = Path('.')
dir_list = [str(x) for x in p.iterdir() if x.is_dir() if x.name.startswith(prefix)]

kk = ells[1]
kkstr1="{:d}".format(kk)
prefix = "ModelS_Fit_M" + Mstr + "_ellmax"+kkstr1+"_"
p1 = Path('.')
dir_list1 = [str(x) for x in p1.iterdir() if x.is_dir() if x.name.startswith(prefix)]

eigs = np.load(dir_list[0]+'/eigenvalues.npy')
neigs = len(eigs)

ploteigs = np.array([])
for dd in dir_list:
    eigs =np.load(dd+'/eigenvalues.npy')
    ploteigs = np.append(ploteigs,eigs)
ploteigs.flatten()

ploteigs1 = np.array([])
for dd in dir_list1:
    eigs =np.load(dd+'/eigenvalues.npy')
    ploteigs1 = np.append(ploteigs1,eigs)
ploteigs1.flatten()

ndeigs = len(ploteigs1)
diffs = 0*ploteigs1
for ii in range(ndeigs):
    diffs[ii] = np.min(np.abs(ploteigs1[ii]-ploteigs))/np.abs(ploteigs1[ii])

ploteigs1[np.where((diffs>1e-4) | (ploteigs1.imag<0.001))] = np.inf

ndir = len(dir_list1)
myrank = CW.rank
size = CW.size
first = True
for kk in range(ndir): #myrank,ndir,size):
    dd = dir_list1[kk]
    tmpeig = ploteigs1[kk*neigs:(kk+1)*neigs]
    ieigs = np.where(np.isfinite(tmpeig))[0]
    for jj in ieigs:
        #print(jj,tmpeig[jj])
        f1 = h5py.File(dd + '/checkpoints/checkpoints_s{:d}'.format(jj+1)+'.h5')

        mybb.load_from_hdf5(f1,0)
        myuu.load_from_hdf5(f1,0)

        #Get parity of u_phi:
        parity.change_scales(3/2)
        parity['g'] = myuu['g'][0]*np.conjugate(myuu['g'][0])
        tmp = d3.integ(parity).evaluate()
        norm = np.sqrt(tmp['g'][0])
        parity.change_scales(3/2)
        parity['g']=myuu['g'][0]+np.flip(myuu['g'][0],axis=1)
        parity['g']=parity['g']*np.conjugate(parity['g'])
        tmp = d3.integ(parity).evaluate()
        evenp = np.sqrt(tmp['g'][0])/norm/2e0
        evenp = evenp.flatten()
        evenp = evenp[0].real
        uparity = 2e0*evenp-1e0
        #print('u_phi parity:',uparity)

        #Get parity of b_phi:
        parity.change_scales(3/2)
        parity['g'] = mybb['g'][0]*np.conjugate(mybb['g'][0])
        tmp = d3.integ(parity).evaluate()
        norm = np.sqrt(tmp['g'][0])
        parity.change_scales(3/2)
        parity['g']=mybb['g'][0]+np.flip(mybb['g'][0],axis=1)
        parity['g']=parity['g']*np.conjugate(parity['g'])
        tmp = d3.integ(parity).evaluate()
        evenp = np.sqrt(tmp['g'][0])/norm/2e0
        evenp = evenp.flatten()
        evenp = evenp[0].real
        bparity = 2e0*evenp-1e0
        #print('b_phi parity:',bparity)
        
        if ((np.abs(uparity-1e0)<1e-2) and (np.abs(np.abs(bparity)-1e0)<1e-2)):
            guess = tmpeig[jj] #.astype(np.clongdouble)
            gamma = 1/(2*np.pi*4.66e-7*guess.imag)/3600/24
            gamma_txt = '{:.0f}'.format(gamma)
            tau = 1/(4.66e-7*guess.real)/3600/24
            if (np.abs(tau)>1000*gamma):
                tau = 0
                tau_txt = 'none'
            else:
                tau_txt = '{:.0f}'.format(tau)

            label='te'+gamma_txt+'_P'+tau_txt
            if first:
                first = False 
                myM = 10**M
                myRe = Re
                mynn = Nmax
                myell = Lmax
                neigs = 32
                solver = MRI_Eigenproblem(target_m,myell,mynn,myRe,myM,neigs,guess,MPI.COMM_SELF,label=label)
                #A = sp.load_npz('A.npz')
                #B = sp.load_npz('B.npz')
                #PR = sp.load_npz('PR.npz')
                subproblem = solver.subproblems_by_group[(target_m, None, None)]
                ss = subproblem.subsystems[0]

            for f in tuple(solver.state):
                f.load_from_hdf5(f1,0)
            eigv = ss.gather(solver.state)
            
            namespace = {}
            solver.evaluator = evaluator.Evaluator(solver.dist, namespace)
            data_dir = '../data/polished_'+label
            print('checkpointing in {}'.format(data_dir))
            
            if (not os.path.exists('{:s}'.format(data_dir))):
                os.mkdir('{:s}'.format(data_dir))
                
                path = data_dir + '/checkpoints'
                checkpoint = solver.evaluator.add_file_handler(path, max_writes=1)
                checkpoint.add_tasks(solver.state)
                ss.scatter(eigv,solver.state)
                
                uparerr = np.abs(uparity-1e0)
                bparerr = np.abs(np.abs(bparity)-1e0)
                #eigenvalue, eigenerr, uparity, bparity
                eigs = np.zeros(4,dtype=np.complex128)
                eigenerr = diffs[kk*neigs:(kk+1)*neigs]
                eigenerr = eigenerr[jj]
                eigs[0] = guess
                eigs[1] = eigenerr
                eigs[2] = uparerr
                eigs[3] = bparerr
                np.save(data_dir+'/eigs.npy',eigs)
                solver.evaluator.evaluate_handlers([checkpoint],sim_time=1,wall_time=1,timestep=1,iteration=1)
            else:
                uparerr = np.abs(uparity-1e0)
                bparerr = np.abs(np.abs(bparity)-1e0)
                #eigenvalue, eigenerr, uparity, bparity
                eigs = np.zeros(4,dtype=np.complex128)
                eigenerr = diffs[kk*neigs:(kk+1)*neigs]
                eigenerr = eigenerr[jj]
                eigs[0] = guess
                eigs[1] = eigenerr
                eigs[2] = uparerr
                eigs[3] = bparerr
                
                eigs_old = np.load(data_dir+'/eigs.npy')
                    
                if ((np.abs(eigs_old[1])>np.abs(eigs[1])) and (np.abs(eigs_old[2])>np.abs(eigs[2])) and (np.abs(eigs_old[3])>np.abs(eigs[3]))):
                    np.save(data_dir+'/eigs.npy',eigs)
                    path = data_dir + '/checkpoints'
                    checkpoint = solver.evaluator.add_file_handler(path, max_writes=1)
                    checkpoint.add_tasks(solver.state)
                    ss.scatter(eigv,solver.state)
                    solver.evaluator.evaluate_handlers([checkpoint],sim_time=1,wall_time=1,timestep=1,iteration=1)
                
