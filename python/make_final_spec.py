import os
import numpy as np
from pathlib import Path
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD

from dedalus.core import operators
import h5py
import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

M = -5.5
Mstr="{:.2f}".format(M)
    
if os.path.exists('final_eigs.npy'):
    eig_list = np.load('final_eigs.npy')
else:
    prefix = 'polished_'
    p = Path('../data')
    dir_list = [str(x) for x in p.iterdir() if x.is_dir() if x.name.startswith(prefix)]
    eig_list = np.zeros(len(dir_list),dtype=np.complex128)
    kk = 0
    for dd in dir_list:
        eig = np.load(dd+'/eigs.npy')
        print(1/(2*np.pi*4.66e-7*eig[0].imag)/3600/24)
        print(1/(4.66e-7*(np.abs(eig[0].real)+1e-5))/3600/24)
        eig_list[kk] = eig[0]
        kk = kk+1
    np.save('final_eigs.npy', eig_list)

fig,ax=plt.subplots()
markers= ['o','x']
#ax.scatter(ploteigs.real,ploteigs.imag,label=kkstr,marker=markers[0])
#ax.scatter(ploteigs1.real,ploteigs1.imag,label='All',marker=markers[0])
gamma = np.round(1/(2*np.pi*4.66e-7*eig_list.imag)/3600/24,decimals=0)
eig60 = eig_list[np.where(gamma==60)]
eig601 = eig_list[np.where((gamma==601)&(eig_list.real>0))]
ax.scatter(eig_list.real,eig_list.imag,marker=markers[1],color='k')
ax.scatter(eig60.real,eig60.imag,marker=markers[0],facecolor='none',edgecolor='r',linewidth=2)
ax.scatter(eig601.real,eig601.imag,marker=markers[0],facecolor='none',edgecolor='#9900fa',linewidth=2)

ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\gamma$')

fig.savefig('../png/Final_Eigs_M'+Mstr+'.png', dpi=600)
