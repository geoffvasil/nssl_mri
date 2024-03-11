import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors
import matplotlib.patches as patches

from scipy.integrate import cumtrapz as integral
from scipy.integrate import trapz

import scipy as sp
import math

save_figs = True

plt.rcParams['pdf.fonttype']=42

def D(x,k=2,d=1):
    """Returns a matrix for the (2k+1)-point
        d-th FD-derivative on a non-uniform grid.
        The interior points are centered.
        The end points are offset.
        
        D^{d}f(x0) ~ sum_{j=-k}^{j=+k} a(j)*f(x(j)) ~
        sum_{j=-k}^{j=+k} sum_{p=0}^{p=2k+1} a(j)*(x(j)-x0)**p D^{p}f(x0)/p!
        
        sum_{j=-k}^{j=+k} a(j)*(x(j)-x0)**p = d! if p == d (otherwise 0).
        
        Parameters
        ----------
        x: numpy float array
        k: half-width of stencil, default k=2
        d: derivative order, default d=1
        
        In terms of the local grid size,
        the error is Order(2k+1-d), default O(4).
        
        """
    
    n = len(x)
    S = np.zeros((n,n))
    w = 2*k+1
    R = np.zeros(w)
    R[d] = math.factorial(d)
    p = np.arange(w)[:,None]
    for i in range(n):
        i0 = np.min([np.max([0,i-k]),n-w])
        i1 = np.min([np.max([w,i+k+1]),n])
        L = (x[i0:i1]-x[i])[None,:]**p
        S[i,i0:i1] = np.linalg.solve(L,R)
    return S
    
# From Larsen @ Schou 2012

data_file = '../data/HMI/rot2d.hmiv72d.ave'
radius_file = '../data/HMI/rmesh.orig'

data = np.genfromtxt(data_file)

n_lat = data.shape[1]
n_r = data.shape[0]
lat = 90-np.arange(n_lat)*15/8
radius = np.genfromtxt(radius_file)[::4]
theta  = np.pi/2 - lat*np.pi/180

r0, r1 = 0.6, 1.00

Omega  = data[np.where((r0 < radius) & (radius<r1))[0],:]
radius = radius[np.where((r0 < radius) & (radius<r1))]

Sin = np.sin(theta[None,:])
Cos = np.cos(theta[None,:])

R = radius[:,None]

X = R*Sin
Y = R*Cos

nlev = 15
levels = np.linspace(np.min(Omega),np.max(Omega),nlev)

# plot Omega

plt.rc('axes', linewidth=2)
plt.rc('font', weight='regular')

pmax = 460
pmin = 290

cmap = 'plasma'
cmap = 'magma'
cmap = 'inferno'
cmap = 'summer_r'

Z = np.copy(Omega)
i = np.where( np.sqrt(X**2 + Y**2) > 1 )
Z[i] = None

fig, ax = plt.subplots()

pcm = ax.pcolormesh(X,Y,Z, shading='gouraud',cmap=cmap,vmin=pmin, vmax=pmax, rasterized=True)

ax.contour(X,Y,Z,colors='k',levels=20)

plt.axis([0,1.01,0,1.01])

ax.set_aspect(1)

ax.set_title(r"$\Omega$",fontsize=16,fontweight="bold");

ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)

plt.xticks(fontsize=13,rotation=0)
plt.yticks(fontsize=13,rotation=0)

cNorm = colors.Normalize(vmin=pmin, vmax=pmax)

#ax_cb = fig.add_axes([0.82, 0.3, 0.03, 0.4])

ax_cb = fig.add_axes([0.25, 0.14, 0.025, 0.31])

cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)

cb.ax.tick_params(labelsize=13)

ax_cb.set_title("nHz",fontsize=13);

ax_cb.yaxis.set_tick_params(width=2)

ax.set_xlabel('$ r\, /\, R_{\odot}$',fontsize=14)

if save_figs:
    plt.savefig('../pdf/rotation_rate.pdf', dpi=600,bbox_inches="tight")


# Compute the shear vector: S = X*grad(Omega)

plt.rc('axes', linewidth=2)
plt.rc('font', weight='regular')

Sr = D(radius) @ Omega
Sr *= X

pmax = 501
pmin = -pmax

cmap = 'RdBu'

Z = np.copy(Sr)
i = np.where( np.sqrt(X**2 + Y**2) > 1 )
Z[i] = None


fig, ax = plt.subplots()


pcm = ax.pcolormesh(X,Y,Z, shading='gouraud',cmap=cmap,vmin=pmin, vmax=pmax, rasterized=True)

plt.axis([0,1.01,0,1.01])

ax.set_aspect(1)

ax.set_title(r"$r\, \sin(\theta)\, \partial_{r}\, \Omega$",fontsize=16,fontweight="bold");

ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)

plt.xticks(fontsize=13,rotation=0)
plt.yticks(fontsize=13,rotation=0)

cNorm = colors.Normalize(vmin=pmin, vmax=pmax)

ax_cb = fig.add_axes([0.25, 0.14, 0.025, 0.31])

cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)

cb.ax.tick_params(labelsize=13)

ax_cb.set_title("nHz",fontsize=13);

ax_cb.yaxis.set_tick_params(width=2)

ax.set_xlabel('$r\, /\, R_{\odot}$',fontsize=14)

if save_figs:
    plt.savefig('../pdf/radial_shear.pdf', dpi=600,bbox_inches="tight")
    
 

# Compute the shear vector: S = X*grad(Omega)

plt.rc('axes', linewidth=2)
plt.rc('font', weight='regular')

Sth = Omega @ D(theta).T
Sth *= Sin

pmax = 100
pmin = -pmax

cmap = 'RdBu'

Z = np.copy(Sth)

i = np.where( np.sqrt(X**2 + Y**2) > 1 )
Z[i] = None


fig, ax = plt.subplots()


pcm = ax.pcolormesh(X,Y,Z, shading='gouraud',cmap=cmap,vmin=pmin, vmax=pmax, rasterized=True)

plt.axis([0,1.01,0,1.01])

ax.set_aspect(1)

ax.set_title(r"$\sin(\theta)\, \partial_{\theta}\, \Omega$",fontsize=16,fontweight="bold");

ax.xaxis.set_tick_params(width=3)
ax.yaxis.set_tick_params(width=3)

plt.xticks(fontsize=13,rotation=0)
plt.yticks(fontsize=13,rotation=0)

cNorm = colors.Normalize(vmin=pmin, vmax=pmax)

#ax_cb = fig.add_axes([0.82, 0.3, 0.03, 1-0.3*2])

ax_cb = fig.add_axes([0.25, 0.14, 0.025, 0.31])

cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)

cb.ax.tick_params(labelsize=13)

ax_cb.set_title("nHz",fontsize=13);

ax_cb.yaxis.set_tick_params(width=2)

ax.set_xlabel('$ r\, /\, R_{\odot}$',fontsize=14)

if save_figs:
    plt.savefig('../pdf/latitudinal_shear.pdf', dpi=600,bbox_inches="tight")
