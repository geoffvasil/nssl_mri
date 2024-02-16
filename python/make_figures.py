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


saturation = 0.97
fig_scales = 3*(3,)
#cases      = ['te601_P292', 'te60_Pnone']
prefix = '../data/polished_'
p = Path('.')
dir_list = [str(x) for x in p.iterdir() if x.is_dir() if x.name.startswith(prefix)]
cases = []
for dd in dir_list:
    parts = dd.split('_')
    cases.append(parts[1]+'_'+parts[2])

π      = np.pi
ri, ro = 0.89, 0.99
volume = 4*π * (ro**3 - ri**3 )/3

nHz = 2*π * 1e-9
day = 3600 * 24
Ω0  = 466  * nHz * day

μ0 = 4*π
ρ0 = 0.0309

rsun    = 696e8
u_units = rsun*nHz
b_units = np.sqrt(μ0*ρ0)*u_units

dtype   = np.complex128
φθr     = d3.SphericalCoordinates('φ', 'θ', 'r')
dist    = d3.Distributor(φθr,dtype=dtype,comm=MPI.COMM_SELF)
basis   = d3.ShellBasis(φθr, (2,384,64), radii=(ri,ro), dtype=dtype, dealias=3*(1,))
φ, θ, r = basis.global_grids(dist, scales=(1,1,1))

# auxiliary data

grad    = d3.grad
curl    = d3.curl
div     = d3.div
angular = d3.angular
integ   = d3.integ
lift    = lambda A: d3.Lift(A, basis.clone_with(k=1), -1)

ρ    = dist.Field(bases=basis.meridional_basis, dtype=dtype,name='ρ')
dρ   = dist.VectorField(φθr,bases=basis.meridional_basis, dtype=dtype,name='dρ')
rvec = dist.VectorField(φθr,bases=basis.meridional_basis,name='rvec')

z = (2*r-ri-ro)/(ro-ri)

ρ_coefs    = [ 0.4287056, -0.50122643, 0.07841967, 0.00206746, 0.00243878]
ρ['g']     = ρ0 * np.polynomial.chebyshev.chebval(z,ρ_coefs)

dρ_coefs   = [-9.78493705, 6.46519649, 0.45630896, 0.24012609, 0.11130384]
dρ['g'][2] = ρ0 * np.polynomial.chebyshev.chebval(z,dρ_coefs)

rvec['g'][2] = r

τi = dist.VectorField(φθr,bases=basis.S2_basis(radius=ri),dtype=dtype,name='τi')
τo = dist.VectorField(φθr,bases=basis.S2_basis(radius=ro),dtype=dtype,name='τo')
τχ = dist.Field(dtype=dtype,name='τχ')
χ  = dist.Field(bases=basis,dtype=dtype,name='χ')

u = dist.VectorField(φθr, bases=basis, name='uu')
b = dist.VectorField(φθr, bases=basis, name='bb')

a  = dist.VectorField(φθr, bases=basis, dtype=dtype, name='a')
da = grad(a) + rvec*lift(τi)

problem1 = d3.LBVP([a,χ,τi,τo,τχ],namespace=locals())
problem1.add_equation("trace(da) + τχ = 0")
problem1.add_equation("div(da) + grad(χ) + lift(τo) = -curl(b)")
problem1.add_equation("angular(a(r=ri)) = 0")
problem1.add_equation("angular(a(r=ro)) = 0")
problem1.add_equation("χ(r=ri)  = 0")
problem1.add_equation("χ(r=ro)  = 0")
problem1.add_equation("integ(χ) = 0")

solver1 = problem1.build_solver()

ψ  = dist.VectorField(φθr, bases=basis, dtype=dtype, name='ψ')
dψ = grad(ρ*ψ) + rvec*lift(τi)

problem2 = d3.LBVP([ψ,χ,τi,τo,τχ],namespace=locals())
problem2.add_equation("trace(dψ) + τχ = 0")
problem2.add_equation("div(dψ) + grad(χ) + lift(τo) = - curl(ρ*u)")
problem2.add_equation("angular(ψ(r=ri)) = 0")
problem2.add_equation("angular(ψ(r=ro)) = 0")
problem2.add_equation("χ(r=ri)  = 0")
problem2.add_equation("χ(r=ro)  = 0")
problem2.add_equation("integ(χ) = 0")

solver2 = problem2.build_solver()

#load data
ncase = len(cases)
myrank = CW.rank
size = CW.size
for kk in range(myrank,ncase,size):
    case = cases[kk]
    eig = np.load(prefix + case + '/eigs.npy')
    γ, ω = eig[0].imag, eig[0].real

    file = h5py.File(prefix + case + '/checkpoints/checkpoints_s1.h5')

    growth =   1 / (γ*Ω0)
    period = 2*π / (ω*Ω0)

    if np.abs(period) > 1000*growth:
        period, gap = r'$\mathrm{P}\,$ ' + '= None', ', '
    else:
        period, gap = r'$\mathrm{P}\,$ ' + f'= {period:.0f} days', ',  '

    #growth = f'τ = {growth:.0f} days'
    growth = r'$\mathrm{t}_{\mathrm{e}}$ = ' + f'{growth:.0f} days'
    
    print()
    print(case + ':')
    print(f'γ = {γ:.4f}' + gap + f'ω = {ω:.4f}')
    print(growth + ', ' + period)
    print(50*'-')

    u.load_from_hdf5(file,0)
    b.load_from_hdf5(file,0)

    b['g'] *= b_units/np.max(np.abs(u['g']))
    u['g'] *= u_units/np.max(np.abs(u['g']))
        
    print('solving: curl(a) = b ...')
    solver1.solve()

    print('solving: curl(ρψ) = ρu ...')
    solver2.solve()

    b_conj = dist.VectorField(φθr, bases=basis, dtype=dtype, name='b_conj')

    b.change_scales(fig_scales)
    b_conj.change_scales(fig_scales)

    b_conj['g'] = b['g'].conjugate()
    j, j_conj   = curl(b), curl(b_conj)
    
    H  = 0.5 * volume * (b_conj @ j + b @ j_conj )
    H /= (integ(b_conj @ b) * integ(j_conj @ j))**(1/2)
    H  = H.evaluate()

    φ, θ, r = basis.global_grids(dist, scales=fig_scales)
    for f in [u,b,a,ψ,H]:
        f.change_scales(fig_scales)
    
    δΩ  = (u['g'][0]/(rsun*nHz*r*np.sin(θ)))[0].real

    vals  = np.sort(np.abs(δΩ.flatten()))
    Ω_sat = vals[int(saturation*len(vals))]
    δΩ   /=  Ω_sat

    ψφ = ψ['g'][0,0].real/Ω_sat
    aφ = a['g'][0,0].real/Ω_sat
    bφ = b['g'][0,0].real/Ω_sat
    H  = H['g'][0].real

    fig = plt.figure(figsize=[7.3,3.5],dpi=600,constrained_layout=True,linewidth=2.0)
    data = [δΩ, ψφ, bφ, aφ, H]
    x, y = np.meshgrid(θ,r)

    labels = [r"$\Omega'$",
               r'$\psi$',
               r'$b_{\phi}$',
               r'$a_{\phi}$',
               r'$\mathcal{H}$']

    unit  = [r'$\mathrm{nHz}$',
             r'$\mathrm{\frac{cm}{s}} R_{\!\odot}$',
             r'$\mathrm{G}$',
             r'$\mathrm{G}\,R_{\!\odot}$','']

    cmaps = ['RdYlBu','RdYlGn','PRGn','PiYG','bwr']

    def style(index,value):
            if labels[index] in [r'$\psi$', r'$a_{\phi}$', r'$\mathcal{H}$']:
                return '{:.1f}'.format(value) + unit[index]
            return '{:.0f}'.format(value) + unit[index]

    for ii in range(5):
        pos = [ 0.15 * ii - 0.265, 0.02, 0.92, 0.96 ]
        pb = fig.add_axes(pos,projection='polar')
        pb.set_xticklabels([])
        pb.set_yticklabels([])
        pb.set_xticks([])
        pb.set_yticks([])

        vals = np.sort(np.abs(data[ii].flatten()))
        vmax = vals[int(saturation*len(vals))]
        
        meplot = pb.pcolormesh(x, y, data[ii].T,
                               cmap =  cmaps[ii], vmin = -vmax, vmax =  vmax,
                               rasterized =  False, shading    =  'auto')

        pb.set_theta_direction(-1)
        pb.set_theta_offset(π/2)
        pb.set_thetalim([π,0])
        pb.set_rlim(ri,ro)
        pb.set_rorigin(0)
        pb.set_aspect(1)
        pb.grid(False)
        
        cbax = inset_axes(pb, width = "3%", height = "30%", loc = 'center',
                          bbox_to_anchor = [0.13, 0, 1, 1], bbox_transform = pb.transAxes)
        
        mycbar = fig.colorbar(meplot,cax=cbax,orientation='vertical',ticks=[-vmax,vmax])
        cbax.yaxis.set_ticks_position('left')
            
        cbax.set_yticklabels( [style(ii, -vmax), style(ii, vmax)], fontsize = 9 )
        pb.text(0.37, 0.13, labels[ii], transform = pb.transAxes,  fontsize = 16 )
        
        if ii == 0 :
            pb.text(0.1, 0.55, growth, transform = pb.transAxes, fontsize = 9 )
            pb.text(0.1, 0.48, period, transform = pb.transAxes, fontsize = 9 )
                
    fig.savefig('../png/' + case + '.png', bbox_inches='tight')
