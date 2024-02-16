"""
MRI spherical 2D eigensolver.
"""

import os
import numpy as np
import scipy
import dedalus.public as d3
from dedalus.core import evaluator, subsystems
import logging
logger = logging.getLogger(__name__)

#Inputs:
# nphi, number of phi points
# nth, number of theta points
# nr, number of raidal points
# Re, Reynolds number
# M, Square of the ratio of the magnetic energy to the differential rotation energy.
# neig, desired number of eigenvalues
# guess, target for the sparse solver
# comm, MPI communicator
def MRI_Eigenproblem(target_m, nth,nr,Re,M,neig,guess,comm,label=None):
    # Dedalus Ball Parameters
    if (target_m==0):
        Mmax = 1
    else:
        Mmax = 3*np.abs(target_m)

    # Original parameters
    Lmax = nth
    Nmax = nr

    L_dealias = 3/2 #Dealias by x harmonics
    N_dealias = 3/2 #Dealias by y orders

    #Dedalus Timestepper & Parameters
    dtype = np.complex128

    #Problem Parameters
    ri = 0.89 #Inner radius
    ro = 0.99 #Outer radius

    Rev = Re
    Pm = 1e0
    Rm = Pm*Rev #Magnetic Reynolds number

    Rei = 1e0/Rev
    Rmi = 1e0/Rm
    
    Mag = M
    sMag = np.sqrt(Mag)

    # Set up output dir
    data_dir = 'ModelS_Fit_M{:.2f}_ellmax{:d}_m{:d}'.format(np.log10(Mag),Lmax,target_m)+'_'+label

    #Define radii for coordinate system creators
    radii = (ri,ro)

    mesh = None
    # Coordinates
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    
    # Distributor
    dist = d3.Distributor(coords, mesh=mesh, dtype=dtype,comm=comm)
    
    # Basis
    basis = d3.ShellBasis(coords, (Mmax,Lmax,Nmax), radii=radii, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias)) #, alpha=(0,0))
    basis_k3 = d3.ShellBasis(coords, (Mmax,Lmax,Nmax), radii=radii, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias), k=3)

    # Useful Basis components
    b_inner = basis.S2_basis(radius=ri) #For inner sphere boundary conditions
    b_outer = basis.S2_basis(radius=ro) #For outer sphere boundary conditions
    b_merid = basis.meridional_basis #For 2D NCCs
    b_merid_k3 = basis_k3.meridional_basis
    lift_basis = basis.clone_with(k=1) # First derivative basis

    # Coordinate arrays
    phi, theta, r = dist.local_grids(basis) #Get local coordinate arrays

    # Linear Fields
    uu = dist.VectorField(coords, bases=basis, name='uu') #Velocity
    pu = dist.Field(bases=basis, name='pu') #Pressure gauge
    
    bb = dist.VectorField(coords, bases=basis, name='bb') #Magnetic field
    pb = dist.Field(bases=basis, name='pb') #Magnetic gauge

    #Background fields
    v0 = dist.VectorField(coords, bases=b_merid, name='v0') #Background Velocity
    w0 = dist.VectorField(coords, bases=b_merid, name='w0') #Background Planetary Vorticity
    B0 = dist.VectorField(coords, bases=b_merid_k3, name='B0') #Background Magnetic Field times r^3
    J0 = dist.VectorField(coords, bases=b_merid, name='J0') #Background Magnetic Field times r^3
    er_drho = dist.VectorField(coords, bases=b_merid, name='er_drho') #Radial Unit Vector times drho/dr
    rvec = dist.VectorField(coords, bases=b_merid, name='rvec') #Radial Vector
    eye = d3.Field(dist=dist, bases=(b_merid,b_merid), tensorsig=(coords,coords), dtype=dtype, name='eye') #Identity tensor

    #Non-constant coefficients to help with operator bandwidth
    rho_ncc = dist.Field(bases=b_merid, name='rho') #rho for the continuity equation
    rho_Rei = dist.Field(bases=b_merid, name='rho_Rei') #rho for the various hydro terms

    # Eigenvalue
    om = dist.Field(name='om')

    #Taus for boundaries
    #Inner boundary
    tau_u_ri = dist.VectorField(coords, bases=b_inner, name='tau_u_ri')
    tau_b_ri = dist.VectorField(coords, bases=b_inner, name='tau_b_ri')
    
    #Outer boundary
    tau_u_ro = dist.VectorField(coords, bases=b_outer, name='tau_u_ro')
    tau_b_ro = dist.VectorField(coords, bases=b_outer, name='tau_b_ro')

    #Gauge taus
    tau_pu = dist.Field(name='tau_pu')
    tau_pb = dist.Field(name='tau_pb')

    # Operator shortcuts
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ddt = lambda A: -1j*om*A

    #From jupyter notebook fitting (can update to be more general)... fixed to ri=0.89, ro=0.99 for now.
    rho_cheby_coefs = [0.4287056,-0.50122643,0.07841967,0.00206746,0.00243878] #Good to 1%
    drho_cheby_coefs = [-9.78493705,6.46519649,0.45630896,0.24012609,0.11130384] #Good to 2%
    z = (2*r-ri-ro)/(ro-ri) #Chebyshev ordinate values between -1,1
    rho = np.polynomial.chebyshev.chebval(z,rho_cheby_coefs)
    drho = np.polynomial.chebyshev.chebval(z,drho_cheby_coefs)

    rho_ncc['g'] = rho
    
    #Non-force free magnetic field, but current stable?
    #Normed so that |B0| = 1 at r=1, theta=pi/2
    B0['g'][0] = 0
    B0['g'][1] = -sMag*(4*r**5+ro**5)*np.sin(theta)/r**3/ro**2/5
    B0['g'][2] = 2*sMag*(r**5-ro**5)*np.cos(theta)/r**3/ro**2/5

    J0['g'][0] = -2*sMag*r*np.sin(theta)/ro**2
    J0['g'][1] = 0
    J0['g'][2] = 0

    #Setup background velocity
    z = (r-ri)/(ro-ri) #Scaled radial coordinate

    fz = (1+0.02*z-0.01*z**2-0.03*z**3) #Radial structure in the rotating frame, from fit to solar data
    dfz = (0.02-0.02*z-0.09*z**2)/(ro-ri) #Radial derivative of fz

    gt = 1-0.145*np.cos(theta)**2-0.148*np.cos(theta)**4 #Angular part of the differential rotation
    dgt = 2*np.sin(theta)*(0.145*np.cos(theta) + 2*0.148*np.cos(theta)**3) #theta derivative of gt

    v0['g'][0] = r*np.sin(theta)*(fz*gt-1) #Phi component only, shifted to rotating frame
    v0['g'][1] = 0*r
    v0['g'][2] = 0*r

    #Vorticity associated with v_0 multiplied by Rossby, plus the coriolis force
    #(which cancels out terms from -rsint in the curl of v0), and then multiplied by the nccs to save build time
    df = 2*fz + r*dfz
    w0['g'][0] = 0
    w0['g'][1] = -np.sin(theta)*df*gt
    w0['g'][2] = fz*(2*np.cos(theta)*gt + np.sin(theta)*dgt)

    #Radial unit vector, multiplied by drho for the two necessary terms
    er_drho['g'][0]=0
    er_drho['g'][1]=0
    er_drho['g'][2]=drho

    #Radial vector
    rvec['g'][0] = 0
    rvec['g'][1] = 0
    rvec['g'][2] = r

    #Identity tensor
    eye['g'] = 0
    for i in range(3):
        eye['g'][i,i] = 1

    #Gradients with taus, for first-order formulation
    grad_u = d3.grad(uu) + rvec*lift(tau_u_ri)
    grad_b = d3.grad(bb) + rvec*lift(tau_b_ri)

    # Viscosity tensor, includes taus for the momentum equation
    sigma = grad_u + d3.transpose(grad_u) - (2/3)*d3.trace(grad_u)*eye

    # Velocity boundary conditions, no taus in the grads
    strain_rate = d3.grad(uu) + d3.transpose(d3.grad(uu))

    # EMF for Magnetic boundary conditions (perfectly conducting)
    emf = Rmi*d3.curl(bb) + d3.cross(B0,uu) + d3.cross(bb,v0)

    logger.info("Building problem.")

    # Problem
    problem = d3.EVP([pu, uu, pb, bb, tau_u_ri, tau_b_ri, tau_u_ro, tau_b_ro, tau_pu, tau_pb], eigenvalue=om, namespace=locals())

    #Equations of motion
    problem.add_equation("rho_ncc*trace(grad_u) + dot(er_drho,uu) + tau_pu = 0")
    
    #Vorticity based equation                                                              
    problem.add_equation("rho_ncc*ddt(uu) - rho_ncc*Rei*div(sigma) - dot(er_drho*Rei,sigma) + rho_ncc*grad(pu) + cross(w0*rho_ncc,uu) + cross(curl(uu),v0*rho_ncc) + cross(B0,curl(bb)) + cross(bb,J0) + lift(tau_u_ro) = 0")
    
    #Only aa
    problem.add_equation("ddt(bb) - Rmi*div(grad_b) + grad(pb) + curl(cross(B0,uu)) + curl(cross(bb,v0)) + lift(tau_b_ro) = 0")

    #Coloumb gauge
    problem.add_equation("trace(grad_b) + tau_pb = 0")

    # Gauges
    problem.add_equation("integ(pu) = 0")
    problem.add_equation("integ(pb) = 0")

    #Boundary conditions
    #Lower boundary
    problem.add_equation("radial(uu(r=ri)) = 0")
    problem.add_equation("angular(radial(strain_rate(r=ri),0),0) = 0")
    problem.add_equation("angular(emf(r=ri),0) = 0")
    problem.add_equation("radial(bb(r=ri)) = 0")

    #Upper boundary
    problem.add_equation("radial(uu(r=ro)) = 0")
    problem.add_equation("angular(radial(strain_rate(r=ro),0),0) = 0")
    problem.add_equation("angular(emf(r=ro),0) = 0") 
    problem.add_equation("radial(bb(r=ro)) = 0")    
    
    # Solver
    print('Building solver.')
    solver = problem.build_solver()
    solver.ncc_cutoff = 1e-13
    subproblem = solver.subproblems_by_group[(target_m, None, None)]
    ss = subproblem.subsystems[0]
    
    print('done building solver')
    solver.eigenvalue_subproblem = sp = subproblem
    # Rebuild matrices if directed or not yet built
    #if not hasattr(sp, 'L_min'):
    #    subsystems.build_subproblem_matrices(solver, [sp], ['M', 'L'])

    return solver

