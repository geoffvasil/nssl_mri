"""
Dedalus script computing the critical magnetic field strength and eigenmode for the MRI
in a local cylindrical approximation to the solar equator.

We use a Legendre basis to solve the EVP:

    s * W * Ψ  + V * Ψ - dr(dr(Ψ)) = 0
    Ψ(r=r0) = 0
    Ψ(r=r1) = 0

where

    W = 2 * r * Ω0 * dr(Ω0)* μ0 * ρ0 / B0**2,
    V = 3/(4*r**2) + r * ρ0 * dr( dr(B0)/(r*ρ0) ) / B0,

and s = 1/Bmax**2 is the eigenvalue and

The formulation uses two tau terms, in equivalent first-order formulation.

To run and plot:
    $ python3 mri_critical_Bmax.py
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters

basis = d3.Legendre
Nr    = 64
(r0, r1) = (0.89, 0.99)

radius   = 696 * 10**(6+2) # cm
rotation = 2*np.pi * 466 * 10**(-9) # radian/second

figs = True

# Bases
rcoord = d3.Coordinate('r')
dist   = d3.Distributor(rcoord, dtype=np.complex128)
rbasis = basis(rcoord, size=Nr, bounds=(r0, r1))

# Substitutions
dr   = lambda A: d3.Differentiate(A, rcoord)
lift = lambda A: d3.Lift(A, rbasis.derivative_basis(1), -1)

#Setup background
def ρ0(r):
    z = (r-r0)/(r1-r0)
    a = [0.031256,0.053193,0.033703,0.023766,0.012326]
    return sum(a[k]*(-z)**k for k in range(len(a)))

def Ω0(r):
    z = (r-r0)/(r1-r0)
    return 1 + 0.02*z - 0.01*z**2 - 0.03*z**3

def B0(r):
    return (1 + 4*(r/r1)**5)/(5*(r/r1)**3)

r      = dist.Field(bases=rbasis, name='r')
r['g'] = dist.local_grid(rbasis)

μ0  =  4 * np.pi
μ0 *= radius**2
μ0 *= rotation**2
μ0 *= 1e-6 # kG

W = ( μ0 * ρ0(r) * r * dr(Ω0(r)**2) / B0(r)**2 ).evaluate()
W['c'][10:] *= 0

V  = dr(B0(r))
V /= r * ρ0(r)
V  = dr(V)
V *= r * ρ0(r) / B0(r)
V += 3/(4*r**2)
V = V.evaluate()
V['c'][20:] *= 0

# Fields
Ψ  = dist.Field(name='Ψ', bases=rbasis)
τ0 = dist.Field(name='τ0')
τ1 = dist.Field(name='τ1')
Ψrr  = dr(dr(Ψ) + lift(τ0)) + lift(τ1)

s  = dist.Field(name='s')

# Problem
problem = d3.EVP([Ψ, τ0, τ1], eigenvalue=s, namespace=locals())
problem.add_equation("s * W * Ψ  + V * Ψ - Ψrr = 0")
problem.add_equation("Ψ(r=r0) = 0")
problem.add_equation("Ψ(r=r1) = 0")

# Solve
solver = problem.build_solver()
solver.solve_dense(solver.subproblems[0])
eigenvalues = solver.eigenvalues.real
evals = np.sort(eigenvalues[eigenvalues>0])
idx=np.where(eigenvalues == evals[0])[0][0]

plt.figure(figsize=(6, 4))
solver.set_state(idx, solver.subsystems[0])

Bmax = int(1000 / np.sqrt(eigenvalues[idx])) # Gauss

rg = r['g'].real
Ψg = (Ψ['g'] / Ψ['g'][1]).real
ξr = Ψg/np.sqrt(rg)/B0(rg)
plt.plot(rg, ξr/np.max(np.abs(ξr)),lw=3)

plt.xlim(r0, r1)
plt.ylabel(r"radial displacement, $\xi_{\,r\,}(r\,)$")
plt.xlabel(r"radius, $r\,/\,R_{\odot}$")
plt.title(f"$B_\max$ = {Bmax} G")
plt.tight_layout()

if figs:
    plt.savefig("mri_critical_mode.pdf")
    plt.savefig("mri_critical_mode.png", dpi=200)
