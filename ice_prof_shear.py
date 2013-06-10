#!/usr/bin/python
from numpy import *
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pylab import mpl
from dolfin import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

### PHYSICAL CONSTANTS ###
spy   = 31556926.             # seconds per year ............... [s]
rho   = 911.                  # density of ice ................. [kg m^-3]
rho_w = 1000.                 # density of water ............... [kg m^-3]
g     = 9.81                  # gravitation acceleration ....... [m s^-2]
n     = 3.                    # flow law exponent
A     = 4.529e-24             # temp-dependent ice-flow factor.. [Pa^-n s^-1]
B     = A**(-1/n)             # ice hardeness .................. [Pa s^(1/n)]
amax  = .5 / spy              # max accumlation/ablation rate .. [m s^-1]

### SIMULATION PARAMETERS ###
dt    = 5.000 * spy           # time step ...................... [s]
t     = 0.                    # begining time .................. [s]
tf    = 2000. * spy            # end time ....................... [s]
H_MIN = 1.                    # Minimal ice thickness .......... [m]

### DOMAIN DESCRIPTION ###
xl    = 0.                    # left edge (divide) ............. [m]
xr    = 1500e3                # right edge (margin/terminus) ... [m]
Hd    = 100.                  # thickness at divide ............ [m]
a     = 1#4/3.
L     = (xr - xl)/a           # length of domain ............... [m]
ela   = 3/4. * L / 1000

# Unit interval mesh
mesh  = IntervalMesh(500,xl,xr)
cellh = CellSize(mesh)
xcrd  = mesh.coordinates()/1000  # divide for units in km.

# Create FunctionSpace
Q     = FunctionSpace(mesh, "CG", 1)

# Boundary conditions:
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1.e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1.e-6

# Dirichlet conditions :
H_bc = DirichletBC(Q, H_MIN, terminus)  # thickness at terminus

# INTIAL CONDITIONS:
# surface :
# This equilibrium profile comes from vanderVeen p. 126, eq 5.50
p0   = 'Hd / pow(n-1,n/(2*n+2)) * pow(( (n+1) * x[0] / L'+\
       '- 1 + n * pow(( 1 - x[0]  / L ),1+1/n) '+\
       '- n *  pow( x[0]/L,1+1/n)),n/(2*n+2))'
zs   = interpolate(Expression(p0,L=L,Hd=Hd,n=n),Q)
zt   = nan_to_num(zs.vector().array())
zt[where(zt <= H_MIN)[0]] = H_MIN
zs.vector().set_local(zt)
#zs   = interpolate(Constant(Hd),Q)

# bed :
zb   = interpolate(Constant(0.0),Q)

# thickness :
H_i  = project(zs-zb,Q)

# accumulation :
adot = Expression('amax * ( .75 - x[0] / L)',L=L,amax=amax)

# variational problem :
H         = Function(Q)       # solution
H0        = Function(Q)       # previous solution

dH        = TrialFunction(Q)  # trial function for solution
phi       = TestFunction(Q)   # test function 

H.assign(H_i)                 # initalize H in solution
H0.assign(H_i)                # initalize H in prev. sol

# SUPG method phihat :        
Hnorm  = sqrt(dot(H, H) + 1e-10)
phihat = phi + cellh/(2*Hnorm)*dot(H, phi.dx(0))

# Continuity equation: weak form of eqn. 9.54 of vanderveen
theta = 1.0
H_mid = theta*H + (1 - theta)*H0
h     = H_mid + zb
D     = 2*A/(n+2) * (rho*g)**n * H_mid**(n+2) * h.dx(0)**(n-1)
fH    = + (H-H0)/dt * phi * dx \
        + D * inner(h.dx(0), phi.dx(0)) * dx \
        - adot * phi * dx

df    = derivative(fH, H, dH)

# Create non-linear solver instance
problem = NonlinearVariationalProblem(fH, H, H_bc, J=df)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 25
prm['newton_solver']['relaxation_parameter'] = 0.8

solver.solve()

# Plot solution
gry = '0.4'
red = '#5f4300'
pur = '#3d0057'
clr = pur

plt.ion()
Hplot = H.vector().array()

fig = plt.figure(figsize=(10,7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])

# plot the accumulation
adotPlot = project(adot, Q).vector().array() * spy
ax3.axhline(lw=2, color = gry)
ax3.axvline(x=ela, lw=2, color = gry)
ax3.plot(xcrd, adotPlot, 'r', lw=2)
ax3.grid()
ax3.set_xlabel('$x$ [km]')
ax3.set_ylabel('$\dot{a}$ [m/a]')

hp, = ax1.plot(xcrd, Hplot, 'k', lw=2)
ax1.set_xlabel('$x$ [km]')
ax1.set_ylabel('$H$ [m]')
ax1.set_ylim([-100,1000])

fig_text = plt.figtext(.80,.95,'Time = 0.0 yr')

ax1.grid()

plt.draw()

# Time-stepping
while t <= tf:
  # Solve the nonlinear system 
  solver.solve()

  # Plot solution
  Hplot = H.vector().array()
  Hplot[where(Hplot < H_MIN)[0]] = H_MIN

  # update the dolfin vectors :
  H_i.vector().set_local(Hplot)
  H.assign(H_i)

  # Copy solution from previous interval
  H0.assign(H)

  hp.set_ydata(Hplot)
  fig_text.set_text('Time = %.0f yr' % (t/spy))
  plt.draw() 

  # Move to next interval and adjust boundary condition
  t += dt
