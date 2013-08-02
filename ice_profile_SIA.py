#!/usr/bin/python
from numpy import *
import numpy as np
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
Tm    = 273.15                # triple point of water .......... [K]
R     = 8.314                 # gas constant ................... [J (mol K)^-1]
c     = 1.73e3
A     = c*exp(-13.9e4/(R*Tm)) # temp-dependent ice-flow factor.. [Pa^-n s^-1]
B     = A**(-1/n)             # ice hardeness .................. [Pa s^(1/n)]
amax  = .5 / spy              # max accumlation u............... [m s^-1]

### SIMULATION PARAMETERS ###
dt    = 50.00 * spy           # time step ...................... [s]
t     = 0.                    # begining time .................. [s]
tf    = 100000. * spy         # end time ....................... [s]
H_MIN = 0.                    # Minimal ice thickness .......... [m]
H_MAX = 5000.                 # Maximum plot height ............ [m]
D_MAX = 1000.                 # maximum depth of bed ........... [m]

### DOMAIN DESCRIPTION ###
xl    = 0.                    # left edge (divide) ............. [m]
xr    = 1500e3                # right edge (margin/terminus) ... [m]
Hd    = 100.0                 # thickness at divide ............ [m]
c     = 1/4.                  # percent of accumulation range .. [%]
L     = c * (xr - xl)         # length of domain ............... [m]
ela   = L / 1000

# Unit interval mesh
mesh  = IntervalMesh(500,xl,xr)
xcrd  = mesh.coordinates()/1000  # divide for units in km.

# Create FunctionSpace
Q     = FunctionSpace(mesh, "CG", 1)

# Boundary conditions:
def divide(x, on_boundary):
  return on_boundary and x[0] < xl + 1.e-6

def terminus(x, on_boundary):
  return on_boundary and x[0] > xr - 1.e-6

# Dirichlet conditions :
H_bc = DirichletBC(Q, H_MIN, terminus)  # thickness at terminus

# INTIAL CONDITIONS:
# surface :
code = '729 - 2184.8  * pow(x[0] / 750000, 2) ' \
       '    + 1031.72 * pow(x[0] / 750000, 4) ' \
       '    - 151.72  * pow(x[0] / 750000, 6) '
zs   = interpolate(Constant(Hd),Q)
zs   = interpolate(Expression("H0 - m * x[0]",m=1e-4, H0=H_MIN),Q)
zs   = interpolate(Expression("H0 + " + code, H0=H_MIN),Q)

# bed :
zb   = interpolate(Constant(0.0),Q)
zb   = interpolate(Expression("- m * x[0]",m=1e-4),Q)
zb   = interpolate(Expression(code), Q)

# thickness :
H_i  = project(zs-zb,Q)

# accumulation :
adot = Constant(0.3 / spy)
adot = Expression('amax * (1 - x[0] / L)',L=L,amax=amax)

# variational problem :
H         = Function(Q)       # solution
H0        = Function(Q)       # previous solution

dH        = TrialFunction(Q)  # trial function for solution
phi       = TestFunction(Q)   # test function 

H.assign(H_i)                 # initalize H in solution
H0.assign(H_i)                # initalize H in prev. sol

# Continuity equation: weak form of eqn. 9.54 of vanderveen
theta = 0.5
H_mid = theta*H + (1 - theta)*H0
h     = H_mid + zb
D     = 2*A/(n+2) * (rho*g)**n * H_mid**(n+2) * h.dx(0)**(n-1)
gn    = Expression("-0.0")
fH    = + (H-H0)/dt * phi * dx \
        + D * inner(h.dx(0), phi.dx(0)) * dx \
        - D * gn * phi * ds \
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
clr = gry

plt.ion()

fig = plt.figure(figsize=(10,7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])
#ax2 = ax1.twinx()

# plot the accumulation
adotPlot = project(adot, Q).vector().array() * spy
ax3.axhline(lw=2, color = gry)
ax3.axvline(x=ela, lw=2, color = gry)
ax3.plot(xcrd, adotPlot, 'r', lw=2)
ax3.set_xlabel('$x$ [km]')
ax3.set_ylabel('$\dot{a}$ [m/a]')
ax3.set_xlim([xl/1000, xr/1000])
ax3.grid()

zbPlot = project(zb, Q).vector().array()
hplot  = project((H + zb), Q).vector().array()
zbp,   = ax1.plot(xcrd, zbPlot, pur, lw=2)
hp,    = ax1.plot(xcrd, hplot, 'k',  lw=2)
ax1.axvline(x=ela, lw=2, color = gry)
ax1.plot(xcrd, [H_MAX] * len(xcrd), 'r+')
ax1.set_xlabel('$x$ [km]')
ax1.set_ylabel('$h$ [m]')
ax1.set_ylim([-D_MAX,H_MAX])
ax1.set_xlim([xl/1000, xr/1000])

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
  hplot = Hplot + zb.vector().array() 

  # update the dolfin vectors :
  H_i.vector().set_local(Hplot)
  H.assign(H_i)

  # Copy solution from previous interval
  H0.assign(H)

  hp.set_ydata(hplot)
  fig_text.set_text('Time = %.0f yr' % (t/spy))
  plt.draw() 

  # Move to next interval and adjust boundary condition
  t += dt
