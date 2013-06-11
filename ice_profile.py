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
Tm    = 273.15                # tripple point of water ......... [K]
R     = 8.314                 # gas constant ................... [J (mol K)^-1]
A     = 1.00e-17              # temp-dependent ice-flow factor.. [Pa^-n s^-1]
B     = A**(-1/n)             # ice hardeness .................. [Pa s^(1/n)]
amax  = .5 / spy              # max accumlation/ablation rate .. [m s^-1]
mu    = 1e16                  # Basal traction constant
p     = 1.                    # Basal sliding exponent
q     = 1.                    # Basal sliding exponent 
sb    = 0.                    # back stress

### SIMULATION PARAMETERS ###
dt    = 5.000 * spy           # time step ...................... [s]
t     = 0.                    # begining time .................. [s]
tf    = 10000. * spy          # end time ....................... [s]
H_MIN = 1.                    # Minimal ice thickness .......... [m]

### DOMAIN DESCRIPTION ###
xl    = 0.                    # left edge (divide) ............. [m]
xr    = 1500e3                # right edge (margin/terminus) ... [m]
Hd    = 100.                  # thickness at divide ............ [m]
a     = 4/3.
L     = (xr - xl)/a           # length of domain ............... [m]
ela   = 3/4. * L / 1000

# Unit interval mesh
mesh  = IntervalMesh(500,xl,xr)
cellh = CellSize(mesh)
xcrd  = mesh.coordinates()/1000  # divide for units in km.

# Create FunctionSpace
Q     = FunctionSpace(mesh, "CG", 1)
MQ    = MixedFunctionSpace([Q, Q])

# Boundary conditions:
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1e-6

# Dirichlet conditions :
H_bc = DirichletBC(MQ.sub(0), H_MIN, terminus)  # thickness at terminus
u_bc = DirichletBC(MQ.sub(1), 0.,    divide)    # velocity at divide
bcs  = []
#bcs  = [H_bc, u_bc]

# Neumann conditions :
code = 'A * pow(rho*g/4 *(H - rho_w/rho * pow(D, 2) /H - sb/(rho*g)), n)'
gn   = Expression(code, A=A, rho=rho, g=g, H=Hd, rho_w=rho_w, D=0, sb=sb, n=n)

class terminus_velocity(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[0] > xr - DOLFIN_EPS

Gamma_N          = terminus_velocity()
boundary_markers = FacetFunction("uint", mesh)
boundary_markers.set_all(1)
Gamma_N.mark(boundary_markers, 0)
dss = ds[Gamma_N]

# INTIAL CONDITIONS:
# surface :
# This equilibrium profile comes from vanderVeen p. 126, eq 5.50
p0   = 'H / pow(n-1,n/(2*n+2)) * pow(( (n+1) * x[0] / L'+\
       '- 1 + n * pow(( 1 - x[0]  / L ),1+1/n) '+\
       '- n *  pow( x[0]/L,1+1/n)),n/(2*n+2))'
zs   = interpolate(Expression(p0,L=L,H=Hd,n=n),Q)
zt   = nan_to_num(zs.vector().array())
zt[where(zt <= H_MIN)[0]] = H_MIN
zs.vector().set_local(zt)
#zs   = interpolate(Constant(H_MIN),Q)

# bed :
zb   = interpolate(Constant(0.0),Q)

# thickness :
H_i  = project(zs-zb,Q)

# half width :
W    = interpolate(Constant(1000.),Q)

# initial velocity :
u_i  = interpolate(Constant(0.0),Q) 

# accumulation :
adot = Expression('amax * ( .75 - x[0] / L)',L=L,amax=amax)

# variational problem :
U         = Function(MQ)                    # solution
H,u       = split(U)                        # solutions for H, u
U0        = Function(MQ)                    # previous solution
H0,u0     = split(U0)                       # previous solutions for H, u

dU        = TrialFunction(MQ)               # trial function for solution
dH,du     = split(dU)                       # trial functions for H, u
j         = TestFunction(MQ)                # test function in mixed space
phi,psi   = split(j)                        # test functions for H, u

U_i = project(as_vector([H_i,u_i]), MQ)     # project inital values on space
U.vector().set_local(U_i.vector().array())  # initalize H, u in solution
U0.vector().set_local(U_i.vector().array()) # initalize H, u in prev. sol

# SUPG method phihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
phihat = phi + cellh/(2*unorm)*dot(u, phi.dx(0))

# Continuity equation: weak form of eqn. 9.54 of vanderveen
theta = 0.5
H_mid = theta*H + (1 - theta)*H0
fH    = + (H-H0)/dt * phi * dx \
        + 1/(2*W) * (2*H_mid*u*W).dx(0) * phihat * dx \
        - adot * phihat * dx

# SUPG method psihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
psihat = psi + cellh/(2*unorm)*dot(u, psi.dx(0))

# Momentum balance: weak form of equation 9.65 of vanderveen
theta = 1.0
u_mid = theta*u + (1 - theta)*u0
h     = H + zb
fu    = + rho * g * H * h.dx(0) * psi * dx \
        + mu * (H - rho_w / rho * zb)**q * u_mid**p * psi * dx \
        + 2. * B * H * u_mid.dx(0)**(1/n) * psi.dx(0) * dx \
        + B * H / W * (((n+2) * u_mid)/(2*W))**(1/n) * psi * dx

f     = fH + fu
df    = derivative(f, U, dU)

# Create non-linear solver instance
problem = NonlinearVariationalProblem(f, U, bcs, J=df)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 100
prm['newton_solver']['relaxation_parameter'] = 0.8

solver.solve()

# Output file
out_file = File("results/ice_profile.pvd")

# Plot solution
gry = '0.4'
red = '#5f4300'
pur = '#3d0057'
clr = pur

plt.ion()
Hplot = project(H, Q).vector().array()
uplot = project(u, Q).vector().array()

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
ax3.set_xlim([xl/1000, xr/1000])

hp, = ax1.plot(xcrd, Hplot, 'k', lw=2)
ax1.set_xlabel('$x$ [km]')
ax1.set_ylabel('$H$ [m]')
ax1.set_ylim([-100,1000])
ax1.set_xlim([xl/1000, xr/1000])
ax1.grid()

ax2 = ax1.twinx()
ax2.axvline(x=ela, lw=2, color = gry)
up, = ax2.plot(xcrd, uplot, clr, lw=2)
ax2.set_ylabel('$u$ [m/a]', color=clr)
ax2.set_xlim([xl/1000, xr/1000])
ax2.grid()
for tl in ax2.get_yticklabels():
  tl.set_color(clr)

fig_text = plt.figtext(.80,.95,'Time = 0.0 yr')

plt.draw()

# Time-stepping
while t < tf:
  # Solve the nonlinear system 
  solver.solve()

  # Plot solution
  Hplot = project(H, Q).vector().array()
  uplot = project(u, Q).vector().array()
  Hplot[where(Hplot < H_MIN)[0]] = H_MIN
  uplot[where(Hplot < H_MIN)[0]] = 0.0

  # update the dolfin vectors :
  H_i.vector().set_local(Hplot)
  u_i.vector().set_local(uplot)
  U_new = project(as_vector([H_i, u_i]), MQ)
  U.vector().set_local(U_new.vector().array())

  # Copy solution from previous interval
  U0.assign(U)

  hp.set_ydata(Hplot)
  up.set_ydata(uplot)
  fig_text.set_text('Time = %.0f yr' % (t/spy)) 
  plt.draw() 

  # Move to next interval and adjust boundary condition
  t += dt
