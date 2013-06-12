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
c     = 1.73e3
A     = c*exp(-13.9e4/(R*Tm)) # temp-dependent ice-flow factor.. [Pa^-n s^-1]
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
  return on_boundary and x[0] < xl + 1.e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1.e-6

# Dirichlet conditions :
H_bc = DirichletBC(MQ.sub(0), H_MIN, terminus)  # thickness at terminus
u_bc = DirichletBC(MQ.sub(1), 0.0,   divide)    # velocity at divide
bcs  = []
bcs  = [H_bc, u_bc]

# Neumann conditions :
code = 'A * pow(rho*g/4 *(H - rho_w/rho * pow(D, 2) /H - sb/(rho*g)), n)'
gn   = Expression(code, A=A, rho=rho, g=g, H=Hd, rho_w=rho_w, D=0, sb=sb, n=n)

boundary_markers = FacetFunction("uint", mesh)

class terminus_velocity(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[0] > xr - 1e-6

Gamma_N = terminus_velocity()
Gamma_N.mark(boundary_markers, 4)

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
zs   = interpolate(Constant(Hd),Q)

# bed :
zb   = interpolate(Constant(0.0),Q)

# initial thickness :
H0   = project(zs-zb,Q)

# half width :
W    = interpolate(Constant(1000000.),Q)

# initial velocity :
u0   = interpolate(Constant(0.0),Q) 

# accumulation :
adot = Expression('amax * ( .75 - x[0] / L)',L=L,amax=amax)

# variational problem :
# Test and trial functions
H, phi = TrialFunction(Q), TestFunction(Q)   # for thickness
u, psi = TrialFunction(Q), TestFunction(Q)   # for velocity
u_     = Function(Q)
H_     = Function(Q)

# SUPG method phihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
phihat = phi + cellh/(2*unorm)*dot(u, phi.dx(0))

# Continuity equation: weak form of eqn. 9.54 of vanderveen
theta = 0.5
H_mid = theta*H + (1 - theta)*H0
fH    = + (H-H0)/dt * phi * dx \
        + 1/(2*W) * (2*H_mid*u*W).dx(0) * phi * dx \
        - adot * phi * dx

# Create bilinear and linear forms
aH = lhs(fH)
LH = rhs(fH)

# SUPG method psihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
psihat = psi + cellh/(2*unorm)*dot(u, psi.dx(0))

# Momentum balance: weak form of equation 9.65 of vanderveen
theta = 1.0
u_mid = theta*u + (1 - theta)*u0
h     = H0 + zb
fu    = + rho * g * H * h.dx(0) * psi * dx \
        + mu * (H0 - rho_w / rho * zb)**q * u_mid**p * psi * dx \
        - 2. * B * H * H.dx(0) * u_mid.dx(0)**(1/n) * psi * dx \
        + 2. * B * H/n * u_mid.dx(0)**(1/n - 1) * u_mid.dx(0) * psi.dx(0) * dx \
        - 2. * B * H/n * u_mid.dx(0)**(1/n - 1) * gn * psi * ds \
        + B * H / W * (((n+2) * u_mid)/(2*W))**(1/n) * psi * dx

fu    = action(fu, u_)
df    = derivative(fu, u_, u)

# Create linear solver instance
H_problem = LinearVariationalProblem(aH, LH, H_, H_bc)
H_solver  = LinearVariationalSolver(H_problem)

# Create non-linear solver instance
u_problem = NonlinearVariationalProblem(fu, u_, u_bc, J=df)
u_solver  = NonlinearVariationalSolver(problem)

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
adotPlot = project(adot, Q).vector().array()
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

ax2 = ax1.twinx()
ax2.axvline(x=ela, lw=2, color = gry)
up, = ax2.plot(xcrd, uplot, clr, lw=2)
ax2.set_ylabel('$u$ [m/a]', color=clr)
ax2.grid()
for tl in ax2.get_yticklabels():
  tl.set_color(clr)

fig_text = plt.figtext(.80,.95,'Time = 0.0 yr')

ax1.grid()

plt.draw()

# Time-stepping
while t < tf:
  # Assemble vector and apply boundary conditions
  #b = assemble(LH)
  #H_bc.apply(b)

  # Solve the system 
  u_solver.solve()
  H_solver.solve()

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
  H0 = H_
  u0 = u_

  hp.set_ydata(Hplot)
  up.set_ydata(uplot)
  fig_text.set_text('Time = %.0f yr' % t) 
  plt.draw() 

  # Save the solution to file
  #out_file << (H, t)
  #out_file << (u, t)

  # Move to next interval and adjust boundary condition
  t += dt
  #raw_input("Push enter to contiue")
