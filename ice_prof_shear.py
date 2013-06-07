#!/usr/bin/python
from numpy import *
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pylab import mpl
from dolfin import *

class FixedOrderFormatter(ScalarFormatter):
  """
  Formats axis ticks using scientific notation with a constant order of 
  magnitude
  """
  def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
    self._order_of_mag = order_of_mag
    ScalarFormatter.__init__(self, useOffset=useOffset, 
                             useMathText=useMathText)
  def _set_orderOfMagnitude(self, range):
    """
    Over-riding this to avoid having orderOfMagnitude reset elsewhere
    """
    self.orderOfMagnitude = self._order_of_mag

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

### SIMULATION PARAMETERS ###
dt    = 5.000         # time step
t     = 0.            # begining time
tf    = 200#50000.        # end time
H_MIN = 1.            # Minimal ice thickness

### PHYSICAL CONSTANTS ###
spy   = 31556926.     # seconds per year
rho   = 911.          # density of ice (kg/m^3)
rho_w = 1000.         # density of water (kg/m^3)
g     = 9.81 * spy**2 # gravitation acceleration (m/yr^2)
n     = 1.            # flow law exponent
B     = 750.e3        # flow law temperature sensitivity factor (Pa*yr^.333)
amax  = .5            # max accumlation/ablation rate
mu    = 1.e16         # Basal traction constant
p     = 1.            # Basal sliding exponent
q     = 1.            # Basal sliding exponent 
sb    = 0.            # back stress
A     = B**-n         # 

### DOMAIN DESCRIPTION ###
xl    = 0.            # left edge (divide)
xr    = 1500.e3       # right edge (margin/terminus)
H0    = 10.           # thickness at divide
a     = 1#4/3.
L     = (xr - xl)/a   # length of domain
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
u_bc = DirichletBC(MQ.sub(1), 0.,    divide)    # velocity at divide
bcs  = []
bcs  = [H_bc, u_bc]

# Neumann conditions :
code = 'A * pow(rho*g/4 *(H - rho_w/rho * pow(D, 2) /H - sb/(rho*g)), n)'
gn   = Expression(code, A=A, rho=rho, g=g, H=H0, rho_w=rho_w, D=0, sb=sb, n=n)

boundary_markers = FacetFunction("uint", mesh)

class terminus_velocity(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and x[0] > xr - 1e-6

Gamma_N = terminus_velocity()
Gamma_N.mark(boundary_markers, 4)

# INTIAL CONDITIONS:
# surface :
# This equilibrium profile comes from vanderVeen p. 126, eq 5.50
p0   = 'H0 / pow(n-1,n/(2*n+2)) * pow(( (n+1) * x[0] / L'+\
       '- 1 + n * pow(( 1 - x[0]  / L ),1+1/n) '+\
       '- n *  pow( x[0]/L,1+1/n)),n/(2*n+2))'
zs   = interpolate(Expression(p0,L=L,H0=H0,n=n),Q)
zt   = nan_to_num(zs.vector().array())
zt[where(zt <= H_MIN)[0]] = H_MIN
zs.vector().set_local(zt)
zs   = interpolate(Constant(H0),Q)

# bed :
zb   = interpolate(Constant(0.0),Q)

# thickness :
H_i  = project(zs-zb,Q)

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
        + (H_mid*u).dx(0) * phihat * dx \
        - adot * phihat * dx

# SUPG method psihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
psihat = psi + cellh/(2*unorm)*dot(u, psi.dx(0))

# Momentum balance: weak form of equation 9.65 of vanderveen
theta = 1.0
u_mid = theta*u + (1 - theta)*u0
h     = H0 + zb
fu    = + u * psi * dx \
        - 2 * A * H / (n+2) * (-rho * g * H * h.dx(0))**n * psi * dx

f     = fH + fu
df    = derivative(f, U, dU)

# Create non-linear solver instance
problem = NonlinearVariationalProblem(f, U, bcs, J=df)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 25
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
#ax1.xaxis.set_major_formatter(FixedOrderFormatter(4))
#ax3.xaxis.set_major_formatter(FixedOrderFormatter(4))

plt.draw()

# Time-stepping
while t < tf:
  # Assemble vector and apply boundary conditions
  #b = assemble(LH)
  #H_bc.apply(b)

  # Solve the nonlinear system 
  solver.solve()

  # Plot solution
  Hplot = project(H, Q).vector().array()
  uplot = project(u, Q).vector().array()
  #Hplot[where(Hplot < H_MIN)[0]] = H_MIN
  #uplot[where(Hplot < H_MIN)[0]] = 0.0

  # update the dolfin vectors :
  H_i.vector().set_local(Hplot)
  u_i.vector().set_local(uplot)
  U_new = project(as_vector([H_i, u_i]), MQ)
  #U.vector().set_local(U_new.vector().array())

  # Copy solution from previous interval
  U0.assign(U)

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
