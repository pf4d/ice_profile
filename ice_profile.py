#!/usr/bin/python
from numpy import *
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
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
dt    = .01           # time step
t     = 0.            # begining time
T     = 50000.        # end time
H_MIN = 1.            # Minimal ice thickness

### PHYSICAL CONSTANTS ###
spy   = 31556926.     # seconds per year
rho   = 911.          # density of ice (kg/m^3)
rho_w = 1000.         # density of water (kg/m^3)
g     = 9.81 * spy**2 # gravitation acceleration (m/yr^2)
n     = 3             # flow law exponent
B     = 750.e3        # flow law temperature sensitivity factor (Pa*yr^.333)
amax  = .5            # max accumlation/ablation rate
mu    = 1.e16         # Basal traction constant
p     = 1.            # Basal sliding exponent
q     = 1.            # Basal sliding exponent 

### DOMAIN DESCRIPTION ###
xl    = 0.            # left edge (divide)
xr    = 1000.e3       # right edge (margin/terminus)
H0    = 20.           # thickness at divide
L     = xr - xl       # length of domain

# Unit interval mesh
mesh  = IntervalMesh(50,xl,xr)
h     = CellSize(mesh)
xcrd  = mesh.coordinates()

# Create FunctionSpace
Q     = FunctionSpace(mesh, "CG", 1)

# Boundary conditions:
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1.e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1.e-6

u_bc = DirichletBC(Q,0.,divide)
H_bc = DirichletBC(Q,H_MIN,terminus)

# INTIAL CONDITIONS:
# Surface
# This equilibrium profile comes from vanderVeen p. 126, eq 5.50
p0   = 'H0 / pow(n-1,n/(2*n+2)) * pow(( (n+1) * x[0] / L'+\
       '- 1 + n * pow(( 1 - x[0]  / L ),1+1/n) '+\
       '- n *  pow( x[0]/L,1+1/n)),n/(2*n+2))'
zs   = interpolate(Expression(p0,L=L,H0=H0,n=n),Q)
# Bed
zb   = interpolate(Constant(0),Q)
# Thickness
H0   = project(zs-zb,Q)

# Half width
W    = interpolate(Constant(1000.),Q)

# Initial Velocity
u0   = interpolate(Constant(0),Q) 

adot = Expression('amax * ( .75 - x[0] / L)',L=L,amax=amax)

# Test and trial functions
u, v   = TrialFunction(Q), TestFunction(Q)   # for velocity
H, phi = TrialFunction(Q), TestFunction(Q)   # for thickness
u_     = Function(Q)                         # most recent velocity solution
H_     = Function(Q)

# Mid-point solution for stabalization
H_mid = 0.5*(H0 + H_)
u_mid = 0.5*(u0 + u_)

# Galerkin variational problem
# Continuity equation
FH = (H-H0) * phi * dx + ((2*W*u_mid*H).dx(0) - adot) * phi * dt * dx 

# Momentum balance: weak form of equation xx of vanderveen
#Fu = -rho * g * H0 * zs.dx(0) * v * dx \
#     -mu * (H0 - rho_w / rho * zb)**q * u**p * v * dx\
#     +2. * B * H0 * u.dx(0)**(1/n) * v.dx(0) *dx\
#     -B * H0 / W * (((n+2) * u)/(2*W))**(1/n) * v * dx

# Momentum balance with regularization on viscosity
small = 1e-6
Fu    = -rho * g * H_mid * zs.dx(0) * v * dx \
        -mu * (H_mid - rho_w / rho * zb)**q * u**p * v * dx\
        +2. * B * H_mid * abs(u.dx(0)+small)**((1-n)/n)*u.dx(0) * v.dx(0) *dx\
        -B * H_mid / W * (((n+2) * u)/(2*W))**(1/n) * v * dx

Fu    = action(Fu,u_)          # Linearize about the most recent solution, u_
J     = derivative(Fu, u_, u)  # Gateaux derivative of Fu at u_ in dir. of u

# Residual for stabization of H only
r     = H-H0 + dt * (H * u_mid).dx(0) - adot * dt

# Add SUPG stabilisation terms
unorm = sqrt(dot(u_mid, u_mid))
FH   += (h/(2.0*unorm))*dot(u_mid, u_mid.dx(0)) * r * dx

# Create bilinear and linear forms
aH = lhs(FH)
LH = rhs(FH)

# Assemble matrix
#AH = assemble(aH)
#H_bc.apply(AH)

# Create linear solver and factorize matrix
#H_solver = LUSolver(AH)
#H_solver.parameters["reuse_factorization"] = False

H_problem = LinearVariationalProblem(aH,LH,H_,H_bc)
H_solver  = LinearVariationalSolver(H_problem)

# Create non-linear solver instance
u_problem = NonlinearVariationalProblem(Fu, u_, u_bc, J)
u_solver  = NonlinearVariationalSolver(u_problem)

prm = u_solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 250
prm['newton_solver']['relaxation_parameter'] = 0.8

u_solver.solve()
H_solver.solve()

# Output file
out_file = File("results/ice_profile.pvd")

# Set intial condition
#u0 = u_ 
H  = H0

# Plot solution
plt.ion()
Hplot = project(H,  Q).vector().array()
uplot = project(u_, Q).vector().array()

fig = plt.figure()
ax1 = fig.add_subplot(111)
hp, = ax1.plot(xcrd, Hplot, 'k', lw=2)
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$H$ [m]')
ax1.set_ylim([0,20])

ax2 = ax1.twinx()
gry = '0.4'
up, = ax2.plot(xcrd, uplot, gry, lw=2)
ax2.set_ylabel('$u$ [m/s]', color=gry)
for tl in ax2.get_yticklabels():
  tl.set_color(gry)

ax1.grid()
ax1.xaxis.set_major_formatter(FixedOrderFormatter(4))

plt.draw()

raw_input("Press enter...")

# Time-stepping
while t < T:
  # Assemble vector and apply boundary conditions
  #b = assemble(LH)
  #H_bc.apply(b)

  # Solve the nonlinear system 
  u_solver.solve()

  # Solve the linear system 
  #H_solver.solve(H.vector(),b)
  H_solver.solve()

  # Copy solution from previous interval
  H0 = H_ 
  u0 = u_

  # Plot solution
  Hplot = project(H,  Q).vector().array()
  uplot = project(u_, Q).vector().array()
  hp.set_ydata(Hplot)
  up.set_ydata(uplot)
  plt.draw() 

  # Save the solution to file
  out_file << (H_, t)
  out_file << (u_, t)

  # Move to next interval and adjust boundary condition
  t += dt
  #raw_input("Push enter to contiue")
