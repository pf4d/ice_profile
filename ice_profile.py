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
dt    = 5.0           # time step
t     = 0.            # begining time
T     = 5000.         # end time
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
H0    = 200.          # thickness at divide
L     = xr - xl       # length of domain

# Unit interval mesh
mesh  = IntervalMesh(500,xl,xr)
cellh = CellSize(mesh)
xcrd  = mesh.coordinates()

# Create FunctionSpace
Q     = FunctionSpace(mesh, "CG", 1)
MQ    = Q * Q

# Boundary conditions:
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1.e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1.e-6

H_bc = DirichletBC(MQ.sub(0), H_MIN, terminus)  # height at terminus
u_bc = DirichletBC(MQ.sub(1), 0.,    divide)    # velocity at divide

# INTIAL CONDITIONS:
# Surface
# This equilibrium profile comes from vanderVeen p. 126, eq 5.50
p0   = 'H0 / pow(n-1,n/(2*n+2)) * pow(( (n+1) * x[0] / L'+\
       '- 1 + n * pow(( 1 - x[0]  / L ),1+1/n) '+\
       '- n *  pow( x[0]/L,1+1/n)),n/(2*n+2))'

# Surface 
zs   = interpolate(Expression(p0,L=L,H0=H0,n=n),Q)
zt   = nan_to_num(zs.vector().array())
zt[where(zt <= H_MIN)[0]] = H_MIN
zs.vector().set_local(zt)

# Bed
zb   = interpolate(Constant(0),Q)

# Thickness
H_i  = project(zs-zb,Q)

# Half width
W    = interpolate(Constant(1000.),Q)

# Initial Velocity
u_i  = interpolate(Constant(0.0),Q) 

adot = Expression('amax * ( .75 - x[0] / L)',L=L,amax=amax)

# variational problem :
h        = Function(MQ)                      # solution
H, u     = split(h)                          # solutions for H, u
h0       = Function(MQ)                      # previous solution
H0, u0   = split(h0)                         # initial value functions

dh       = TrialFunction(MQ)                 # trial function for solution
dH, du   = split(dh)                         # trial functions for H, u
j        = TestFunction(MQ)                  # test function in mixed space
phi, v   = split(j)                          # test functions for H, u

h_i = project(as_vector([H_i,u_i]), MQ)      # project inital values on space
h.vector().set_local(h_i.vector().array())   # initalize H, u in solution
h0.vector().set_local(h_i.vector().array())  # initalize H, u in prev. sol

# Mid-point solution for stabalization
H_mid = 0.5*(H0 + H)
theta = 1#0.878
u_mid = theta*u + (1 - theta)*u0

# SUPG method phihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
phihat = phi + cellh/(2*unorm)*dot(u, phi.dx(0))

# Continuity equation
fH = (H-H0)/dt * phi * dx + ((2*W*u*H_mid).dx(0) - adot) * phihat * dx 

# Momentum balance: weak form of equation 9.65 of vanderveen
fu = - rho * g * H0 * zs.dx(0) * v * dx - \
       mu * (H0 - rho_w / rho * zb)**q * u_mid**p * v * dx + \
       2. * B * H0 * u_mid.dx(0)**(1/n) * v.dx(0) * dx - \
       B * H0 / W * (((n+2) * u_mid)/(2*W))**(1/n) * v * dx

# Momentum balance with regularization on viscosity
#small = 1e-6
#fu    = -rho * g * H_mid * zs.dx(0) * v * dx \
#        -mu * (H_mid - rho_w / rho * zb)**q * u**p * v * dx\
#        +2. * B * H_mid * abs(u.dx(0)+small)**((1-n)/n)*u.dx(0) * v.dx(0) *dx\
#        -B * H_mid / W * (((n+2) * u)/(2*W))**(1/n) * v * dx

f     = fH + fu
df    = derivative(f, h, dh)

# Create non-linear solver instance
problem = NonlinearVariationalProblem(f, h, [H_bc, u_bc], J=df)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 250
prm['newton_solver']['relaxation_parameter'] = 0.8

solver.solve()

# Output file
out_file = File("results/ice_profile.pvd")

# Plot solution
plt.ion()
Hplot = project(H, Q).vector().array()
uplot = project(u, Q).vector().array()

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
hp, = ax1.plot(xcrd, Hplot, 'k', lw=2)
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$H$ [m]')
ax1.set_ylim([0,2000])

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
  solver.solve()

  # Copy solution from previous interval
  h0.assign(h)

  # Plot solution
  Hplot = project(H, Q).vector().array()
  uplot = project(u, Q).vector().array()
  Hplot[where(Hplot < H_MIN)[0]] = H_MIN
  
  # update the dolfin vectors :
  H_i.vector().set_local(Hplot)
  h_new = project(as_vector([H_i, u]), MQ)
  h.vector().set_local(h_new.vector().array())

  hp.set_ydata(Hplot)
  up.set_ydata(uplot)
  plt.draw() 

  # Save the solution to file
  #out_file << (H, t)
  #out_file << (u, t)

  # Move to next interval and adjust boundary condition
  t += dt
  #raw_input("Push enter to contiue")
