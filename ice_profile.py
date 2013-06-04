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
sb    = 0.            # back stress
A     = B**-n         # 

### DOMAIN DESCRIPTION ###
xl    = 0.            # left edge (divide)
xr    = 1500.e3       # right edge (margin/terminus)
H0    = 200.          # thickness at divide
a     = 4/3.
L     = (xr - xl)/a   # length of domain
ela   = 3/4. * L / 1000

# Unit interval mesh
mesh  = IntervalMesh(500,xl,xr)
cellh = CellSize(mesh)
xcrd  = mesh.coordinates()/1000  # divide for units in km.

# Create FunctionSpace
Q     = FunctionSpace(mesh, "CG", 1)
MQ    = Q * Q

# Boundary conditions:
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1.e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1.e-6

# Dirichlet conditions :
H_bc = DirichletBC(MQ.sub(0), H_MIN, terminus)  # height at terminus
u_bc = DirichletBC(MQ.sub(1), 0.,    divide)    # velocity at divide

# Neumann conditions :
code = 'A * pow(rho*g/4 *(H - rho_w/rho * pow(D, 2) /H - sb/(rho*g)), n)'
gn   = Expression(code, A=A, rho=rho, g=g, H=H0, rho_w=rho_w, D=0, sb=sb, n=n)

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

# bed :
zb   = interpolate(Constant(0),Q)

# thickness :
H_i  = project(zs-zb,Q)

# half width :
W    = interpolate(Constant(1000.),Q)

# initial velocity :
u_i  = interpolate(Constant(0.0),Q) 

# accumulation :
adot = Expression('amax * ( .75 - x[0] / L)',L=L,amax=amax)

# variational problem :
h        = Function(MQ)                      # solution
H, u     = split(h)                          # solutions for H, u
h0       = Function(MQ)                      # previous solution
H0, u0   = split(h0)                         # previous solutions for H, u

dh       = TrialFunction(MQ)                 # trial function for solution
dH, du   = split(dh)                         # trial functions for H, u
j        = TestFunction(MQ)                  # test function in mixed space
phi, v   = split(j)                          # test functions for H, u

h_i = project(as_vector([H_i,u_i]), MQ)      # project inital values on space
h.vector().set_local(h_i.vector().array())   # initalize H, u in solution
h0.vector().set_local(h_i.vector().array())  # initalize H, u in prev. sol

# SUPG method phihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
phihat = phi + cellh/(2*unorm)*dot(u, phi.dx(0))

# Continuity equation: weak form of eqn. 9.54 of vanderveen
theta = 0.5
H_mid = theta*H + (1 - theta)*H0
fH    = (H-H0)/dt * phi * dx + \
        1/(2*W) * (2*H_mid*u*W).dx(0) * phihat * dx - \
        adot * phihat * dx

# Momentum balance: weak form of equation 9.65 of vanderveen
theta = 1
u_mid = theta*u + (1 - theta)*u0
#          2. * B * H/n * u_mid.dx(0)**(1/n - 1) * gn * v * ds - \
fu    = - rho * g * H * zs.dx(0) * v * dx - \
          mu * (H - rho_w / rho * zb)**q * u_mid**p * v * dx + \
          2. * B * H.dx(0) * u_mid.dx(0)**(1/n) * v * dx - \
          2. * B * H/n * u_mid.dx(0)**(1/n - 1) * u_mid.dx(0) * v.dx(0) * dx - \
          B * H / W * (((n+2) * u_mid)/(2*W))**(1/n) * v * dx

#          2. * B * H0 * gn * v * ds + \
fu    = - rho * g * H0 * zs.dx(0) * v * dx - \
          mu * (H0 - rho_w / rho * zb)**q * u_mid**p * v * dx + \
          2. * B * H0 * u_mid.dx(0)**(1/n) * v.dx(0) * dx - \
          B * H / W * (((n+2) * u_mid)/(2*W))**(1/n) * v * dx

# Momentum balance with regularization on viscosity
#s  = 1e-6
#fu = - rho * g * H0 * zs.dx(0) * v * dx - \
#       mu * (H0 - rho_w / rho * zb)**q * u_mid**p * v * dx + \
#       2. * B * H_mid * abs(u.dx(0) + s)**((1-n)/n)*u.dx(0) * v.dx(0) * dx - \
#       B * H0 / W * (((n+2) * u_mid)/(2*W))**(1/n) * v * dx

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
gry = '0.4'
red = '#5f4300'
pur = '#3d0057'
clr = red

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
ax1.set_ylim([-200,2000])

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
  zs = project(H, Q)

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
