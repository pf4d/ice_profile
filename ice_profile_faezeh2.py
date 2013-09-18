#!/usr/bin/python
from numpy import *
import numpy as np
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pylab import mpl
from dolfin import *

f = open('data/faezeh_profile.dat')
header = f.readline()

xdat = []
Hdat = []
hdat = []
jdat = []
bdat = []
wdat = []

for l in f.readlines():
  lst = l.split('\t')
  xdat.append(float(lst[0]))
  Hdat.append(float(lst[1]))
  hdat.append(float(lst[2]))
  jdat.append(float(lst[3]))
  bdat.append(float(lst[4]))
  wdat.append(float(lst[5].rstrip('\n')))
  
xdat = array(xdat)
Hdat = array(Hdat) 
hdat = array(hdat) 
jdat = array(jdat) 
bdat = array(bdat) 
wdat = array(wdat) 
num  = len(xdat)

# For plotting
mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

### PHYSICAL CONSTANTS ###
spy   = 31556926.         # seconds per year ............... [s]
rho_i = 911.              # density of ice ................. [kg m^-3]
rho_w = 1000.             # density of water ............... [kg m^-3]
rho_p = 1028.             # density of sea water ........... [kg m^-3]
g     = 9.81              # gravitation acceleration ....... [m s^-2]
n     = 3.                # flow law exponent
m     = 3.                # bed friction exponent
A     = 5.6e-17 / spy     # temp-dependent ice-flow factor.. [Pa^-n s^-1]
Bs    = 100.              # sliding parameter .............. [Pa m^-2/3 s^1/3]
B     = A**(-1/n)         # ice hardeness .................. [Pa s^(1/n)]
beta  = 1e9               # basal resistance ............... [Pa s m^-1]
C     = 7.624e6           # basal resistance ............... [Pa m^-1/3 s^1/3]
amax  = .5 / spy          # max accumlation/ablation rate .. [m s^-1]
mu    = 1.0               # Basal traction constant
sb    = 0.0               # back stress

### SIMULATION PARAMETERS ### 
H_MIN = 0.0               # Minimal ice thickness .......... [m]
H_MAX = 600.              # Maximum plot height ............ [m]
D_MAX = 500.              # maximum depth of bed ........... [m]
u_MAX = 2000.             # maximum velocity to plot  ...... [m/s]

### DOMAIN DESCRIPTION ###
xl    = min(xdat)         # left edge (divide) ............. [m]
xr    = max(xdat)         # right edge (margin/terminus) ... [m]
Hd    = 100.0             # thickness at divide ............ [m]
c     = 1/3.              # percent of accumulation range .. [%]
L     = c * (xr - xl)     # length of domain ............... [m]
ela   = L / 1000

# unit interval mesh :
mesh  = IntervalMesh(num-1,xl,xr)
cellh = CellSize(mesh)
mesh.coordinates()[:,0] = xdat
xcrd  = mesh.coordinates()[:,0]/1000  # divide for units in km.

# create FunctionSpace :
Q     = FunctionSpace(mesh, "CG", 1)

# boundary conditions :
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1e-6

# Dirichlet conditions :
u_bc = DirichletBC(Q, 0.0, divide)    # velocity at divide

# Neumann conditions :
code = 'A * pow(rho_i*g/4 * (H - rho_p/rho_i*pow(D,2)/H - sb/(rho_i*g)), n)'
gn   = Expression(code, A=A, rho_i=rho_i, rho_p=rho_p, 
                  g=g, D=0, sb=sb,  H=H_MIN, n=n)

# INTIAL CONDITIONS:
# surface :
zs   = interpolate(Constant(0.0), Q)
zs.vector().set_local(bdat + Hdat)

# bed :
zb   = interpolate(Constant(0.0), Q)
zb.vector().set_local(bdat)

# thickness :
H    = interpolate(Constant(0.0), Q)
H.vector().set_local(Hdat)

# half width :
W    = interpolate(Constant(1000.), Q)
#W.vector().set_local(wdat)

# variational problem :
u    = Function(Q)                    # solution
phi  = TestFunction(Q)                # test function
du   = TrialFunction(Q)              # trial function

# SUPG method phihat :        
unorm   = sqrt(dot(u, u) + 1e-10)
phihat  = phi + cellh/(2*unorm)*dot(u, phi.dx(0))

# Continuity equation: weak form of eqn. 9.54 of vanderveen
fH      = + 1/W * (H*u*W).dx(0) * phihat * dx

# Momentum balance: weak form of equation 9.65 of vanderveen
vsc_reg        = 5e-7
basal_drag     = + mu * Bs * ((H - rho_p/rho_i * zb) * u)**(1/m) * phi * dx

# Momentum balance: Vieli and Payne, 2005
driving_stress = + rho_i * g * H * zs.dx(0) * phi * dx 
lat_drag       = + B * H / W * (((n+2) * u)/(2*W) + vsc_reg)**(1/n) * phi * dx 
basal_drag     = + beta * u * phi * dx
long_stress    = + 2. * B * H * (u.dx(0) + vsc_reg)**(1/n) * phi.dx(0) * dx 

fu    = basal_drag + long_stress + lat_drag + basal_drag
f     = fH + fu
df    = derivative(f, u, du)

# Create non-linear solver instance
problem = NonlinearVariationalProblem(f, u, u_bc, J=df)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 100
prm['newton_solver']['relaxation_parameter'] = 0.8

#===============================================================================
# Plot solution
gry = '0.4'
red = '#5f4300'
pur = '#3d0057'
clr = pur

plt.ion()

fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# plot the accumulation
Hplot  = project(H, Q).vector().array()
hplot  = project(H + zb, Q).vector().array()
uplot  = project(u, Q).vector().array() * spy
zbPlot = project(zb, Q).vector().array()
zbp,   = ax1.plot(xcrd, zbPlot, red, lw=2)
hp,    = ax1.plot(xcrd, hplot, 'k', lw=2)
ax1.plot(xcrd, [H_MAX] * len(xcrd), 'r+')
ax1.set_xlabel('$x$ [km]')
ax1.set_ylabel('$h$ [m]')
ax1.set_ylim([-D_MAX,H_MAX])
ax1.set_xlim([xl/1000, xr/1000])
ax1.grid()

up,    = ax2.plot(xcrd, uplot, clr, lw=2)
ax2.axvline(x=ela, lw=2, color = gry)
ax2.set_ylabel('$u$ [m/a]', color=clr)
ax2.set_xlim([xl/1000, xr/1000])
ax2.set_ylim([-100, u_MAX])
ax2.grid()
for tl in ax2.get_yticklabels():
  tl.set_color(clr)

plt.draw()

# Solve the nonlinear system 
solver.solve()

# Plot solution
Hplot = project(H, Q).vector().array()
uplot = project(u, Q).vector().array()
hplot = project(H + zb, Q).vector().array() 

hp.set_ydata(hplot)
up.set_ydata(uplot * spy)
plt.draw() 
plt.show() 
