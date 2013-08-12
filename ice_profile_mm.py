#!/usr/bin/python
from numpy import *
import numpy as np
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pylab import mpl
from dolfin import *


def update_length(mesh, u, dt):
  """
  evolve the ice length.
  """
  x0    = mesh.coordinates()[:,0]
  x     = x0 + u*dt
  mesh.coordinates()[:,0] = x
  return x / 1000

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

### PHYSICAL CONSTANTS ###
a   = 31556926.           # seconds per year ............... [s]
rho_i = 911.              # density of ice ................. [kg m^-3]
rho_w = 1000.             # density of water ............... [kg m^-3]
rho_p = 1028.             # density of sea water ........... [kg m^-3]
g     = 9.81              # gravitation acceleration ....... [m s^-2]
n     = 3.                # flow law exponent
m     = 3.                # bed friction exponent
A     = 5.6e-17 / a       # temp-dependent ice-flow factor.. [Pa^-n s^-1]
A     = 1e-18 / a
Bs    = 100.              # sliding parameter .............. [Pa m^-2/3 s^1/3]
B     = A**(-1/n)         # ice hardeness .................. [Pa s^(1/n)]
beta  = 1e9               # basal resistance ............... [Pa s m^-1]
C     = 7.624e6           # basal resistance ............... [Pa m^-1/3 s^1/3]
amax  = .5 / a            # max accumlation/ablation rate .. [m s^-1]
mu    = 1.0               # Basal traction constant
sb    = 0.0               # back stress

### SIMULATION PARAMETERS ### 
dt    = 50.000 * a        # time step ...................... [s]
t     = 0.                # begining time .................. [s]
tf    = 5000. * a         # end time ....................... [s]
H_MIN = 25.0              # Minimal ice thickness .......... [m]
H_MAX = 1000.             # Maximum plot height ............ [m]
D_MAX = -100.             # maximum depth of bed ........... [m]
u_MAX = 1000.             # maximum velocity to plot  ...... [m/s]
u_MIN = -100.             # minimum velocity to plot  ...... [m/s]
x_MAX = 500e3             # maximum x-distance to plot ..... [m]
x_MIN = 0.                # minimum x-distance to plot ..... [m]

### DOMAIN DESCRIPTION ###
xl    = 0.                # left edge (divide) ............. [m]
xr    = 500e3             # right edge (margin/terminus) ... [m]
Hd    = 100.0             # thickness at divide ............ [m]
c     = 1                 # percent of accumulation range .. [%]
L     = c*(xr - xl)       # length of domain ............... [m]
ela   = L / 1000

# unit interval mesh :
mesh  = IntervalMesh(100,xl,xr)
cellh = CellSize(mesh)
xcrd  = mesh.coordinates()[:,0] / 1000      # divide for units in km.

# create FunctionSpace :
Q     = FunctionSpace(mesh, "CG", 1)
MQ    = MixedFunctionSpace([Q, Q])

# boundary conditions :
def divide(x,on_boundary):
  return on_boundary and x[0] < xl + 1e-6

def terminus(x,on_boundary):
  return on_boundary and x[0] > xr - 1e-6

# Dirichlet conditions :
H_bc = DirichletBC(MQ.sub(0), H_MIN, terminus)  # thickness at terminus
u_bc = DirichletBC(MQ.sub(1), 0.0,   divide)    # velocity at divide
bcs  = [H_bc, u_bc]

# Neumann conditions :
code = 'A * pow(rho_i*g/4 * (H - rho_p/rho_i*pow(D,2)/H - sb/(rho_i*g)), n)'
q    = Expression(code, A=A, rho_i=rho_i, rho_p=rho_p, 
                  g=g, D=0, sb=sb,  H=H_MIN, n=n)

# Neumann conditions :
code = 'A * pow(rho_i*g*H/4 * (1 - rho_i/rho_w), n)'
q    = Expression(code, A=A, rho_i=rho_i, rho_w=rho_w, g=g, H=H_MIN, n=n)
#q    = Expression(code, A=A, rho_i=rho_i, rho_w=rho_w, g=g, H=1e6, n=n)

# INTIAL CONDITIONS:
# surface :
code = '729 - 2184.8  * pow(x[0] / 750000, 2) ' \
       '    + 1031.72 * pow(x[0] / 750000, 4) ' \
       '    - 151.72  * pow(x[0] / 750000, 6) '
zs   = interpolate(Constant(H_MIN),Q)
#zs   = interpolate(Expression("H0 - m * x[0]",m=1e-4, H0=H_MIN),Q)
#zs   = interpolate(Expression("H0 + " + code, H0=H_MIN),Q)

# bed :
zb   = interpolate(Constant(0.0),Q)
#zb   = interpolate(Expression("- m * x[0]",m=1e-4),Q)
#zb   = interpolate(Expression(code), Q)


# thickness :
H_i  = project(zs-zb,Q)
#H_i.vector().set_local(genfromtxt("data/H.txt"))

# half width :
W    = interpolate(Constant(1000.),Q)

# initial velocity :
u_i  = interpolate(Constant(0.0),Q) 
#u_i.vector().set_local(genfromtxt("data/u.txt"))

# accumulation :
adot = Constant(0.3/a)
adot = Expression('amax * (1 - x[0] / L)',L=L,amax=amax)

# variational problem :
U         = Function(MQ)                    # solution
H,u       = split(U)                        # solutions for H, u
U0        = Function(MQ)                    # previous solution
H0,u0     = split(U0)                       # previous solutions for H, u

dU        = TrialFunction(MQ)               # trial function for solution
dH,du     = split(dU)                       # trial functions for H, u
j         = TestFunction(MQ)                # test function in mixed space
phi,psi   = split(j)                        # test functions for H, u

U_k = project(as_vector([H_i,u_i]), MQ)     # project inital values on space
U.vector().set_local(U_k.vector().array())  # initalize H, u in solution
U0.vector().set_local(U_k.vector().array()) # initalize H, u in prev. sol

# SUPG method phihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
phihat = phi + cellh/(2*unorm)*dot(u, phi.dx(0))

# Continuity equation: weak form of eqn. 9.54 of vanderveen
theta = 0.5
H_mid = theta*H + (1 - theta)*H0
fH    = + (H-H0)/dt * phi * dx \
        + (H_mid*u).dx(0) * phihat * dx \
        - adot * phihat * dx

#Ldot  = ((H_mid*u).dx(0) - L*adot) / H_mid.dx(0)
#fH    = + (H-H0)/dt * phi * dx \
#        + 1/L * (H_mid*u).dx(0) * phihat * dx \
#        - Ldot/L * H_mid.dx(0) * phihat * dx \
#        - adot * phihat * dx

# SUPG method psihat :        
unorm  = sqrt(dot(u, u) + 1e-10)
psihat = psi + cellh/(2*unorm)*dot(u, psi.dx(0))

# Momentum balance: weak form of equation 9.65 of vanderveen
theta = 0.5 
u_mid = theta*u + (1 - theta)*u0
h     = H + zb

num1  = u_mid.dx(0)
num2  = u_mid
num3  = (H - rho_p/rho_i * zb) * u_mid

num1  = interpolate(Constant(0.0), Q)
num2  = interpolate(Constant(0.0), Q)
num3  = interpolate(Constant(0.0), Q)

fu    = + rho_i * g * H * h.dx(0) * psi * dx \
        + 2. * B * H * num1 * psi.dx(0) * dx \
        + B * H / W * (n+2)/(2*W) * num2 * psi * dx \
        + beta * u_mid * psi * dx

#fu    = + rho_i * g * H * h.dx(0) * psi * dx \
#        + mu * Bs * num3 * psi * dx \
#        + zero * q * psi * ds \
#        + 2. * B * H * num1 * psi.dx(0) * dx \
#        + B * H / W * num2 * psi * dx

#fu    = + rho_i * g * H * h.dx(0) * psi * dx \
#        + 2. * B * H / L * num1 * psi.dx(0) * dx \
#        + L * B * H / W * num2 * psi * dx \
#        + L * beta * sqrt(one) * u_mid * psi * dx

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

#solver.solve()

# Plot solution
gry = '0.4'
red = '#5f4300'
pur = '#3d0057'
clr = pur

plt.ion()

fig = plt.figure(figsize=(10,7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1])
ax2 = ax1.twinx()

# plot the accumulation
adotPlot = project(adot, Q).vector().array() * a
ax3.axhline(lw=2, color = gry)
ax3.axvline(x=ela, lw=2, color = gry)
ax3.plot(xcrd, adotPlot, 'r', lw=2)
ax3.grid()
ax3.set_xlabel('$x$ [km]')
ax3.set_ylabel('$\dot{a}$ [m/a]')
ax3.set_xlim([xl/1000, xr/1000])

Hplot  = project(H, Q).vector().array()
hplot  = project(H + zb, Q).vector().array()
uplot  = project(u, Q).vector().array() * a
zbPlot = project(zb, Q).vector().array()
zbp,   = ax1.plot(xcrd, zbPlot, red, lw=2)
hp,    = ax1.plot(xcrd, hplot, 'k', lw=2)
elem,  = ax1.plot(xcrd, [H_MAX] * len(xcrd), 'r+')
ax1.set_xlabel('$x$ [km]')
ax1.set_ylabel('$h$ [m]')
ax1.set_xlim([x_MIN/100, x_MAX/1000])
ax1.set_ylim([D_MAX,     H_MAX])
ax1.grid()

up,    = ax2.plot(xcrd, uplot, clr, lw=2)
ax2.axvline(x=ela, lw=2, color = gry)
ax2.set_ylabel('$u$ [m/a]', color=clr)
ax2.set_xlim([x_MIN/1000, x_MAX/1000])
ax2.set_ylim([u_MIN,      u_MAX])
ax2.grid()
for tl in ax2.get_yticklabels():
  tl.set_color(clr)

fig_time = plt.figtext(.80, .95, 'Time = 0.0 yr')
fig_flux = plt.figtext(.10, .95, 'Flux = 0.0 m/a')

plt.draw()
bcs = homogenize(bcs)

# Time-stepping
while t < tf:
  converged  = False
  atol, rtol = 1e-8, 1e-7              # abs/rel tolerances
  omega      = 0.8                     # relaxation parameter
  bcs_u      = homogenize(bcs)         # we know the essential BCs, i.e., 
                                       #  the residual is zero
  nIter      = 0                       # number of iterations
  residual   = 1                       # residual
  rel_res    = residual                # initial epsilon
  maxIter    = 100                     # max iterations
  
  num1_n     = Function(Q)
  num2_n     = Function(Q)
  num3_n     = Function(Q)
  while not converged and nIter < maxIter:
    nIter  += 1
    A, b    = assemble_system(df, -f, bcs_u)
    solve(A, U_k.vector(), b)
    rel_res = U_k.vector().norm('l2')
    
    c = assemble(f)
    for bc in bcs_u:
      bc.apply(c)
    residual = b.norm('l2')
   
    converged = residual < atol or rel_res < rtol

    U.vector()[:] += omega*U_k.vector()    # New u vector

    Hplot = project(H, Q).vector().array()
    uplot = project(u, Q).vector().array()
    
    Hplot[Hplot < H_MIN] = H_MIN
    uplot[uplot < 0.0]   = 0.0

    num1_t = project(u_mid.dx(0), Q).vector().array()
    num2_t = project(u_mid,       Q).vector().array()

    num1_t = num1_t**(1/1)
    num2_t = num2_t**(1/1)

    num1_t[num1_t < 0] = 1E-18
    num2_t[num2_t < 0] = 1E-18
   
    num1.vector().set_local(num1_t)
    num2.vector().set_local(num2_t)
    
    # update the dolfin vectors :
    H_i.vector().set_local(Hplot)
    u_i.vector().set_local(uplot)
    U_new = project(as_vector([H_i, u_i]), MQ)
    U.vector().set_local(U_new.vector().array())
    
    string = "Newton iteration %d: r (abs) = %.3e (tol = %.3e) " \
             +"r (rel) = %.3e (tol = %.3e)"
    print string % (nIter, residual, atol, rel_res, rtol)

  
  ## Solve the nonlinear system 
  #solver.solve()

  ## Plot solution

  hplot = project(H + zb, Q).vector().array() 
  uplot = project(u, Q).vector().array()

  #xcrd = update_length(mesh, uplot, dt)

  hp.set_ydata(hplot)
  hp.set_xdata(xcrd)
  up.set_ydata(uplot * a)
  up.set_xdata(xcrd)
  elem.set_xdata(xcrd)
  fig_time.set_text('Time = %.0f yr'  % (t/a))
  flux = project(q,Q).vector().array()[0] * a
  fig_flux.set_text('Flux = %.0f m/a' % flux)
  plt.draw() 

  # Copy solution from previous interval
  U0.assign(U)
  
  # Move to next interval and adjust boundary condition
  t += dt
