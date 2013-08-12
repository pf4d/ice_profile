from dolfin import *
from pylab  import plot, show, spy
from pylab  import sqrt as psqrt
import sys

bb         = bool(int(sys.argv[1]))
print sys.argv[0]

mesh       = IntervalMesh(10, 0, 1)  
Q          = FunctionSpace(mesh, "Lagrange",1)       # Order 1 function space
MQ         = Q * Q
           
dU         = TrialFunction(MQ)
dU1, dU1   = split(dU)
Phi        = TestFunction(MQ)
phi1, phi2 = split(Phi)
U          = Function(MQ)
u1, u2     = split(U)

# boundary conditions :
def left(x, on_boundary):
  return on_boundary and x[0] < 0 + 1e-6

def right(x, on_boundary):
  return on_boundary and x[0] > 1 - 1e-6

u1_bc_l = DirichletBC(MQ.sub(0), 1.0, left)
u1_bc_r = DirichletBC(MQ.sub(0), 0.0, right)
u2_bc_l = DirichletBC(MQ.sub(1), 0.0, left)

bcs     = [u1_bc_l, u1_bc_r, u2_bc_l] 

# SUPG method stabilization :
vnorm    = sqrt(dot(u2, u2) + 1e-10)
cellh    = CellSize(mesh)
phi2hat  = phi2 + cellh/(2*vnorm)*dot(u2, phi2.dx(0))

F1  = + u1.dx(0) * phi1 * dx \
      + u1.dx(0) * phi1.dx(0) * dx
F2  = + u1.dx(0) * u2.dx(0) * phi2 * dx \
      + u1 * phi2 * dx

F   = F1 + F2

Jac = derivative(F, U, dU)

u1_init = Expression("1-x[0]")
u2_init = Constant(1.0)

U_k = project(as_vector([u1_init, u2_init]), MQ) # project inital values
U.vector().set_local(U_k.vector().array())       # initalize u1, u2 in solution


if bb:
  problem = NonlinearVariationalProblem(F, U, bcs, Jac)
  solver  = NonlinearVariationalSolver(problem)
  solver.solve()
else:
  converged  = False
  atol, rtol = 1e-7, 1e-10           # abs/rel tolerances
  lmbda      = 1.0                   # relaxation parameter
  bcs_u      = homogenize(bcs)       # residual is zero on essential boundaries
  nIter      = 0                     # number of iterations
  residual   = 1                     # residual
  rel_res    = residual              # initial epsilon
  maxIter    = 25                    # max iterations

  u1_        = Function(Q)
  u2_        = Function(Q)
  
  while not converged and nIter < maxIter:
    nIter  += 1                                # increment interation
    Jac     = derivative(F, U, dU)
    A, b    = assemble_system(Jac, -F, bcs_u)  # assemble system
    solve(A, U_k.vector(), b)                  # determine step direction
    rel_res = U_k.vector().norm('l2')          # calculate norm

    # calculate residual :
    a = assemble(F)
    for bc in bcs_u:
      bc.apply(a)
    residual  = b.norm('l2')
   
    converged = residual < atol or rel_res < rtol
    
    U.vector()[:] += lmbda*U_k.vector()        # New u vector

    # ensure non-negativity :
    #u1_t = project(u1, Q).vector().array()
    #u2_t = project(u2, Q).vector().array()

    #u1_t[u1_t <= 0] = 1E-18
    #u2_t[u2_t <= 0] = 1E-18

    #u2_t = u2_t**(1/3.)

    #u1_.vector().set_local(u1_t)
    #u2_.vector().set_local(u2_t)
    #U_new = project(as_vector([u1_, u2_]), MQ)
    #U.vector().set_local(U_new.vector().array())

    string = "Newton iteration %d: r (abs) = %.3e (tol = %.3e) " \
             +"r (rel) = %.3e (tol = %.3e)"
    print string % (nIter, residual, atol, rel_res, rtol)

xcrd = mesh.coordinates()[:,0]
plot(xcrd, project(u1, Q).vector().array())
plot(xcrd, project(u2, Q).vector().array())

show()



