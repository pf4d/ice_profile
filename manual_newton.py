from dolfin import *
from pylab  import plot, show, spy
from pylab  import sqrt as psqrt
import sys

bb       = bool(int(sys.argv[1]))
print sys.argv[0]

mesh     = IntervalMesh(4, 0,1)  
Q        = FunctionSpace(mesh, "Lagrange",1)       # Order 1 function space
MQ       = Q * Q

dU       = TrialFunction(MQ)
dU1, dU1 = split(dU)
test_U   = TestFunction(MQ)
t1, t2   = split(test_U)
U        = Function(MQ)
u1, u2   = split(U)

left, right = compile_subdomains([
  "(std::abs( x[0] )     < DOLFIN_EPS) && on_boundary",
  "(std::abs( x[0]-1.0 ) < DOLFIN_EPS) && on_boundary"])

bcs = [DirichletBC(MQ.sub(0), 1, left), 
       DirichletBC(MQ.sub(0), 0, right),
       DirichletBC(MQ.sub(1), 0, left),
       DirichletBC(MQ.sub(1), 1, right)]

k   = 1.0*u1 + 1.0
c   = 1e-9

# SUPG method stabilization :
vnorm    = sqrt(dot(u2, u2) + 1e-10)
cellh    = CellSize(mesh)
t2hat    = t2 + cellh/(2*vnorm)*dot(u2, t2.dx(0))

F1  = + u1 * u1.dx(0) * t1.dx(0) * dx \
      + u1.dx(0) * t1.dx(0) * dx
F2  = + c * u2 * t2hat * dx

F   = F1 + F2

Jac = derivative(F, U, dU)

u1_init = Expression("1-x[0]")
u2_init = Constant(1)

U_k = project(as_vector([u1_init, u2_init]), MQ) # project inital values
U.vector().set_local(U_k.vector().array())       # initalize u1, u2 in solution


if bb:
  problem = NonlinearVariationalProblem(F, U, bcs, Jac)
  solver  = NonlinearVariationalSolver(problem)
  solver.solve()
else:
  atol, rtol = 1e-7, 1e-10                       # abs/rel tolerances
  lmbda      = 1.0                               # relaxation parameter
  U_inc      = Function(MQ)                      # residual
  bcs_u      = homogenize(bcs)                   # residual is zero on boundary
  nIter      = 0                                 # number of iterations
  residual   = 1                                 # residual
  rel_res    = 1                                 # initial epsilon
  maxIter    = 200                               # max iterations

  while residual > atol and rel_res > rtol and nIter < maxIter:
    nIter  += 1                                  # increment interation
    A, b    = assemble_system(Jac, -F, bcs)      # assemble system
    solve(A, U_inc.vector(), b)                  # determine step direction
    rel_res = U_inc.vector().norm('l2')          # calculate norm

    # calculate residual :
    a = assemble(F)
    for bc in bcs_u:
      bc.apply(a)
    residual = b.norm('l2')
    
    U.vector()[:] += lmbda*U_inc.vector()        # New u vector

    # ensure non-negativity :
    u1_t = project(u1, Q).vector().array()
    u2_t = project(u2, Q).vector().array()

    u1_t[u1_t <= 0] = 1E-18
    u2_t[u2_t <= 0] = 1E-18

    u2_t = u2_t**(1/3.)

    u1_ = Function(Q)
    u1_.vector().set_local(u1_t)
    u2_ = Function(Q)
    u2_.vector().set_local(u2_t)
    U_new = project(as_vector([u1_, u2_]), MQ)
    U.vector().set_local(U_new.vector().array())

    string = "Newton iteration %d: r (abs) = %.3e (tol = %.3e) " \
             +"r (rel) = %.3e (tol = %.3e)"
    print string % (nIter, residual, atol, rel_res, rtol)

xcrd = mesh.coordinates()[:,0]
plot(xcrd, project(u1, Q).vector().array())
plot(xcrd, project(u2, Q).vector().array())

show()
