from dolfin import *
from pylab  import plot, show, spy


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

bcs = [DirichletBC(MQ.sub(0), 1,   left), 
       DirichletBC(MQ.sub(0), 0,   right),
       DirichletBC(MQ.sub(1), 0,   left),
       DirichletBC(MQ.sub(1), 1,   right)]

k   = 1.0*u1 + 1.0

F1  = k * sqrt(u1.dx(0)) * t1.dx(0) * dx
F2  = u2 * t2 * dx
F   = F1 + F2

Jac = derivative(F, U, dU)

u1_init = Expression("1-x[0]")
u2_init = Constant(1E-18)

U_k = project(as_vector([u1_init, u2_init]), MQ) # project inital values
U.vector().set_local(U_k.vector().array())       # initalize u1, u2 in solution


if False:
  problem = NonlinearVariationalProblem(F, U, bcs, Jac)
  solver = NonlinearVariationalSolver(problem)
  solver.solve()
else:
  a_tol, r_tol = 1e-7, 1e-10                     # abs/rel tolerances
  U_inc   = Function(MQ)                         # residual
  bcs     = homogenize(bcs)                      # residual is zero on boundary
  nIter   = 0                                    # number of iterations
  eps     = 1                                    # initial epsilon
  maxIter = 200                                  # max iterations

  while eps > r_tol and nIter < maxIter:         # Newton iterations
    nIter += 1                                   # increment interation
    A, b   = assemble_system(Jac, -F, bcs)       # assemble system
    solve(A, U_inc.vector(), b)                  # determine step direction
    eps    = U_inc.vector().norm('l2')           # calculate norm

    a     = assemble(F)
    for bc in bcs:
      bc.apply(a)
    fnorm = b.norm('l2')
    lmbda = 1.0                                  # step size, initially 1
    
    U.vector()[:] += lmbda*U_inc.vector()        # New u vector

    # ensure non-negativity :
    u1_t = project(u1, Q).vector().array()
    u2_t = project(u2, Q).vector().array()

    u1_t[u1_t <= 0] = 1E-18
    u2_t[u2_t <= 0] = 1E-18

    u1_ = Function(Q)
    u1_.vector().set_local(u1_t)
    u2_ = Function(Q)
    u2_.vector().set_local(u2_t)
    U_new = project(as_vector([u1_, u2_]), MQ)
    U.vector().set_local(U_new.vector().array())

    print '      {0:2d}  {1:3.2E}  {2:5e}'.format(nIter, eps, fnorm)

plot(project(u1, Q).vector().array())
plot(project(u2, Q).vector().array())
show()
