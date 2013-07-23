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

F1  = sqrt(k) * u1.dx(0) * t1.dx(0) * dx
F2  = u2 * t2 * dx
F   = F1 + F2

Jac = derivative(F, U, dU)

u1_init = Expression("1-x[0]")
u2_init = Constant(0.0)

U_k = project(as_vector([u1_init, u2_init]), MQ) # project inital values
U.vector().set_local(U_k.vector().array())       # initalize u1, u2 in solution


if False:
  problem = NonlinearVariationalProblem(F, U, bcs, Jac)
  solver = NonlinearVariationalSolver(problem)
  solver.solve()
else:
  a_tol, r_tol = 1e-7, 1e-10
  U_inc   = Function(MQ)
  bcs     = homogenize(bcs)
  nIter   = 0
  eps     = 1
  maxIter = 200

  while eps > r_tol and nIter < maxIter:         # Newton iterations
    nIter += 1
    A, b   = assemble_system(Jac, -F, bcs)
    solve(A, U_inc.vector(), b)                  # Determine step direction
    eps    = U_inc.vector().norm('l2')

    a     = assemble(F)
    for bc in bcs:
      bc.apply(a)
    fnorm = b.norm('l2')
    lmbda = 1.0                                  # step size, initially 1

    U.vector()[:] += lmbda*U_inc.vector()        # New u vector

    print '      {0:2d}  {1:3.2E}  {2:5e}'.format(nIter, eps, fnorm)

plot(project(u1, Q).vector().array())
plot(project(u2, Q).vector().array())
show()
