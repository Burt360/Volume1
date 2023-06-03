from iterative_solvers import *

def run(fn, n, plot=False, sparse=False, omega=None):
    A = diag_dom(n, as_sparse=sparse)
    b = np.random.rand(n)

    if omega is None:
        x = fn(A, b, plot=plot) if not sparse else fn(A, b, maxiter=1000)
    else:
        # SOR
        soln = fn(A, b, omega, maxiter=1000)
        x = soln[0]
        print('converged:', soln[1], '| iters:', soln[2])
    prod = A@x
    
    # print(x, prod, b)
    print(f'{n:<2}', np.allclose(prod, b))

    if sparse:
        # return
        print(prod[:10], b[:10])


def test_jacobi():
    return
    m, M = 1, 10

    print('Jacobi')
    for n in range(m, M+1):
        run(jacobi, n)

    # run(jacobi, 8, True)

def test_gs():
    return
    m, M = 1, 10

    print('Gauss-Seidel')
    for n in range(m, M+1):
        run(gauss_seidel, n)

    # run(gauss_seidel, 8, True)

def test_gs_sparse():
    return
    print('Gauss-Seidel sparsre')
    run(gauss_seidel_sparse, 1000, sparse=True)

def test_sor():
    return
    print('SOR')
    run(sor, 1000, sparse=True, omega=1)

def test_hot_plate():
    # return
    n = 20
    print('Hot plate')
    hot_plate(n, 1, plot=True, maxiter=100)

def test_prob7():
    return
    print(prob7())