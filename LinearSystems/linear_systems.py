# linear_systems.py
"""Volume 1: Linear Systems.
Nathan Schill
Sec. 3
Tues. Oct. 18, 2022
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla

from time import perf_counter as pc
from matplotlib import pyplot as plt

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """

    n = np.shape(A)[0]
    B = A.copy()

    # Iterate across columns
    for j in range(n):
        # Row reduce each row below the jth row and starting with the jth column
        for i in range(j+1, n):
            B[i,j:] -= (B[i, j] / B[j, j]) * B[j, j:]
    
    return B


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """

    n = np.shape(A)[0]
    u = A.copy()
    l = np.identity(n)

    # Iterate across columns
    for j in range(n-1):
        # Row reduce each row below the jth row and starting with the jth column
        for i in range(j+1, n):
            l[i,j] = u[i, j] / u[j, j]
            u[i,j:] -= l[i,j] * u[j, j:]
    
    return l, u


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    n = np.shape(A)[0]

    L, U = lu(A)

    # Init y, x as row vectors for easier coding
    y = [0] * n
    x = [0] * n

    ### Solve Ly = b for y
    # Iterate down the entries of y
    for i in range(n):
        y[i] = b[i] - sum([L[i,j] * y[j] for j in range(i)])

    ### Solve Ux = y for x
    # Iterate up the entries of x
    for i in range(n-1, 0-1, -1):
        x[i] = 1/U[i][i] * (y[i] - sum([U[i][j] * x[j] for j in range(i+1, n)]))

    return np.array(x)


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    
    MAX_N = 12

    # Values of n to time
    N = [2**i for i in range(MAX_N)]

    # Dictionary of times
    test_times = {
        'inv' : list(),
        'solve' : list(),
        'lu_factor-lu_solve' : list(),
        'lu_solve' : list()
    }

    # For each size n, store the time for each type of solve
    for n in N:
        A = np.random.random((n,n))
        b = np.random.random(n)

        start = None
        end = None

        ### 1: la.inv
        start = pc()
        la.inv(A) @ b
        end = pc()
        test_times['inv'].append(end - start)

        ### 2: la.solve
        start = pc()
        la.solve(A, b)
        end = pc()
        test_times['solve'].append(end - start)

        ### 3: lu.factor and lu_solve
        start = pc()
        L, P = la.lu_factor(A)
        la.lu_solve((L, P), b)
        end = pc()
        test_times['lu_factor-lu_solve'].append(end - start)

        ### 4: lu_solve
        L, P = la.lu_factor(A)

        start = pc()
        la.lu_solve((L, P), b)
        end = pc()
        test_times['lu_solve'].append(end - start)

    # Plot the times for each solve method
    for test, times in test_times.items():
        plt.loglog(N, times, label=test)
    
    # Set x-axis scale as log_2
    plt.xscale('log', base=2)

    # Label axes and title, show legend, and show plot
    plt.xlabel('n')
    plt.ylabel('time (seconds)')
    plt.title('Time to solve Ax=b')
    plt.legend()
    plt.show()


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """

    B = sparse.diags([1, -4, 1], (-1, 0, 1), shape=(n, n))
    I = np.identity(n)

    form = [[None for _ in range(i)] +
            [I if i >= 1 else None]+ [B] + [I if i <= n-2 else None] +
            [None for _ in range(n-i)]
            for i in range(n)]
    
    A = sparse.bmat(form)

    def test():
        plt.spy(A, markersize=1)
        plt.show()
    
    test()

    return A


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    
    MAX_N = 7

    # Values of n to time
    N = [2**i for i in range(1, MAX_N)]

    # Dictionary of times
    test_times = {
        'sparse_solve' : list(),
        'reg_solve' : list()
    }

    # For each size n, store the time for each type of solve
    for n in N:
        A = prob5(n)
        b = np.random.random(n**2)

        start = None
        end = None

        ### 1: sparse_solve
        B = A.tocsr()

        start = pc()
        spla.spsolve(B, b)
        end = pc()
        test_times['sparse_solve'].append(end - start)

        ### 2: la_solve
        C = A.toarray()

        start = pc()
        la.solve(C, b)
        end = pc()
        test_times['reg_solve'].append(end - start)

    # Plot the times for each solve method
    for test, times in test_times.items():
        plt.loglog([n for n in N], times, label=test)
    
    # Set x-axis scale as log_2
    plt.xscale('log', base=2)

    # Label axes and title, show legend, and show plot
    plt.xlabel('n')
    plt.ylabel('time (seconds)')
    plt.title('Time to solve Ax=b where A is (n^2 x n^2)')
    plt.legend()
    plt.show()

