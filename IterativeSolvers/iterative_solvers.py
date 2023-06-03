# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Nathan Schill
Section 2
Tues. Apr. 18, 2023
"""

import numpy as np
from numpy import linalg as la
from scipy import sparse
from matplotlib import pyplot as plt


# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.
    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.
        as_sparse: If True, an equivalent sparse CSR matrix is returned.
    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix."""
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = sparse.dok_matrix((n,n))
    rows = np.random.choice(n, size=num_entries)
    cols = np.random.choice(n, size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr()          # convert to row format for the next step
    for i in range(n):
        A[i,i] = abs(B[i]).sum() + 1
    return A.tocsr() if as_sparse else A.toarray()


# Problems 1 and 2
def jacobi(A, b, plot=False, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via the Jacobi Method.
    
    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """

    n = A.shape[0]
    D = np.diagonal(A)

    # Initial guess
    x0 = np.zeros(n)

    # Init absolute errors as negative one so not mistaken for real errors
    abs_errors = -1 * np.ones(maxiter)

    # Iterate according to equation (15.2)
    for i in range(maxiter):
        x1 = x0 + (b - A@x0)/D

        # Record absolute error
        abs_errors[i] = la.norm(A@x1 - b, ord=np.inf)

        # Check distance between iterations
        if la.norm(x1 - x0, ord=np.inf) < tol:
            break
        
        x0 = x1
    
    if plot:
        # Keep only real errors and plot
        abs_errors = abs_errors[:i+1]
        plt.semilogy(np.arange(i+1) + 1, abs_errors)

        # Plot properties
        plt.xlabel('Iteration')
        plt.ylabel('Absolute error of approximation')
        plt.title('Convergence of Jacobi Method')
        plt.show()

    return x0


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    
    n = A.shape[0]

    # Initial guess
    x0 = np.zeros(n)
    x1 = np.copy(x0)

    # Init absolute errors as negative one so not mistaken for real errors
    abs_errors = -1 * np.ones(maxiter)

    # Iterate according to equation (15.2)
    for i in range(maxiter):
        for j in range(n):
            x1[j] = x0[j] + (b[j] - A[j]@x0)/A[j,j]

        # Record absolute error
        abs_errors[i] = la.norm(A@x1 - b, ord=np.inf)

        # Check distance between iterations
        if la.norm(x1 - x0, ord=np.inf) < tol:
            break
        
        x0 = np.copy(x1)
    
    if plot:
        # Keep only real errors and plot
        abs_errors = abs_errors[:i+1]
        plt.semilogy(np.arange(i+1) + 1, abs_errors)

        # Plot properties
        plt.xlabel('Iteration')
        plt.ylabel('Absolute error of approximation')
        plt.title('Convergence of Gauss-Seidel Method')
        plt.show()

    return x0

# TODO?: Needs more iterations than 100 to converge (many more)
# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    
    n = A.shape[0]

    # Initial guess
    x0 = np.zeros(n)
    x1 = np.copy(x0)

    # Iterate according to equation (15.2)
    for _ in range(maxiter):
        for j in range(n):
            ### Code from lab PDF
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]

            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Ajx = A.data[rowstart:rowend] @ x0[A.indices[rowstart:rowend]]

            x1[j] = x0[j] + (b[j] - Ajx)/A.diagonal()[j]

        # Check distance between iterations
        if la.norm(x1 - x0, ord=np.inf) < tol:
            break
        
        x0 = np.copy(x1)

    return x0


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not method converged.
        (int): The number of iterations computed.
    """
    
    n = A.shape[0]
    converged = False

    # Initial guess
    x0 = np.zeros(n)
    x1 = np.copy(x0)

    # Iterate according to equation (15.2)
    diag = A.diagonal()
    for i in range(maxiter):
        for j in range(n):
            ### Code from lab PDF
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]

            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Ajx = A.data[rowstart:rowend] @ x0[A.indices[rowstart:rowend]]

            x1[j] = x0[j] + omega*(b[j] - Ajx)/diag[j]

        # Check distance between iterations
        if la.norm(x1 - x0, ord=np.inf) < tol:
            converged = True
            break
        
        x0 = np.copy(x1)

    return x0, converged, i


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False, A=None, b=None):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.
        A, b: matrix and vector to use instead of generating them in this function.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """

    if A is None and b is None:
        # Linear Systems lab problem 5 to create matrix A
        B = sparse.diags([1, -4, 1], (-1, 0, 1), shape=(n, n))
        I = np.identity(n)

        form = [[None for _ in range(i)] +
                [I if i >= 1 else None]+ [B] + [I if i <= n-2 else None] +
                [None for _ in range(n-i)]
                for i in range(n)]
        
        A = sparse.bmat(form).tocsr()

        # Construct vector b
        tile = np.array([-100] + [0] * (n-2) + [-100])
        b = np.tile(tile, n)
    
    # Solve
    u, converged, iterations = sor(A, b, omega, tol, maxiter)
    
    if plot:
        # Reshape and plot
        u_plot = np.reshape(u, (n,n))
        im = plt.pcolormesh(u_plot, cmap='coolwarm')
        ax = plt.gca()

        # Plot properties
        plt.title('Hot plate steady state')
        cbar = plt.colorbar(im)
        cbar.ax.set_title('Temp (deg C)')

        # Hide major ticks and major tick labels
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.show()

    return u, converged, iterations


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    
    # n and omegas
    n = 20
    O = np.arange(1, 2, 0.05)

    # Pre-generate A and b
    # Linear Systems lab problem 5 to create matrix A
    B = sparse.diags([1, -4, 1], (-1, 0, 1), shape=(n, n))
    I = np.identity(n)

    form = [[None for _ in range(i)] +
            [I if i >= 1 else None]+ [B] + [I if i <= n-2 else None] +
            [None for _ in range(n-i)]
            for i in range(n)]
    
    A = sparse.bmat(form).tocsr()

    # Construct vector b
    tile = np.array([-100] + [0] * (n-2) + [-100])
    b = np.tile(tile, n)
    
    # Get number of iterations for each omega
    num_iters = -1 * np.ones_like(O)
    for i, omega in enumerate(O):
        num_iters[i] = hot_plate(n, omega, tol=1e-2, maxiter=1000, A=A, b=b)[2]

    # Plot
    plt.plot(O, num_iters)
    plt.xlabel('Omega')
    plt.ylabel('Number of iterations')
    plt.title('Hot plate: omega vs number of iterations')
    plt.show()

    return O[np.argmin(num_iters)]