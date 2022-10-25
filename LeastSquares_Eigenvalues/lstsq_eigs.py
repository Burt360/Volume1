# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Nathan Schill
Sec. 3
Tues. Nov. 1, 2022
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    
    # Compute the QR decomposition
    Q, R = la.qr(A, mode='economic')
    
    # Solve for the least squares solution using equation 4.1 in the lab PDF
    x_hat = la.solve_triangular(R, Q.T @ b)

    return x_hat
    

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    
    # Load data
    data = np.load('housing.npy')
    
    # The first column contains years and the second column contains ones,
    # to create a line of best fit b = slope * x + intercept
    A = np.column_stack((data[:,0], np.ones(len(data))))

    # Vector of price indices
    b = data[:,1]

    # Find the least squares solution
    slope_hat, intercept_hat = least_squares(A, b)

    # Plot the data points
    plt.scatter(data[:,0], data[:,1])

    # Plot the line of best fit
    x = np.linspace(0, 16, 50)
    y = np.polyval((slope_hat, intercept_hat), x)
    plt.plot(x, y)

    # Set the y-axis min as 0, and show the plot
    plt.ylim(ymin=0)
    plt.xlabel('Year (2000s)')
    plt.ylabel('Price index')
    plt.title('Price index trends in early 2000s')
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    
    def compare_coeffs(years, b, coeffs, deg):
        print(coeffs_hat)
        print(np.polyfit(years, b, deg))
        print()

    # Load data
    data = np.load('housing.npy')
    years = data[:,0]

    fig, axes = plt.subplots(1, 4)
    for ax, deg in zip(axes, (3, 6, 9, 12)):
        
        # Vandermonde matrix of degree deg using the years values
        A = np.vander(years, deg + 1)

        # Vector of price indices
        b = data[:,1]

        # Find the least squares solution
        coeffs_hat = la.lstsq(A, b)[0]

        # Plot the data points
        ax.scatter(data[:,0], data[:,1])

        # Plot the polynomial of degree deg of best fit
        x = np.linspace(0, 16, 50)
        y = np.polyval((coeffs_hat), x)
        ax.plot(x, y)
        
        # Set the y-axis min as 0
        ax.set_ylim(ymin=0)
        ax.set_title(f'Degree {deg}')
        ax.set_xlabel('Year (2000s)')
        ax.set_ylabel('Price index')

    # Set title and show the plot
    plt.suptitle('Price index trends in early 2000s')
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    
    # Load the ellipse data
    data = np.load('ellipse.npy')
    x = data[:,0]
    y = data[:,1]
    
    # Create a matrix equation for ax**2 + bx + cxy + dy + ey**2 = 1
    A = np.column_stack((x**2, x, x*y, y, y**2))
    b = np.ones(len(data))

    # Get the least squares solution
    coeffs_hat = la.lstsq(A, b)[0]

    # Plot the data points and the ellipse
    plt.scatter(x, y)
    plot_ellipse(*coeffs_hat)

    # Label and title the plot, then show
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fit ellipse to points')
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """

    ### Find dominant eigenvalue using Algorithm 4.1 given in the lab PDF

    m, n = A.shape

    x_prev = np.random.rand(n)
    x_prev /= la.norm(x_prev)
    
    for i in range(N):
        x_curr = A@x_prev
        x_curr /= la.norm(x_curr)
        
        if la.norm(x_curr - x_prev) < tol:
            break
        else:
            x_prev = x_curr

    return np.dot(x_curr, A) @ x_curr, x_curr


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    
    ### Find eigenvalues of A using Algorithm 4.2 given in the lab PDF

    m, n = A.shape
    S = la.hessenberg(A)

    for i in range(N):
        Q, R = la.qr(S)
        S = R@Q
    
    eigs = list()

    i = 0
    while i < n:
        if i == n-1 or S[i+1, i] < tol:
            # S_i is 1x1
            eigs.append(S[i,i])
        else:
            # S_i is 2x2
            a, b, c, d = S[i,i], S[i,i+1], S[i+1,i], S[i+1,i+1]

            eigs.append((a+d + cmath.sqrt((a+d)**2 - 4*(a*d-b*c)))/2)
            eigs.append((a+d - cmath.sqrt((a+d)**2 - 4*(a*d-b*c)))/2)

            i += 1
        i += 1
    
    return eigs
