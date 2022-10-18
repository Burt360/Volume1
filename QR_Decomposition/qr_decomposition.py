# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Nathan Schill
Section 3
Tues. Oct. 25, 2022
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    
    ### Algorithm as given in Algorithm 3.1 in QR Decomposition PDF.

    m, n = A.shape
    Q = A.copy()
    R = np.zeros((n, n))

    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]

        for j in range(i+1, n):
            R[i,j] = np.dot(Q[:,j].T, Q[:,i])
            Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]

    return Q, R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    
    ### det(A) as given in equation 3.1 in QR Decomposition PDF.
    return np.abs(np.prod(np.diag(la.qr(A, mode='economic')[1])))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    
    m, n = A.shape
    Q, R = la.qr(A, mode='economic')

    # Ax = QRx = b -> Rx = Q^T b = y since Q is orthonormal
    y = Q.T @ b

    # Init x as row  for easier coding
    x = [0] * n

    ### Solve Rx = y for x
    # Iterate up the entries of x
    for i in range(n-1, 0-1, -1):
        x[i] = 1/R[i][i] * (y[i] - sum([R[i][j] * x[j] for j in range(i+1, n)]))

    return np.array(x)


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    
    ### Algorithm as given in Algorithm 3.2 in QR Decomposition PDF.
    m, n = A.shape
    Q = np.identity(m)
    R = A.copy()

    sign = lambda x: 1 if x >= 0 else -1

    for k in range(n):
        u = R[k:,k].copy()
        u[0] = u[0] + sign(u[0]) * la.norm(u)
        u = u/la.norm(u)

        R[k:,k:] = R[k:,k:] - 2*np.outer(u, np.dot(u, R[k:,k:]))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u, np.dot(u, Q[k:,:]))

    return Q.T, R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """

    ### Algorithm as given in Algorithm 3.3 in QR Decomposition PDF.
    
    m, n = A.shape
    H = A.copy()
    Q = np.identity(m)

    sign = lambda x: 1 if x >= 0 else -1

    for k in range(n-2):
        u = H[k+1:,k].copy()
        u[0] = u[0] + sign(u[0]) * la.norm(u)

        u = u/la.norm(u)

        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u, np.dot(u.T, H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2*np.outer(np.dot(H[:,k+1:], u), u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u, np.dot(u.T, Q[k+1:,:]))

    return H, Q.T
