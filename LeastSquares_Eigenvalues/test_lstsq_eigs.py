import pytest
from lstsq_eigs import *

import numpy as np

def test_least_squares():
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    b = np.array([0,1,0])

    test_x_hat = least_squares(A, b)
    x_hat = np.linalg.lstsq(A, b, rcond=None)[0]

    assert np.allclose(test_x_hat, x_hat)

def test_line_fit():
    pass#line_fit()

def test_polynomial_fit():
    pass#polynomial_fit()

def test_ellipse_fit():
    pass#ellipse_fit()

def test_power_method():
    # Construct a random matrix with positive entries.
    A = np.random.random((10,10))

    # Compute the eigenvalues and eigenvectors of A via SciPy.
    eigs, vecs = la.eig(A)
    # Get the dominant eigenvalue and eigenvector of A.
    # The eigenvector of the kth eigenvalue is the kth column of 'vecs'.
    loc = np.argmax(eigs)
    lamb, x = eigs[loc], vecs[:,loc]
    # Verify that Ax = lambda x.
    np.allclose(A @ x, lamb * x)

    lamb_test, x_test = power_method(A)
    assert np.allclose(A @ x_test, lamb_test * x_test)

def test_qr_algorithm():
    # Construct a symmetric random matrix with positive entries.
    A = np.random.random((5, 5))
    A += A.T

    eigs_test = np.sort_complex(qr_algorithm(A))
    eigs = np.sort_complex(la.eig(A)[0])
    print(eigs_test)
    print(eigs)

    assert np.allclose(eigs_test, eigs)
