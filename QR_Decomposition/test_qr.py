import numpy as np
from scipy import linalg as la

import qr_decomposition as qr

def test_prob1():
    m, n = 6, 4
    A = np.random.random((m, n))

    Q, R = qr.qr_gram_schmidt(A)

    assert A.shape == Q.shape
    assert R.shape == (n, n)

    assert np.allclose(np.triu(R), R)
    assert np.allclose(Q.T @ Q, np.identity(4))
    assert np.allclose(Q @ R, A)

def test_prob2():
    m, n = 4, 4
    A = np.random.random((m, n))

    assert np.isclose(qr.abs_det(A), np.abs(la.det(A)))

def test_prob3():
    m, n = 4, 4
    A = np.random.random((m, n))
    b = np.random.random(m)

    x = qr.solve(A, b)
    x_goal = la.solve(A, b)

    assert np.allclose(x, x_goal)

def test_prob4():
    m, n = 5, 3
    A = np.random.random((m, n))
    
    Q, R = qr.qr_householder(A)

    assert Q.shape == (m, m)
    assert R.shape == (m, n)
    
    np.allclose(Q @ R, A)

def test_prob5():
    A = np.random.random((8,8))
     
    H, Q = qr.hessenberg(A)
     
    assert np.allclose(np.triu(H, -1), H)
    assert np.allclose(Q @ H @ Q.T, A)
    