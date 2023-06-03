import pytest
from linear_systems import *

def test_ref():
    A = np.array([[1,2,3],[4,5,6],[7,8,10]], dtype=float)
    ref_goal = np.array([[1,2,3],[0,-3,-6],[0,0,1]], dtype=float)

    assert np.allclose(ref(A), ref_goal)

def test_lu():
    A = np.array([[1,2,3],[4,5,6],[7,8,10]], dtype=float)

    u_goal = np.array([[1,2,3],[0,-3,-6],[0,0,1]], dtype=float)
    l_goal = np.array([[1,0,0],[4,1,0],[7,2,1]])

    l, u = lu(A)

    assert np.allclose(u, u_goal)
    assert np.allclose(l, l_goal)

def test_solve():
    A = np.array([[1,2,3],[4,5,6],[7,8,10]], dtype=float)
    b = np.array([1,2,3])

    l, u = lu(A)

    x = solve(A, b)
    x_goal = np.array([[-1/3, 2/3, 0]], dtype=float)
    
    assert np.allclose(x, x_goal)


    C = np.array([[8.0, 8.0, 9.0, 2.0, 3.0], [1.0, 8.0, 2.0, 0.0, 1.0],
                  [6.0, 1.0, 2.0, 1.0, 0.0], [1.0, 3.0, 5.0, 9.0, 5.0], [3.0, 7.0, 0.0, 6.0, 7.0]], dtype=float)
    d = np.array([9., 4., 0., 7., 2.])

    l, u = lu(C)
    y = solve(C, d)
    y_goal = [-0.41082639, 0.26716065, 1.06689307, 0.06401153, 0.13975507]

    assert np.allclose(y, y_goal)