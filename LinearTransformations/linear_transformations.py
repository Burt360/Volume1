# linear_transformations.py
"""Volume 1: Linear Transformations.
Nathan Schill
Section 3
Tues. Oct. 11, 2022
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
from time import time
from time import perf_counter as pc


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create transformation matrix and return product of left-multiplying A by it.
    transform = np.array([[a, 0], [0, b]])
    return transform@A

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create transformation matrix and return product of left-multiplying A by it.
    transform = np.array([[1, a], [b, 1]])
    return transform@A

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create transformation matrix and return product of left-multiplying A by it.
    transform = np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])/(a**2 + b**2)
    return transform@A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create transformation matrix and return product of left-multiplying A by it.
    transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return transform@A

def test_transformations():
    # Load the data.
    horse = np.load('horse.npy')

    # Make an empty list for the axes.
    axes = [None] * 6

    # Plot the original data.
    data = horse
    axes[0] = plt.subplot(231)
    axes[0].plot(data[0], data[1], 'k,')
    axes[0].set_title('Original')

    # Plot the stretched data.
    data = stretch(horse, 1/2, 6/5)
    axes[1] = plt.subplot(232)
    axes[1].plot(data[0], data[1], 'k,')
    axes[1].set_title('Stretch')

    # Plot the sheared data.
    data = shear(horse, 1/2, 0)
    axes[2] = plt.subplot(233)
    axes[2].plot(data[0], data[1], 'k,')
    axes[2].set_title('Shear')

    # Plot the reflected data.
    data = reflect(horse, 0, 1)
    axes[3] = plt.subplot(234)
    axes[3].plot(data[0], data[1], 'k,')
    axes[3].set_title('Reflection')

    # Plot the rotated data.
    data = rotate(horse, np.pi/2)
    axes[4] = plt.subplot(235)
    axes[4].plot(data[0], data[1], 'k,')
    axes[4].set_title('Rotation')

    # Plot the data through the composition of the four previous transformations in order.
    data = rotate(reflect(shear(stretch(horse, 1/2, 6/5), 1/2, 0), 0, 1), np.pi/2)
    axes[5] = plt.subplot(236)
    axes[5].plot(data[0], data[1], 'k,')
    axes[5].set_title('Composition')

    # Set attributes of each axes.
    for i in range(6):
        axes[i].set_aspect('equal')
        #axes[i].axis([-1,1,-1,1])
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)

    # Set plot title and show plot.
    plt.suptitle('Linear transformations')
    plt.show()

#test_transformations()

# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """

    # Create a list of equally spaced t-vals between 0 and T.
    ts = np.linspace(0, T, 150)

    # Compute the earth's coordinates at each time t.
    pe_0 = np.array([x_e, 0])
    pe = np.array([rotate(pe_0, t*omega_e) for t in ts]).T
    
    # Compute the moon's coordinates at each time t.
    pm_0 = np.array([x_m, 0])
    pm_rel_pe = np.array([rotate(pm_0 - pe_0, t*omega_m) for t in ts])
    pm = np.array([pm_t + pe_t for pm_t, pe_t in zip(pm_rel_pe, pe.T)]).T

    # Plot earth and moon coordinates.
    ax = plt.subplot(111)
    ax.plot(pe[0], pe[1], label='Earth')
    ax.plot(pm[0], pm[1], label='Moon')
    
    # Set aspect, set title, activate legend, label axes.
    ax.set_aspect('equal')
    plt.title('Earth and moon motion')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    # Show plot.
    plt.show()

#solar_system(3/2 * np.pi, 10, 11, 1, 13)


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    
    # Generate sizes for matrices and vector.
    N = 9
    ns = [2**n for n in range(N)]
    
    # Generate random matrices and vector.
    rand_ms0 = [random_matrix(2**n) for n in range(N)]
    rand_ms1 = [random_matrix(2**n) for n in range(N)]
    rand_vs = [random_vector(2**n) for n in range(N)]

    ### matrix_vector_product
    ax = plt.subplot(121)
    mv_times = [None] * N
    for i in range(N):
        start = time()
        matrix_vector_product(rand_ms0[i], rand_vs[i])
        end = time()
        mv_times[i] = end - start
    
    # Create plot.
    ax.plot(ns, mv_times, marker='o')
    ax.set_title('Matrix-Vector Multiplication')
    ax.set_xlabel('n')
    ax.set_ylabel('Seconds')

    ### matrix_matrix_product
    ax = plt.subplot(122)
    mm_times = [None] * N
    for i in range(N):
        start = time()
        matrix_matrix_product(rand_ms0[i], rand_ms1[i])
        end = time()
        mm_times[i] = end - start
    
    # Create plot.
    ax.plot(ns, mm_times, marker='o')
    ax.set_title('Matrix-Matrix Multiplication')
    ax.set_xlabel('n')

    # Show plots.
    plt.show()

#prob3()

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """

    # Use time.perf_counter. Rename time for convenience here.
    time = pc

    # Generate sizes for matrices and vector.
    N = 8
    ns = [2**n for n in range(N)]
    
    # Generate random matrices and vector.
    rand_ms0 = [random_matrix(2**n) for n in range(N)]
    rand_ms1 = [random_matrix(2**n) for n in range(N)]
    rand_vs = [random_vector(2**n) for n in range(N)]

    ##### Linear plot
    lin_ax = plt.subplot(121)
    ##### Log-log plot
    log_ax = plt.subplot(122)

    ### matrix_vector_product
    mv_times = [None] * N
    for i in range(N):
        start = time()
        matrix_vector_product(rand_ms0[i], rand_vs[i])
        end = time()
        mv_times[i] = end - start
    lin_ax.plot(ns, mv_times, marker='o', label='matrix_vector_product')
    log_ax.loglog(ns, mv_times, marker='o', label='matrix_vector_product', base=2)

    ### matrix_matrix_product
    mm_times = [None] * N
    for i in range(N):
        start = time()
        matrix_matrix_product(rand_ms0[i], rand_ms1[i])
        end = time()
        mm_times[i] = end - start
    lin_ax.plot(ns, mm_times, marker='o', label='matrix_matrix_product')
    log_ax.loglog(ns, mm_times, marker='o', label='matrix_matrix_product', base=2)

    ### np matrix-vector
    np_mv_times = [None] * N
    for i in range(N):
        start = time()
        np.dot(rand_ms0[i], rand_vs[i])
        end = time()
        np_mv_times[i] = end - start
    lin_ax.plot(ns, np_mv_times, marker='o', label='np matrix-vector')
    log_ax.loglog(ns, np_mv_times, marker='o', label='np matrix-vector', base=2)

    ### np matrix-matrix
    np_mm_times = [None] * N
    for i in range(N):
        start = time()
        np.dot(rand_ms0[i], rand_ms1[i])
        end = time()
        np_mm_times[i] = end - start
    lin_ax.plot(ns, np_mm_times, marker='o', label='np matrix-matrix')
    log_ax.loglog(ns, np_mm_times, marker='o', label='np matrix-matrix', base=2)

    # Create linear plot.
    lin_ax.set_title('Linear plot')
    lin_ax.set_xlabel('n')
    lin_ax.set_ylabel('Seconds')
    lin_ax.legend(loc='upper left')

    # Create log-log plot.
    log_ax.set_title('Log-log plot')
    log_ax.set_xlabel('n')
    log_ax.legend(loc='upper left')

    # Show plots.
    plt.show()

#prob4()