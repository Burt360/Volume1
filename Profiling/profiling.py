# profiling.py
"""Python Essentials: Profiling.
Nathan Schill
Section 2
Tues. Mar. 28, 2023
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter as pc
from numba import jit


# Problem 1
def max_path(filename='triangle.txt'):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename='triangle_large.txt'):
    """Find the maximum vertical path in a triangle of values."""

    # Read file into list of lists
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    
    # Start with penultimate row
    for i in range(len(data)-2, -1, -1):
        # Iterate across entries
        for j in range(len(data[i])):
            # Add to the current entry the max of the two entries below
            data[i][j] += max(data[i+1][j], data[i+1][j+1])
    
    # Max path sum is at top of triangle
    return data[0][0]


# Problem 2
### Hint: Prime factorizations -> only need to check primes
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""

    # Init with 2 since the only even prime
    primes = [2] + [None] * (N-1)
    num_primes = 1

    # Start at 3 and iterate until have N primes
    current = 3
    stop = 1
    stop_prime_sq = primes[stop-1]**2
    while True:
        # Need to check one more prime since sqrt(current) > stop_prime
        if current > stop_prime_sq:
            stop += 1
            stop_prime_sq = primes[stop-1]**2
        
        # Check whether each known prime (skipping 2 and less than stop_prime) divides current
        for p in primes[1:stop]:
            if current % p == 0:
                # Is not prime
                break
        # Is prime
        else:
            primes[num_primes] = current
            num_primes += 1
            
            if num_primes >= N:
                return primes
        
        current += 2


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    # Take norm of each column of A-x (x broadcasted to each column), and get argmin
    return np.argmin(np.linalg.norm(A-x[:, np.newaxis], axis=0))


# Problem 4
### Hint: Use comprehensions (list, dictionary, etc.) and NumPy array operations
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""

    # Read and sort names in file
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))

    # Create dictionary mapping letters to scores
    abc_scores = {L:S+1 for S, L in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    
    # Return total score from file
    return sum(
        [(1 + position)*
            sum(
                [abc_scores[letter] for letter in name]
            )
        for position, name in enumerate(names)]
    )


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    
    # First two fib terms
    f1, f2 = 1, 1
    yield f1
    yield f2

    # Yield following fib terms
    while True:
        sum = f1 + f2
        yield int(sum)
        f1 = f2
        f2 = sum

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    
    # Break when a fib term has N or more digits
    for i, n in enumerate(fibonacci()):
        if len(str(n)) >= N:
            return i+1


# Problem 6
### Numba: lab machines, ssh lab machines, or Colab
def prime_sieve(N):
    """Yield all primes that are less than N."""
    
    # Create list of 2 up to N
    L = list(range(2, N+1))

    # Continue while L is not empty
    while L:
        # Get first element
        n = L[0]

        # Delete all multiples of first element
        L = [i for i in L if i % n != 0]
        
        # Yield first element
        yield n


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    # Just copy above code with @jit decorator above
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    
    # Compile the function
    A = np.random.random((2,2))
    matrix_power_numba(A, 2)

    # Exponents of 2 for matrix sizes to time
    MIN, MAX = 2, 7
    N = 2**np.arange(MIN, MAX+1)

    # Lists to store times
    normal = np.zeros(MAX-MIN+1)
    numba = np.zeros(MAX-MIN+1)
    numpy = np.zeros(MAX-MIN+1)

    # Time each matrix size for each method
    for i, m in enumerate(N):
        A = np.random.random((m,m))
        
        # normal
        start = pc()
        matrix_power(A, n)
        end = pc()
        normal[i] = end - start
        
        # numba
        start = pc()
        matrix_power_numba(A, n)
        end = pc()
        numba[i] = end - start

        # numpy
        start = pc()
        np.linalg.matrix_power(A, n)
        end = pc()
        numpy[i] = end - start

    # Plot
    plt.loglog(N, normal, base=2, label='Pure Python')
    plt.loglog(N, numba, base=2, label='Numba')
    plt.loglog(N, numpy, base=2, label='NumPy')

    # Plot properties
    plt.legend()
    plt.xlabel('Matrix dimension')
    plt.ylabel('Time (seconds)')
    plt.title('Time matrix power implementations')
    plt.show()