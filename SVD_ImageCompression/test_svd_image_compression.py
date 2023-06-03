from svd_image_compression import *
import pytest

import numpy as np
from scipy import linalg as la

def test_compact_svd():
    for m in range(1, 4):
        for n in range(1, 4):
            A = np.random.rand(m, n)
            U1, S1, V1h = compact_svd(A)
            
            U1hU1 = U1.conj().T@U1
            V1hV1 = V1h@V1h.conj().T

            assert np.allclose(U1hU1, np.identity(U1hU1.shape[0]))
            assert np.allclose(V1hV1, np.identity(V1hV1.shape[0]))
            assert np.allclose(A, U1@np.diag(S1)@V1h)
            assert np.linalg.matrix_rank(A) == len(S1)

def test_visualize_svd():
    A = [[3, 1], [1, 3]]
    #visualize_svd(A)

    A = np.random.rand(2, 2)
    #visualize_svd(A)

def test_svd_approx():
    for m in range(2, 5):
        for n in range(2, 5):
            for s in range(1, 6):
                A = np.random.rand(m, n)
                
                if s > np.linalg.matrix_rank(A):
                    with pytest.raises(ValueError):
                      A_s, bytes = svd_approx(A, s)
                    continue
                    
                A_s, num_entries = svd_approx(A, s)

                '''print(A)
                print(A_s)
                print()'''
                assert np.linalg.matrix_rank(A_s) == s
                assert num_entries == m*s + s + s*n

def test_lowest_rank_approx():
    for m in range(2, 5):
        for n in range(2, 5):
            for err in range(1, 10):
                err /= 10

                A = np.random.rand(m, n)

                try:
                    A_s, num_entries = lowest_rank_approx(A, err)

                    '''print(A)
                    print(A_s)
                    print(num_entries, A.size)
                    print()'''

                    # sigma_{s+1} < err
                    assert la.norm(A - A_s, ord=2) < err
                except ValueError:
                    # sigma_r >= err
                    assert la.norm(A, ord=2) >= err

def test_compress_image():
    compress_image('hubble_gray.jpg', 20)
    compress_image('hubble.jpg', 20)