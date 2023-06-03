from tkinter import Image
from image_segmentation import *
import pytest

import numpy as np
from scipy.sparse.csgraph import laplacian as splap

def test_laplacian():
    A1 =  np.array([[0, 1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1],
                    [1, 1, 0, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0]])
    
    A2 =  np.array([[0, 3, 0,  0, 0,  0,],
                    [3, 0, 0,  0, 0,  0],
                    [0, 0, 0,  1, 0,  0],
                    [0, 0, 1,  0, 2, .5],
                    [0, 0, 0,  2, 0,  1],
                    [0, 0, 0, .5, 1,  0]])
    
    assert np.allclose(laplacian(A1), splap(A1))
    assert np.allclose(laplacian(A2), splap(A2))

def test_connectivity():
    A1 =  np.array([[0, 1, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 1],
                    [1, 1, 0, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0]])
    
    A2 =  np.array([[0, 3, 0,  0, 0,  0,],
                    [3, 0, 0,  0, 0,  0],
                    [0, 0, 0,  1, 0,  0],
                    [0, 0, 1,  0, 2, .5],
                    [0, 0, 0,  2, 0,  1],
                    [0, 0, 0, .5, 1,  0]])

    test1 = connectivity(A1)
    assert test1[0] == 1
    print('A1 alg. connect.: ', test1[1])

    test2 = connectivity(A2)
    assert np.allclose(test2, (2, 0))

def test_prob3():
    dream = ImageSegmenter('dream.png')
    print(dream.scaled.shape)
    print(dream.brightness.shape)
    print(dream.flat_brightness.shape)

    #dream.show_original()

def test_prob4():
    dream = ImageSegmenter('dream.png')
    A, D = dream.adjacency()

    print(A.shape)
    print(len(D))

def test_prob5():
    dream = ImageSegmenter('dream.png')
    A, D = dream.adjacency()

    print(dream.cut(A, D))

def test_prob6():
    ImageSegmenter('dream.png').segment()

    ImageSegmenter('dream_gray.png').segment()