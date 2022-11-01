# image_segmentation.py
"""Volume 1: Image Segmentation.
Nathan Schill
Section 3
Tues. Nov. 8, 2022
"""

import math
import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from imageio import imread
from matplotlib import pyplot as plt


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """

    # Sum along rows to get degree matrix, and subtract A
    return np.diag(np.sum(A, axis=1)) - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    
    # Get the Laplacian
    L = laplacian(A)

    # Keep the real parts of each eigenvalue, sort them in ascending order, and
    # convert to zero any that are less than tol
    eigvals = tuple(0 if ev < tol else ev for ev in sorted(np.real(la.eigvals(L))))

    return eigvals.count(0), eigvals[1]    


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """

    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int64), R[mask]
    ### Was originally np.int, but gave deprecation warning ###


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file.
        Store a scaled version.
        Store its brightness values as a flat array.
        Store a flattened version."""
        
        # Read and scale the image
        self.scaled = imread(filename) / 255

        # If the image is 2D, then it's grayscale. Otherwise (if it's 3D), it's color.
        self.isgray = len(self.scaled.shape) == 2
        
        if self.isgray:
            # If the image is grayscale, the brightness is the regular image array
            self.brightness = self.scaled
        else:
            # Otherwise, the image is color, so the brightness averages the RGB values of each pixel
            self.brightness = self.scaled.mean(axis=2)
        
        # Flatten the brightness
        self.flat_brightness = np.ravel(self.brightness)

    # Problem 3
    def show_original(self):
        """Display the original image."""
        
        # Display either grayscale or color
        if self.isgray:
            plt.imshow(self.scaled, cmap='gray')
        else:
            plt.imshow(self.scaled)
        
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""

        # height, width
        m, n = self.scaled.shape[0], self.scaled.shape[1]

        # Create A as an mnxmn sparse matrix and D as a vector of length mn
        A = sparse.lil_matrix((m*n,m*n))
        D = [None] * m*n

        for i in range(m*n):
            # For each vertex, get its neighbors and the distances to each
            neighbors, distances = get_neighbors(i, r, m, n)

            # Calculate the weight for each neighbor
            weights = [math.exp(
                        - abs(self.flat_brightness[i] - self.flat_brightness[j]) / sigma_B2
                        - distances[k]/sigma_X2)
                        for k, j in enumerate(neighbors)]
            
            # Set the weights for vertex i's neighbors
            A[i, neighbors] = weights

            # Store the degree of vertex i
            D[i] = sum(weights)

        # Convert A to a csc_matrix
        A = sparse.csc_matrix(A)

        return A, D
    
    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""

        # height, width
        m, n = self.scaled.shape[0], self.scaled.shape[1]
        
        # Get Laplacian and D^(-1/2)
        L = sparse.csgraph.laplacian(A)
        D_n_one_half = sparse.diags([1/math.sqrt(i) for i in D])

        # Compute D^(-1/2) L D^(-1/2)
        product = D_n_one_half @ L @ D_n_one_half

        # Get the second smallest eigenvalue and reshape it to be mxn
        ev_mat = sp_linalg.eigsh(product, which='SM', k=2)[1][:,1].reshape((m,n))

        # Create the mask of entries greater than zero
        mask = ev_mat > 0

        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        
        # Get mask
        A, D = self.adjacency()
        mask = self.cut(A, D)

        # Create 1 row of 3 subplots
        fig, subplots = plt.subplots(1,3)

        if self.isgray:
            # Original
            subplots[0].imshow(self.scaled, cmap='gray')
            # Segment 1
            subplots[1].imshow(mask*self.scaled, cmap='gray')
            # Segment 2
            subplots[2].imshow(~mask*self.scaled, cmap='gray')
        else:
            # Original
            subplots[0].imshow(self.scaled)
            # Segment 1
            subplots[1].imshow(np.dstack((mask*self.scaled[:,:,0], mask*self.scaled[:,:,1], mask*self.scaled[:,:,2])))
            # Segment 2
            subplots[2].imshow(np.dstack((~mask*self.scaled[:,:,0], ~mask*self.scaled[:,:,1], ~mask*self.scaled[:,:,2])))
        
        # Turn off axes
        [subplots[i].axis('off') for i in range(3)]
        plt.show()


# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
