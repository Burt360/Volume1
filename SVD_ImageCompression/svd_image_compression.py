# svd_image_compression.py
"""Volume 1: The SVD and Image Compression.
Nathan Schill
Section 3
Tues. Nov. 15, 2022
"""

import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    
    # Get eigenvalues of AhA, get V, then take Vh
    evals, V = la.eigh(A.conj().T@A)
    Vh = V.conj().T
    
    # Compute the singular values of A
    svals = np.sqrt(evals, dtype=complex)

    # Get the indices of the singular values in sorted order, then sort S and Vh
    sorted_indices = np.argsort(svals)[::-1]
    S = svals[sorted_indices]
    Vh = Vh[sorted_indices]

    # Get the number of positive (nonzero) singular values
    r = sum(sval > tol for sval in svals)

    # Keep only r singular values and vectors in Vh
    S1 = S[:r]
    V1h = Vh[:r]

    # Get V1 from Vh
    V1 = V1h.conj().T

    # Compute U1
    U1 = A@V1/S1

    return U1, S1, V1h


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    
    # 200 points in [0, 2*pi]
    theta = np.linspace(0, 2*np.pi, 200)

    # Compute C (circle) and E
    C = [np.cos(theta), np.sin(theta)]
    E = [[1, 0, 0], [0, 0, 1]]

    # Compute SVD of A
    U, S, Vh = la.svd(A)
    S = np.diag(S)
    
    # Create four subplots
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    # Plot E and C
    axs[0].plot(E[0], E[1])
    axs[0].plot(C[0], C[1])
    axs[0].set_title('C')

    # Plot Vh@E and Vh@C
    VhE = Vh@E
    VhC = Vh@C
    axs[1].plot(VhE[0], VhE[1])
    axs[1].plot(VhC[0], VhC[1])
    axs[1].set_title('Vh@C')

    # Plot Vh@E and Vh@C
    SVhE = S@VhE
    SVhC = S@VhC
    axs[2].plot(SVhE[0], SVhE[1])
    axs[2].plot(SVhC[0], SVhC[1])
    axs[2].set_title('S@Vh@C')

    # Plot u@Vh@E and U@Vh@C
    USVhE = U@SVhE
    USVhC = U@SVhC
    axs[3].plot(USVhE[0], USVhE[1])
    axs[3].plot(USVhC[0], USVhC[1])
    axs[3].set_title('U@S@Vh@C')

    # Set aspect for each subplot, and set view limits
    for ax in axs:
        ax.set_aspect('equal')
        LIM = 4.1
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)

    # Fix spacing and show plot
    plt.tight_layout()
    plt.show()

# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    TOL = 1e-6
    
    # Get SVD
    U, S, Vh = la.svd(A)

    # Compute compact SVD
    r = sum(sval > TOL for sval in S)
    U1 = U[:,:r]
    S1 = S[:r]
    V1h = Vh[:r]

    if s > r:
        raise ValueError('s is greater than rank(A) (the number of singular values)')
    
    # Compute truncated SVD
    Us = U[:,:s]
    Ss = S[:s]
    V1s = Vh[:s]

    return Us@np.diag(Ss)@V1s, sum((Us.size, s, V1s.size))
    


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    TOL = 1e-6
    
    # Get SVD
    U, S, Vh = la.svd(A)

    # Compute r (rank(A)), then find s (the number of singular values to include
    # such that sigma_{s+1} < err)
    r = sum(sval > TOL for sval in S)
    s = sum(sval >= err for sval in S[:r])

    if s == r:
        # All of the singular values are greather than or equal to err,
        # so err is less than or equal to the smallest singular value
        raise ValueError('err is less than or equal to the smallest singular value of A')
    
    # Compute truncated SVD
    Us = U[:,:s]
    Ss = S[:s]
    V1s = Vh[:s]

    return Us@np.diag(Ss)@V1s, sum((Us.size, s, V1s.size))



# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    
    # Read image, determine whether gray, and set cmap accordingly
    image = imread(filename) / 255
    isgray = True if len(image.shape) == 2 else False
    cmap_val = 'gray' if isgray else None

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2)
    
    # Show original image
    axs[0].imshow(image, cmap=cmap_val)
    axs[0].axis('off')
    axs[0].set_title('original')

    if isgray:
        # Get the approximated image and the number of entries
        approx_image, num_entries = svd_approx(image, s)
    else:
        # For each layer, get the approximated image and the number of entries
        red, red_entries = svd_approx(image[:,:,0], s)
        green, green_entries = svd_approx(image[:,:,1], s)
        blue, blue_entries = svd_approx(image[:,:,2], s)

        # Create the approximated color image and the total number of entries
        approx_image = np.dstack((red, green, blue))
        num_entries = red_entries + green_entries + blue_entries
    
    # Clip values outside [0, 1]
    approx_image = np.clip(approx_image, 0, 1)

    # Show approximated image
    axs[1].imshow(approx_image, cmap=cmap_val)
    axs[1].axis('off')
    axs[1].set_title(f'rank {s} approximation')
    
    # Compute number of entries saved and title plot
    entries_saved = image.size - num_entries
    plt.suptitle(f'Entries saved: {entries_saved}')

    # Show plot
    plt.show()