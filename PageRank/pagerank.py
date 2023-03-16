# solutions.py
"""Volume 1: The Page Rank Algorithm.
Nathan Schill
Section 2
Tues. Mar. 21, 2023
"""

import numpy as np
import networkx as nx
from itertools import combinations


# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        
        # Get n
        self.n = A.shape[0]

        # Check whether number of labels matches number of nodes
        # Create labels if none provided
        if labels is not None:
            if len(labels) != self.n:
                raise ValueError('number of labels not equal to number of nodes')
        else:
            labels = [str(i) for i in range(self.n)]
        
        # Store labels
        self.labels = labels

        # Get column indices of sinks by checking columns that sum to 0
        # Replace sink columns with ones
        sink_col_indices = np.where(A.sum(axis=0) == 0)[0]
        A[:, sink_col_indices] = np.ones((self.n, len(sink_col_indices)))
        
        # Divide by column sums
        self.Ahat = A/A.sum(axis=0)


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # Set up and solve equation (13.6)
        left = np.eye(self.n) - epsilon*self.Ahat
        right = (1-epsilon)/self.n * np.ones(self.n)
        soln = np.linalg.solve(left, right)

        return {label:val for label, val in zip(self.labels, soln)}

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        
        # Define B as in equation (13.7) and get its eigenstuff
        B = epsilon*self.Ahat + (1-epsilon)/self.n * np.ones((self.n,self.n))
        evals, evects = np.linalg.eig(B)

        # The the sorted order of the evals (by magnitude)
        sorted_indices = np.abs(evals).argsort()[::-1]

        # Transpose the evects matrix, sort them using the eval sort, and
        # get the first (corresponding to eval 1), then normalize by the 1-norm
        p = evects.T[sorted_indices][0]
        p /= np.linalg.norm(p, ord=1)

        return {label:val for label, val in zip(self.labels, p)}
        

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        # Define B as in equation (13.7)
        B = epsilon*self.Ahat + (1-epsilon)/self.n * np.ones((self.n,self.n))
        
        # Initial p0
        p0 = np.ones(self.n)/self.n
        
        for _ in range(maxiter):
            # Compute next vector
            prod = B @ p0
            p1 = prod/np.linalg.norm(prod, ord=1)

            # Break if tol reached
            if np.linalg.norm(p1 - p0, ord=1) < tol:
                break
            
            p0 = p1
        
        return {label:val for label, val in zip(self.labels, p0)}


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """

    # Sort d by keys (alphabetically)
    d = {key : d[key] for key in sorted(d)}
    
    # Sort by value (get), and reverse to sort from largest to smallest
    return sorted(d, key=d.get)[::-1]


# Problem 4
def rank_websites(filename='web_stanford.txt', epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    
    # Get each line in a list
    with open(filename) as file:
        lines = file.read().splitlines()

    # Get unique ids
    unique_ids_set = set()
    for line in lines:
        # Split ids
        iids = line.split('/')
        
        for iid in iids:
            if iid not in unique_ids_set:
                unique_ids_set.add(iid)
    
    # Sort ids alphabetically (as strings), then construct dictionary
    # mapping ids to indices in adjacency matrix (to be created)
    sorted_iids = sorted(list(unique_ids_set))
    unique_ids = dict()
    for i, iid in enumerate(sorted_iids):
        unique_ids[iid] = i
    
    # Construct adjacency matrix
    n = len(sorted_iids)
    A = np.zeros((n,n))
    for line in lines:
        # Split ids
        iids = line.split('/')

        # The current page and its index in A
        iid = iids[0]
        j = unique_ids[iid]

        # For each id pointed to by iid (the current page),
        # increment the corresponding location in A
        for point_to_id in iids[1:]:
            i = unique_ids[point_to_id]
            A[i, j] += 1
    
    # Construct graph and return labels in ranked order
    dg = DiGraph(A, labels=sorted_iids)
    ranks = dg.itersolve(epsilon=epsilon)

    return get_ranks(ranks)


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    
    # Get each line in a list
    with open(filename) as file:
        # Skip the first line (the header)
        file.readline()
        lines = file.read().splitlines()
    
    # Get unique teams
    unique_teams_set = set()
    for line in lines:
        # Split teams
        teams = line.split(',')
        
        for team in teams:
            if team not in unique_teams_set:
                unique_teams_set.add(team)
    
    # Sort teams alphabetically, then construct dictionary
    # mapping teams to indices in adjacency matrix (to be created)
    sorted_teams = sorted(list(unique_teams_set))
    unique_teams = dict()
    for i, team in enumerate(sorted_teams):
        unique_teams[team] = i
    
    # Construct adjacency matrix
    n = len(sorted_teams)
    A = np.zeros((n,n))
    for line in lines:
        # Split teams
        teams = line.split(',')

        # The winning team and its index in A
        winner = teams[0]
        i = unique_teams[winner]

        # The losing team and its index in A
        loser = teams[1]
        j = unique_teams[loser]

        # Record winner beating loser in A
        A[i, j] += 1
    
    # Construct graph and return labels in ranked order
    dg = DiGraph(A, labels=sorted_teams)
    ranks = dg.itersolve(epsilon=epsilon)

    return get_ranks(ranks)


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """

    # Create directed graph
    dg = nx.DiGraph()

    # Read file
    with open(file=filename, encoding='utf-8') as file:
        lines = file.read().splitlines()
    
    # Populate graph
    for line in lines:
        # Split names, skipping the movie title
        names = line.split('/')[1:]
        
        # Get pairs of names in order
        pairs = list(combinations(names, 2))

        for pair in pairs:
            # Edges point from second actor to first actor in pair
            first, second = pair
            if dg.has_edge(second, first):
                dg[second][first]['weight'] += 1
            else:
                dg.add_edge(second, first, weight=1)
    
    # Rank actors
    ranks = nx.pagerank(dg, alpha=epsilon)
    return get_ranks(ranks)