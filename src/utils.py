""" Utils for data structures and small helper functions. 

"""
import numpy as np
from numba import njit 



def load_data():
    """ Loads TSP instances
    
        Returns:
            data: dict, contains distance matrix and optimal value of tour.
    
    """

    dist_matrix_15 = np.loadtxt("TSP_CrossEntropy/data/tsp_15_291.txt") #15 cities and min cost = 291
    dist_matrix_26 = np.loadtxt("TSP_CrossEntropy/data/tsp_26_937.txt")
    dist_matrix_17 = np.loadtxt("TSP_CrossEntropy/data/tsp_17_2085.txt")
    dist_matrix_42 = np.loadtxt("TSP_CrossEntropy/data/tsp_42_699.txt")
    dist_matrix_48 = np.loadtxt("TSP_CrossEntropy/data/tsp_48_33523.txt")

    data = {15:(dist_matrix_15, 291),
            17:(dist_matrix_17, 2085),
            26:(dist_matrix_26, 937),
            42:(dist_matrix_42, 699),
            48:(dist_matrix_48, 33523)}
    return data 


def unif_matrix(m):
    """ Generate a transition matrix of size (m,m), with uniform probabilities.
        Such that transitionning to state i, when being in i happens with
        probability 0.

        Args:
            m: number of rows in the matrix to generate.

        Returns:
            p: transition matrix with uniform probabilities.
    """
    p = np.empty((m, m))
    p.fill(1 / (m - 1))
    for i in range(m):
        p[i,i] = 0
    return p

def init_structures(n: int, r: int):
    """ Initialize data structures to be used.

        Args:
            n: number of cities
            r: number of routes to sample at each step

        Returns:
            T: np.ndarray of shape (n, n), transition matrix
            Tp: np.ndarray of shape (n, n), copy of the transition matrix
            routes: np.ndarray of shape (r, n+1), array containing r routes,
            one per row. The first and last column will always be 0 from start/end.
            routes_distance: np.ndarray of shape (r,), the i-th row contain the distance
            of the i-th route in routes.
    
    """
    T = unif_matrix(n)
    Tp = unif_matrix(n)
    routes = np.zeros((r, n+1), dtype=np.int64)
    routes_distance = np.zeros((r,))
    unvisited = np.array([True]*n)
    return T, Tp, routes, routes_distance, unvisited



@njit(inline="always")
def cost(routes: np.ndarray, distances: np.ndarray, i: int):
    """ Computes the distance of the i-th route in routes.

        Args: 
            routes: data structure of all routes
            distances: distance table between cities
            i: id of the route to compute distance
        Returns:
            c: float, distance of i-th route in routes
    """
    c = 0
    for j in range(routes.shape[1] - 1):
        c += distances[routes[i, j], routes[i, j+1]]
    return c

@njit(inline="always")
def count_transition(routes: np.ndarray, 
                    i: int, 
                    j: int,
                    ids: np.ndarray):
    """ Counts the transition between cities i and j in the best theta
        (theta is the lenght of ids).
        routes. ids is table that gives the routes to look at when
        counting the transitions.

        Args:
            routes: table containing all routes, one per row
            i: id of city
            j: id of city
            ids: contains id of best routes.

        Returns:
            c: int, count of transitions. 
    """
    c = 0
    for r in ids:
        for k in range(routes.shape[1]-1):
            if routes[r, k] == i and routes[r, k+1] == j:
                c += 1
    return c 

@njit(inline="always")
def diff_matrix(P1: np.ndarray, P2: np.ndarray):
    """ Compute a distance between two matrices P1 and P2 defined as
        the element wise sum of absolute differences.
        
        Args:
            P1: matrix 1
            P2: matrix 2
        
        Returns:
            d: distance between P1 and P2
    """
    d = np.abs(P1 - P2)
    return np.sum(d)


