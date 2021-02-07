""" Sampling functions and utils

"""
from numba import njit
import numpy as np 
from utils import cost

@njit(inline="always")
def generate_route(i: int,
                    T: np.ndarray, 
                    Tp: np.ndarray,
                    routes: np.ndarray,
                    unvisited: np.ndarray,
                    routes_distance: np.ndarray,
                    distances: np.ndarray,
                    cost: callable = cost):
    """ Generates a route and stores it in the i-th row
        of routes. Route distance are also updated.

        Args:
            i: id of route generated, must be < routes.shape[0]
            T: current Transition matrix
            Tp: Transition matrix companion for copies
            routes: table containing all routes
            unvisited: table containing boolean indicating whether 
                       each city has been visited in current route
            routes_distance: table containing distances associated to each route
            distances: distance matrix between cities
            cost: function to compute a route's distance.
        
        Returns:
            None

    """
    assert i < routes.shape[0], "Route index is out of range"
    #-- Restore Tp's values to T's
    Tp[:,:] = T[:,:]
    #-- Restore unvisited cities
    unvisited[:] = True
    unvisited[0] = False
    #-- Iterate to generate visits
    for k in range(T.shape[0]-2):
        #- Prevents transtionning to current state
        Tp[:, routes[i, k]] = 0.00000000001 # this should be an exact 0, putting 0 causes numba errors in the multinomial sampling.
        # investiguate this further or transition out of numba.
        #- RE-normalize rows
        # Be numba friendly
        for row in range(Tp.shape[0]):
            Tp[row, :] = Tp[row, :] / np.sum(Tp[row, :])
        # row_sums = Tp.sum(axis=1)
        # Tp = Tp / row_sums[:, np.newaxis]
        
        #- Sample next city to visit
        draw = np.random.multinomial(1, Tp[routes[i, k], :], 1)
        next_visit = np.where(draw == 1)[1][0]
        routes[i, k+1] = int(next_visit)
        #- Update unvisited state
        unvisited[int(next_visit)] = False
    #-- Assign last visit
    routes[i, k+2] = np.where(unvisited)[0][0]
    #-- Update distance table 
    routes_distance[i] = cost(routes, distances, i)
    return None 


@njit(inline="always")
def generate_all_route(T: np.ndarray, 
                       Tp: np.ndarray,
                       routes: np.ndarray,
                       unvisited: np.ndarray,
                       routes_distance: np.ndarray,
                       distances: np.ndarray):
    """ Generates all routes and update all distances.

        Args:
            T: current Transition matrix
            Tp: Transition matrix companion for copies
            routes: table containing all routes
            unvisited: table containing boolean indicating whether 
                       each city has been visited in current route
            routes_distance: table containing distances associated to each route
            distances: distance matrix between cities
        
        Returns:
            None

    """
    for i in range(routes.shape[0]):
        generate_route(i,
                        T, 
                        Tp,
                        routes,
                        unvisited,
                        routes_distance,
                        distances)




