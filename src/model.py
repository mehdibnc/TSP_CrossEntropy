"""
Main model - Cross entropy


"""
from numba import njit 
import numpy as np 
from sampling import generate_all_route
from utils import count_transition, diff_matrix, init_structures
from viz import viz_convergence

@njit(inline="always")
def iteration(T: np.ndarray, 
            Tp: np.ndarray,
            routes: np.ndarray,
            unvisited: np.ndarray,
            routes_distance: np.ndarray,
            distances: np.ndarray,
            theta: float,
            generate_all_route = generate_all_route,
            count_transition = count_transition):
    """ Perfoms one iteration of CE, sample routes, retrieve best theta%
        and update transition matrix.
        T is modified in place.

        Args:
            T: current Transition matrix
            Tp: Transition matrix companion for copies
            routes: table containing all routes
            unvisited: table containing boolean indicating whether 
                       each city has been visited in current route
            routes_distance: table containing distances associated to each route
            distances: distance matrix between cities
            theta: proportion of routes to keep for params estimation

        Returns:
            mu: mean distance of the best theta % routes 
            low_b: quantile 0.25 of the best theta % routes 
            up_b: quantile 0.75 of the best theta % routes 
    
    TODO: returning mean, low_b and up_b creates additionnal slices of array
          not necessary to keep it to get better perf. Add an option to not do it.
          - Set min value for probabilities as parameter.

    """
    #-- Sample routes
    generate_all_route(T, 
                        Tp,
                        routes,
                        unvisited,
                        routes_distance,
                        distances)
    #-- Retrives best ids
    sample_best_size = int(theta * routes.shape[0])
    best_ids = np.argsort(routes_distance)[:sample_best_size]
    Tp[:, :] = T[:, :]
    #-- Count transition
    for i in range(routes.shape[1] - 1):
        for j in range(routes.shape[1] - 1):
            count = count_transition(routes, i, j, best_ids)
            T[i, j] = count / sample_best_size if count > 0 else 0.00001
        #- Normalize 
        T[i, :] = T[i, :] / np.sum(T[i, :])

    return np.mean(routes_distance[best_ids]), np.quantile(routes_distance[best_ids], 0.25), np.quantile(routes_distance[best_ids], 0.75)

@njit
def CE(T: np.ndarray, 
            Tp: np.ndarray,
            routes: np.ndarray,
            unvisited: np.ndarray,
            routes_distance: np.ndarray,
            distances: np.ndarray,
            theta: float,
            n_iter: int,
            epsilon: float,
            n_stag: int,
            diff_matrix = diff_matrix):
    """ Iterates the CE procedures until max number of iterations
        of stagnation

        Args:
            T: current Transition matrix
            Tp: Transition matrix companion for copies
            routes: table containing all routes
            unvisited: table containing boolean indicating whether 
                       each city has been visited in current route
            routes_distance: table containing distances associated to each route
            distances: distance matrix between cities
            theta: proportion of routes to keep for params estimation

        Returns:
            

    """ 
    err_bounds = np.zeros((2, n_iter))    
    means = np.zeros((n_iter,))
    stag = 0
    np.random.seed(0)
    for it in range(n_iter):
        mu, low_b, up_b = iteration(T, 
                                    Tp,
                                    routes,
                                    unvisited,
                                    routes_distance,
                                    distances,
                                    theta)
        means[it] = mu 
        err_bounds[0, it] = mu - low_b
        err_bounds[1, it] = up_b - mu
        #- Stagnation update
        diff = diff_matrix(T, Tp)
        if diff < epsilon:
            stag += 1
        else:
            stag = 0
        #- Early stop^
        if stag >= n_stag:
            break 
    return means[:it], err_bounds[:, :it], it



def TSP_CE_Solver(distances: np.ndarray,
                  theta: float,
                  R: int,
                  n_iter: int,
                  epsilon: float = 0.005,
                  n_stag: int = 10,
                  tag: str = '',
                  cout_th: float = None):
    """ Solves the TSP instance and save monitoring figures.
    
    """
    #-- initialize data structures
    T, Tp, routes, routes_distance, unvisited = init_structures(distances.shape[0], R)
    #-- Run cross entropy
    mu, err_bounds, it = CE(T, 
                            Tp,
                            routes,
                            unvisited,
                            routes_distance,
                            distances,
                            theta,
                            n_iter,
                            epsilon,
                            n_stag)
    #-- Generates monitoring figures
    viz_convergence(np.arange(it),
                    mu, 
                    err_bounds, 
                    tag = tag,
                    cout_th = cout_th)




