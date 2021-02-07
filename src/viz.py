""" 
Utils for cross entropy and visualisations.
"""
import matplotlib.pyplot as plt 
import numpy as np 


def viz_convergence(steps: np.ndarray,
                    mean_cost:np.ndarray, 
                    err_bounds: np.ndarray, 
                    tag: str = "",
                    cout_th: float = None):
    """cout_emp is a list of empirical costs
     cout_th is the minimal theoritical cost if known
     std is the list of standard deviation
     Function returns a plot of the evolution of cout_emp
    """
    loc = f"TSP_CrossEntropy/figures/tsp_convergence_{tag}.png"
    plt.figure(figsize = (10,8))
    if cout_th:
        plt.axhline(y = cout_th, color = 'g', linestyle = '-', label = 'Minimal possible cost')
    plt.errorbar(steps, mean_cost, yerr=err_bounds, linestyle = 'None', marker = 'o',ecolor = 'r')
    plt.legend(loc = "upper right")
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Average cost of sampled routes. \n Red bar indicate quantiles 0.25 and 0.75.')
    plt.savefig(loc)
    return None


