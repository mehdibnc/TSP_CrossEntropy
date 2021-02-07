# TSP_CrossEntropy
Applying a Cross Entropy technique to the Travling Salesman Problem.


## Requirements

`Python 3.8.5` is used. Requirements needed : `pip install -r requirements.txt`.

## Data

The algorithm is tested on a few instances in `data` these were downloaded [here](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html).


## Method 

The general idea of Cross Entropy is to associate an estimation problem to the original optimisation problem, here a TSP. It relies on finding a good probability distribution to generate solutions to the TSP. This [book](https://www.springer.com/gp/book/9780387212401) provides a thorough overview of the method and its application. 


The algorithm converges quickly on small instances, in a few iterations, towards a degenerate distribution. When sampling this distribution you get a solution to the problem, with a cost close to the optimal one.

![alt text](https://github.com/mehdibnc/TSP_CrossEntropy/blob/master/figures/tsp_convergence_15_cities_291.png)

Note that the default parameters of this implementation include a sample size equal to `5 * c^2` where `c` is the number of cities in the instance. Given the nature of the method, performance is highly sensitive to this parameter. A larger sample comes with better monte carlo approximations and therefore solutions of better quality but also increases the computationnal load. Decreasing the value will make the algorithm faster but will likely decrease solutions quality. With this value, we get good results event for the largest instance tested here.


[a link](https://github.com/mehdibnc/TSP_CrossEntropy/figures/tsp_convergence_48_cities_33523.png?)