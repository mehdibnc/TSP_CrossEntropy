# TSP_CrossEntropy
Solving the Traveling Salesman Problem using a cross entropy technique


## Requirements

`Python 3.8.5` is used. Requirements needed : `pip install -r requirements.txt`.

## Data

The algorithm is tested on a few instances in `data` these were downloaded [here](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html).


## Method 

We use a cross entropy method to solve the TSP. The general idea is to associate an estimation problem to the original optimisation problem of the TSP. It relies on finding a good probability distribution to generate solutions to the TSP. See Method.pdf for details about the method and the algorithm implemented in this repo. 

