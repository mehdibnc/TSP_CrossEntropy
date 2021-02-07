from utils import init_structures, load_data
from sampling import generate_all_route, generate_route
from model import TSP_CE_Solver
import numpy as np 
import time 

def main():
    """ 
        Runs CE solver on benchmark TSP instances.   
    
    """
    #Loading benchmark instances
    data = load_data()
    #-- Running solver
    for c in data:
        distances_data = data[c][0]
        print(f"Instance at {c} cities")
        start = time.time()
        TSP_CE_Solver(distances_data,
                    0.1,
                    5*c**2,
                    n_iter = 100,
                    tag= f'{c}_cities_{data[c][1]}',
                    cout_th = data[c][1])
        print(f'Running time : {time.time()-start}') # note that with current implementation, first run also counts compilation

if __name__ == '__main__':
    main()
