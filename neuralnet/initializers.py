import numpy as np

def standard_gaussian(row,col):
    return np.random.normal(0,1,(row,col))

def uniform(row,col):
    return 2*np.random.random((row,col)) - 1