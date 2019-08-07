import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self,x):
        return 1/(1+np.exp(-x))
    
    def backward(self,x):
        return x*(1-x)

class ReLU:
    def __init__(self):
        pass

    def forward(self,x):
        return np.maximum(0,x)
    
    def backward(self,x):
        output = np.zeros_like(x)
        output[x > 0] == 1
        return output