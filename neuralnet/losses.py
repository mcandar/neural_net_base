import numpy as np

class MSE:
    def __init__(self):
        pass
    
    def error(self,x,y):
        return np.mean((x-y)**2)
    
    def grad_output(self,y,layer,act_deriv):
        return (y - layer) * act_deriv(layer)

    def grad_hidden(self,last_grad,weight,act_deriv,layer):
        return np.dot(last_grad,weight.T) * act_deriv(layer)