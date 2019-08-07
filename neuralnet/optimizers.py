import numpy as np

class GradientDescent:
    def __init__(self,loss,activation,learning_rate):
        self.loss = loss
        self.activation = activation
        self.learning_rate = learning_rate

    def optimize(self,y,weights,layer):
        # Back propagate the errors using the chain rule, and update weights
        k = len(weights)-1
        grad = self.loss.grad_output(y,layer[k+1],self.activation.backward)
        weights[k] += np.dot(layer[k].T,grad)*self.learning_rate

        for i in range(k-1,-1,-1):
            grad = self.loss.grad_hidden(grad,weights[i+1],self.activation.backward,layer[i+1])
            weights[i] += np.dot(layer[i].T,grad)*self.learning_rate

        return weights