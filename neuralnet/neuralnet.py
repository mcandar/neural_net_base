import numpy as np
from neuralnet.utils import prep_batch
from neuralnet.optimizers import GradientDescent


class NeuralNet:
    def __init__(self,hidden,initializer,optimizer):
        self.hidden = hidden
        self.initializer = initializer
        self.training_error = []
        self.validation_error = []
        self.layer = None
        self.optimizer = optimizer
    
    def __compile(self,X,y,seed):
        "Prepare for training."
        self.weight = self.__weights(y.shape[1],X.shape[1],self.hidden,seed=seed)
        self.layer = [0 for i in range(len(self.weight)+1)]
        self.error = [0 for i in range(len(self.weight))]
        self.delta = self.error.copy()
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        
    def __weights(self,row,col,hidden,seed):
        "Create connections."
        np.random.seed(seed)
        num_hidden = len(hidden)
        result = []
        result.append(self.initializer(col,hidden[0])) # input weights
        for i in range(1,num_hidden):
            result.append(self.initializer(hidden[i-1],hidden[i])) # hidden layer connections with neighbor layers
        result.append(self.initializer(hidden[-1],row)) # output weights
        return result
        
    def train(self,X,y,epoch=100,batch_size=16,val_X=None,val_y=None,seed=None):
        if self.layer is None:
            self.__compile(X,y,seed)
            
        for _ in range(epoch):
            for _X,_y in prep_batch(batch_size,X,y):
                self.layer[0] = _X
            
                # Calculate forward through the network. (feed forward)
                for i in range(1,len(self.layer)):
                    self.layer[i] = self.optimizer.activation.forward(np.dot(self.layer[i-1],self.weight[i-1]))

                self.weight = self.optimizer.optimize(_y,self.weight,self.layer)
                
            self.training_error.append(self.optimizer.loss.error(y,self.predict(X)))
            if val_X is not None:
                self.validation_error.append(self.optimizer.loss.error(val_y,self.predict(val_X)))

        return self
        
    def predict(self,X):
        x = X.copy()
        for weight in self.weight:
            x = self.optimizer.activation.forward(np.dot(x,weight))
        return x