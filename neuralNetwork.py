import numpy as np

class NeuralNetwork:

    def __init__(self, num_layers:int, num_perceptrons:list, alpha:float = 0.1):
        self.W_list = []
        self.num_layers = num_layers
        self.num_n = num_perceptrons
        self.apha = alpha
        self._create_weights()
        
    def _create_weights(self):
        '''random initialization of weights WITHOUT BIAS'''
        for i in range(self.num_layers-1):
            if i == 0: # for weights between input and first hidden layer -> add 1 plus row (bias)
                w = np.random.rand(self.num_n[i]+1,self.num_n[i+1])
            else:
                w = np.random.rand(self.num_n[i],self.num_n[i+1])
            self.W_list.append(w)

    def _add_bias(self,x:np.array)->np.array:
        '''add 1 at the end of vector x'''
        return np.concatenate((x,[1]))

    def sigmoid(self, x:float)->float:
        '''Activation function - logistical sigmoid'''
        return 1/(1 + np.exp(-x))
    
    def relu(self,x:float)->float:
        '''Activation function - Rectified Linear Unit'''
        return max([0,x])

    def fit(self, X_train, y_train, X_test, y_test, num_epochs=10):
        pass

    def predict(self,X:np.array)->np.array:
        '''for given input return prediction WITHOUT ACTIVATAION FUNCTION'''
        X = np.array([ self._add_bias(x) for x in X])
        for i in range(self.num_layers-1):
            X = np.dot(X,self.W_list[i])
        return X

model = NeuralNetwork(4,[2,5,3,1])
data = np.array([[i,i] for i in range(10)])
print(model.predict(data))
