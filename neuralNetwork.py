import numpy as np

class NeuralNetwork:

    def __init__(self, num_layers:int, num_perceptrons:list, alpha:float = 0.1):
        if num_layers != len(num_perceptrons):
            raise AttributeError('Number of layers and perceptron list size does not match')
        self.W_list = []
        self.num_layers = num_layers
        self.num_n = num_perceptrons
        self.alpha = alpha
        self._create_weights()
        
    def _create_weights(self):
        '''random initialization of weights'''
        for i in range(self.num_layers-1):
            if i == 0: # for weights between input and first hidden layer -> add 1 plus row (bias)
                w = np.random.rand(self.num_n[i]+1,self.num_n[i+1])
            else:
                w = np.random.rand(self.num_n[i],self.num_n[i+1])
            self.W_list.append(w)

    def _add_bias(self,x:np.array)->np.array:
        '''add 1 at the end of vector x'''
        return np.concatenate((x,[1]))

    def sigmoid(self, x:float or np.array)->float or np.array:
        '''Activation function - logistical sigmoid'''
        return 1/(1 + np.exp(-x))

    def sigmoid_derivation(self,x:float or np.array)->float or np.array:
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def relu(self,x:float or np.array)->float or np.array:
        '''Activation function - Rectified Linear Unit'''
        if isinstance(x, np.ndarray):
            return np.array([self.relu(i) for i in x])
        return max([0,x])

    def relu_derivation(self,x:float or np.array)->int or np.array:
        ''' derivation of relu activation function
            raises:
                AttributeError: if x is 0'''
        if isinstance(x, np.ndarray):
            return np.array([self.relu_derivation(i) for i in x])
        if x > 0:
            return x
        if x < 0:
            return 0
        raise AttributeError('for 0 derivation of relu is not defined')

    def calculate_error(self,predicted:float,real:float)->float:
        '''method for calculating error'''
        return ((predicted-real)**2)/2

    def calculate_accuracy(self, inputs, targets):
        pass

    def fit(self, X_train, y_train, num_epochs=10):
        
        error_history = []
        accuracy_history = []

        for ep in range(num_epochs):
            E = 0

            for per in np.random.permutation(X_train.shape[0]):
                x = self._add_bias(X_train[per,:]) 

                net = [x] # outputs of the network without activation function
                deltas = [] # computed errors of neurons
                inputs = [x]

                for i in range(self.num_layers-1):              # calculating output from NN
                    x = np.dot(x,self.W_list[i])                # multiplying with weights
                    net.append(x)                         # storing output for error calculation
                    x = np.array([self.sigmoid(row) for row in x]) # applying activation function
                    inputs.append(x)

                y = y_train[per]
                E += self.calculate_error(x[0],y)

                for i in range(self.num_layers-1,-1,-1): # computing derivation of error for every layer
                    if i == self.num_layers-1: #if output layer
                        d = (x-y) * net[-1]
                    elif i == 0:                        # if input layer
                        d = np.dot(self.W_list[i],deltas[-1]) 
                    else:
                        d = np.dot(self.W_list[i],deltas[-1]) * self.sigmoid_derivation(net[-1])

                    deltas.append(d)
                    net.pop()
                deltas.reverse()
                for i in range(self.num_layers-1): # updating weights
                    self.W_list[i] += (self.alpha * np.dot( deltas[i] ,inputs[i]))
            error_history.append(E/X_train.shape[0])
            #acc = self.compute_accuracy(inputs, targets)
            #accuracy_history.append(acc)
            if (ep+1) % 10 == 0: print(f'Epoch {ep+1}, E = {error_history[-1]}, accuracy = {0}')

        return error_history,accuracy_history

    def predict(self,X:np.array)->np.array:
        '''for given input return prediction'''
        X = np.array([ self._add_bias(x) for x in X])
        for i in range(self.num_layers-1):
            X = np.dot(X,self.W_list[i])
            X = np.array([self.sigmoid(row) for row in X]) # applying activation function
        return X

