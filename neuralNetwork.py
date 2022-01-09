import numpy as np

class NeuralNetwork:

    def __init__(self, num_perceptrons:list,activation:str):
        num_layers = 3
        if num_layers != len(num_perceptrons):
            raise AttributeError('Number of layers and perceptron list size does not match')
        self.W_list = []
        self.num_layers = num_layers
        self.num_n = num_perceptrons
        self._create_weights()

        if activation in ['tanh','relu','sigmoid']:
            if activation == 'sigmoid':
                self.fun = self.sigmoid
                self.der_fun = self.sigmoid_derivation
            elif activation == 'relu':
                self.fun = self.relu
                self.der_fun = self.relu_derivation
            elif activation == 'tanh':
                self.fun = self.tanh
                self.der_fun = self.tanh_derivation
        else:
            raise AttributeError('Unknown activation function')
        
    def _create_weights(self):
        '''random initialization of weights'''
        for i in range(self.num_layers-1):
            if i == 0: # for weights between input and first hidden layer -> add 1 plus row (bias)
                w = np.random.rand(self.num_n[i]+1,self.num_n[i+1])
            else:
                w = np.random.rand(self.num_n[i],self.num_n[i+1])
            w = np.array([i*0.6-0.3 for i in w]) # changing the range to [-0.3,0.3]
            self.W_list.append(w)

    def _add_bias(self,x:np.array)->np.array:
        '''add 1 at the end of vector x'''
        return np.concatenate((x,[1]))

    def tanh(self,x:np.array)->np.array:
        '''Activation function - hyperbolic tangens'''
        return np.tanh(x)

    def tanh_derivation(self,x:float or np.array)->float or np.array:
        if isinstance(x, np.ndarray):
            return np.array([self.tanh_derivation(i) for i in x])
        return 1-x**2

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
        return max(0,x)

    def calculate_error(self,predicted:np.array,real:np.array)->float:
        '''method for calculating error'''
        return np.mean((real-predicted)**2)

    def fit(self, X_train, y_train, X_test,y_test, num_epochs=10, alpha:float = 0.1):
        
        error_history = []
        test_error_history = []

        for ep in range(num_epochs):

            for per in np.random.permutation(X_train.shape[0]):
                x = self._add_bias(X_train[per,:]) 

                net = [0,0]

                net[0] = np.dot(x,self.W_list[0])
                h  = self.fun(net[0])

                net[1] = np.dot(h,self.W_list[1])
                y = net[1][0] # from array to numbers

                d = y_train[per]
                
                deltas =  [0,0]
                deltas[1] = np.array([d-y])
                deltas[0] = self.W_list[1].T * deltas[1] * self.der_fun(net[0])   

                self.W_list[0] += (alpha * np.outer(x,deltas[0]))
                self.W_list[1] += (alpha * np.outer(h,deltas[1]))

            predicted = self.predict(X_train)
            E = self.calculate_error(predicted, y_train)
            error_history.append(E)

            predicted = self.predict(X_test)
            E = self.calculate_error(predicted, y_test)
            test_error_history.append(E)

        return error_history,test_error_history

    def predict(self,X:np.array)->np.array:
        '''for given input return prediction'''
        X = np.array([ self._add_bias(x) for x in X])
        for i in range(self.num_layers-2):
            X = np.dot(X,self.W_list[i])
            X = np.array([self.fun(row) for row in X]) # applying activation function
        X = np.dot(X,self.W_list[-1])
        return X.reshape(X.shape[0])
