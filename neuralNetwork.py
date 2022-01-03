import numpy as np

class NeuralNetwork:

    def __init__(self, num_layers:int, num_perceptrons:list, alpha:float = 0.1):
        self.W_list = []
        self.num_layers = num_layers
        self.num_n = num_perceptrons
        self.apha = alpha
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
        if isinstance(x, np.ndarray):
            return np.array([self.sigmoid(i) for i in x])
        return 1/(1 + np.exp(-x))
    
    def relu(self,x:float or np.array)->float or np.array:
        '''Activation function - Rectified Linear Unit'''
        if isinstance(x, np.ndarray):
            return np.array([self.relu(i) for i in x])
        return max([0,x])

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
        return (predicted-real)**2

    def fit(self, X_train, y_train, num_epochs=10):
        
        error_history = []
        accuracy_history = []

        for ep in range(num_epochs):
            E = 0

            for i in np.random.permutation(X_train.shapep[0]):
                x = self._add_bias(X_train[i,:]) 

                for i in range(self.num_layers-1):
                    X = np.dot(X,self.W_list[i])
                    X = np.array([self.relu(row) for row in X]) # applying activation function

                y = y_train[i,:]
                E += self.calculate_error(predicted,y)

                #for j in range(self.num_layers-1):

    def predict(self,X:np.array)->np.array:
        '''for given input return prediction'''
        X = np.array([ self._add_bias(x) for x in X])
        for i in range(self.num_layers-1):
            X = np.dot(X,self.W_list[i])
            X = np.array([self.relu(row) for row in X]) # applying activation function
        return X

model = NeuralNetwork(4,[2,5,3,1])
data = np.array([[i,i] for i in range(10)])
print(model.predict(data))