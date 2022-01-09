import numpy as np

class NeuralNetwork:
    def __init__(self,num_layers:int, num_perceptrons:list,activation:str):
        '''
        Attributes:
        ----------
        num_layers:int
            number of layers
        num_perceptrons:list
            list of ints, i-th element sets number of neurons of i-th layer
        activation:str
            string with name of activation function used (can be sigmoid, relu, tanh)
        '''
        if num_layers != len(num_perceptrons):
            raise AttributeError('Number of layers and perceptron list size does not match')
        self.W_list = []
        self.num_layers = num_layers
        self.num_n = num_perceptrons
        self._create_weights()                      # initialize random weights

        if activation in ['tanh','relu','sigmoid']: # setting activation function and its derivate
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
        ''' derivation of relu activation function'''
        if isinstance(x, np.ndarray):
            return np.array([self.relu_derivation(i) for i in x])
        return max(0,x)

    def calculate_error(self,predicted:np.array,real:np.array)->float:
        '''method for calculating error'''
        return np.mean((real-predicted)**2)

    def fit(self, X_train:np.array, y_train:np.array, X_test:np.array = None,y_test:np.array= None, num_epochs:int=300, alpha:float = 0.03)->tuple or list:
        ''' trains the network
        Attributes:
        ----------
        X_train: 2D np.ndarray
            matrix of inputs
        y_train: 1D np.ndarray
            vector of outputs
        X_test: 2D np.ndarray, optional
            matrix of test inputs, if added the function will return a list of error on the testing set per epoch
        y_test: 1D np.ndarray, optional
            vector of testing outputs
        num_epochs: int, defaul 300
        alpha: float, default 0.03
            learning rate
        
        RETURNS:
        -------
        list
            if testing set is not added then returns list of error history on training set per epoch
        tuple(list,list)
            if testing set added then returns tuple of lists, first with error history on train then on test
        '''
        error_history = []                                                                          #error per epoch for train set
        test_error_history = []                                                                     #error per epoch for test set

        for ep in range(num_epochs):                                                                #for every epoch

            for per in np.random.permutation(X_train.shape[0]):                                     #random permutaion of train data
                x = self._add_bias(X_train[per,:])                                                  #adding bias to one data point
                h=[x]                                                                               #list of inputs for every layer

                net = []                                                                            #list of results without actiation function
                for i in range(self.num_layers-2):                                                  #for every hidden layer
                    net.append(np.dot(h[-1],self.W_list[i]))                                        #calculating linear part of perceptron and adding to list
                    h.append(self.fun(net[-1]))                                                     #applying activation function and adding to list
                net.append(np.dot(h[-1],self.W_list[-1]))                                           #calculate last layer

                y = net[-1][0]                                                                      #getting predicted output as scalar
                d = y_train[per]                                                                    #getting real output
                
                deltas = [0 for _ in range(self.num_layers-1)]                                      #list of derivation of loss function per neuron
                deltas[-1] = np.array([d-y])                                                        #calculating delta for last neuron
                for i in range(self.num_layers-3,-1,-1):                                            #for every layer, backpropagation
                    deltas[i] = np.dot(self.W_list[i+1] , deltas[i+1].T) * self.der_fun(net[i])     #delta for i-th layer

                for i in range(self.num_layers-1):                                                  #updating weights
                    self.W_list[i] += (alpha * np.outer(h[i], deltas[i]))

            predicted = self.predict(X_train)                                                       #calculating error on the end of epoch
            E = self.calculate_error(predicted, y_train)
            error_history.append(E)                                                                 #appending to output list

            if X_test is not None:                                                                  # if test set was added then calculate error on test set
                predicted = self.predict(X_test)
                E = self.calculate_error(predicted, y_test)
                test_error_history.append(E)
        if X_test is not None:                                                                      #if test set added then return error history for train and test
            return error_history,test_error_history
        return error_history                                                                        #if only train set was added then return error history for train

    def predict(self,X:np.array)->np.array:
        '''for given input returns prediction'''
        X = np.array([ self._add_bias(x) for x in X])  # adding bias to every input
        for i in range(self.num_layers-2):             # for every matrice of weights
            X = np.dot(X,self.W_list[i])               # calculate linear part of perceptron
            X = np.array([self.fun(row) for row in X]) # applying activation function
        X = np.dot(X,self.W_list[-1])                  # calculating last layer 
        return X.reshape(X.shape[0])                   # returning an 1D array
    
    def evaluate(self,file_path:str)->float:
        ''' imports data from file_path, calculates prediction and returns mean error'''
        data = np.load(file_path)   # load data
        X = data[:,:2]              # split input and output
        y = data[:,2]

        predicted = self.predict(X) # calculate prediction
        err = self.calculate_error(predicted, y) # calculate mean error
        return err
    def save_weights(self,file_path:str)->None:
        '''exports weight to file_path'''
        with open(file_path) as f:
            for array in self.W_list:
                np.save(f,array)
    def load_weights(self,file_path:str)->None:
        '''imports weights from file
        Raises:
        ------
        AttributeError
            if numpy matrices in file have differente dimension then in the model
        '''
        with open(file_path) as f:
            for i in range(self.num_layers-1):
                w = np.load(f)
                if w.shape == self.W_list[i].shape:
                    self.W_list[i] = w
                else:
                    raise AttributeError('Dimension of weight matrix does not match')