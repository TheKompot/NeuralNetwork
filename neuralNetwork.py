import numpy as np

class NeuralNetwork:

    def __init__(self, num_layers:int, num_perceptrons:list, alpha = 0.1):
        self.W_list = []
        self.num_layers = num_layers
        self.num_n = num_perceptrons
        self.apha = alpha
        self._create_weights()
        
    def _create_weights(self):
        '''random initialization of weights'''
        for i in range(self.num_layers-1):
            w = np.random.rand(self.num_n[i],self.num_n[i+1])
            self.W_list.append(w)

    def fit(self, X_train, y_train, X_test, y_test, num_epochs=10):
        pass

    def predict(X):
        pass
model = NeuralNetwork(4,[2,5,3,1])