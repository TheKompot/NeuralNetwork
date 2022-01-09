from utils import k_fold
from neuralNetwork import NeuralNetwork

import numpy as np

test_file_path = "ADD FILE PATH"

### SAVED WEIGHTS
model = NeuralNetwork(num_layers=4, num_perceptrons=[2,48,11,1], activation='sigmoid')
model.load_weights('weights.npy')
print('Loss of loaded model:',model.evaluate('mlp_train.txt'))
print('Loss on new data',model.evaluate(test_file_path))

### TRAINING 
data = np.loadtxt('mlp_train.txt')
X = data[:,:2]
y = data[:,2]

model = NeuralNetwork(num_layers=4, num_perceptrons=[2,48,11,1], activation='sigmoid')
# Trenovanie cez k-fold crossvalidation na mojom pocitaci trvalo 10 min
k_fold(model=model, X=X, y=y.reshape((1,y.shape[0])), k=8)
# bez k-fold crossvalidation iba nieco malo cez minutu
#model.fit(X,y)

print('Loss on train data',model.evaluate('mlp_train.txt'))
print('Loss on new data',model.evaluate(test_file_path))