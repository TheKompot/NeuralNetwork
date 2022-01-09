import numpy as np
import matplotlib.pyplot as plt
import os
from neuralNetwork import NeuralNetwork


def press_to_quit(e):
    if e.key in {'q', 'escape'}:
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures


def show_history(history, block=True):
    fig = plt.figure(num='Training history')
    fig.canvas.mpl_connect('key_press_event', press_to_quit)

    plt.title('Loss per epoch')
    plt.plot(history['loss'], '-b', label='training loss')
    try:
        plt.plot(history['val_loss'], '-r', label='validation loss')
    except KeyError:
        pass
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(left=-1); plt.ylim(bottom=-0.01)

    plt.tight_layout()
    plt.show(block=block)


def show_data(X, y, predicted=None, s=30, block=True):
    plt.figure(num='Data', figsize=(9,9)).canvas.mpl_connect('key_press_event', press_to_quit)
    
    if predicted is not None:
        predicted = np.asarray(predicted).flatten()
        plt.subplot(2,1,2)
        plt.title('Predicted')
        plt.scatter(X[:, 0], X[:, 1],
                    c=predicted, cmap='coolwarm',
                    s=10 + s * np.maximum(0, predicted))
        
        plt.subplot(2,1,1)
        plt.title('Original')
    y = np.asarray(predicted).flatten()
    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap='coolwarm',
                s=10 + s * np.maximum(0, y))
    plt.tight_layout()
    
    plt.show(block=block)


def train_test_split(data:np.array, test_ratio:float)->tuple:
    '''
    function for splitting data randomly in to two groups in ratio 1-test_ratio : test_ratio
    '''
    data = data.copy()
    np.random.shuffle(data)
    border = len(data) - round(len(data)*test_ratio)
    train = data[:border]
    test = data[border:]

    return train,test

def k_fold(model:NeuralNetwork, X:np.array, y:np.array, k:int)->tuple:
    ''' k-fold crossvalidation, creates k groups from data, then trains model k times, always selecting a diferente group as test data'''
    X = X.copy()
    y = y.copy()

    data = np.concatenate((X,y.T),axis=1)

    train_err,test_err = [],[]

    np.random.shuffle(data)
    if len(data)%k == 0:
        folds = np.split(data,k)
    else:
        # if cannot equaly distribute data then 
        leftover = data[:len(data)%k] 
        data = data[len(data)%k:] # remove rows so the rest is divisible by k
        folds = np.split(data,k) 
        for i in range(len(leftover)): # add one row to every fold from leftover
            folds[i] = np.concatenate((folds[i],leftover[i].reshape((1,len(leftover[i])))))

    for i in range(k):
        test = folds[i]
        train1, train2 = None, None
        if i > 0: # if not first fold
            train1 = np.concatenate(folds[:i]) # select folds before test
        if i < k-1: # if not last fold
            train2 = np.concatenate(folds[i+1:]) # select folds after test

        
        if train1 is None: # if test is first fold
            train = train2
        elif train2 is None: # if test is last fold
            train = train1
        else:
            train = np.concatenate((train1,train2))

        X_train, y_train = train[:,:len(train[0])-1], train[:,-1]
        X_test, y_test = test[:,:len(test[0])-1], test[:,-1]

        train_temp,test_temp =  model.fit(X_train,y_train,X_test,y_test,300,alpha=0.03)
        train_err.extend(train_temp)
        test_err.extend(test_temp)
    return train_err,test_err
