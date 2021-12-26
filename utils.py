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
    plt.plot(history.history['loss'], '-b', label='training loss')
    try:
        plt.plot(history.history['val_loss'], '-r', label='validation loss')
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

def k_fold(model:NeuralNetwork, X:np.array, y:np.array, k:int)->None:
    ''' k-fold crossvalidation, creates k groups from data, then trains model k times, always selecting a diferente group as test data'''
    X = X.copy()
    y = y.copy()

    data = np.concatenate((X,y.T),axis=1)

    np.random.shuffle(data)
    folds = np.split(data,k)
    for i in range(k):
        print('fold ',i)
        test = folds[i]
        train1, train2 = None, None
        if i > 0:
            train1 = np.concatenate(folds[:i])
        if i < k-1:
            train2 = np.concatenate(folds[i+1:])

        
        if train1 is None:
            train = train2
        elif train2 is None:
            train = train1
        else:
            train = np.concatenate((train1,train2))

        X_train, y_train = train[:,:len(train[0])-1], train[:,-1]
        X_test, y_test = test[:,:len(test[0])-1], test[:,-1]

        model.fit(X_train,y_train,X_test,y_test)
        print('train')
        print(X_train,y_train)
        print('test')
        print(X_test,y_test)

X = np.array([[i,i] for i in range(10)])
y = np.array([[i for i in range(10)]])

model = NeuralNetwork()
k_fold(model, X, y, 2)