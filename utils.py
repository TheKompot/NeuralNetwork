import numpy as np
import matplotlib.pyplot as plt
import os



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
