import matplotlib.pyplot as plt
import numpy as np
import neural_network
from random import randint

def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


X_train, y_train, X_val, y_val = load_data_large()
x = [5, 20, 50, 100, 200]
y = []
y_2 = []
for i in x:
    losses_train, losses_val, train_error, valid_error, train_pred, valid_pred = \
        neural_network.train_and_valid(X_train, y_train, X_val, y_val, 100, i, randint(0, 1), 0.01)
    print(len(losses_train))
    y.append(losses_train[99])
    y_2.append(losses_val[99])
plt.plot(x, y)
plt.plot(x, y_2)
plt.title('Hidden Units')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Cross-Entropy')
plt.legend(["training", "validation"], loc="lower right")
plt.show()


'''
X_train, y_train, X_val, y_val = load_data_large()
x = [0.001]
y = []
y_2 = []
for i in x:
    losses_train, losses_val, train_error, valid_error, train_pred, valid_pred = \
        neural_network.train_and_valid(X_train, y_train, X_val, y_val, 100, 50, randint(0, 1), i)
    plt.plot(losses_train)
    plt.plot(losses_val)
    plt.title(f'Learning Rate = {i}')
    plt.xlabel('Learning Rate Value')
    plt.ylabel('Cross-Entropy')
    plt.legend(["training", "validation"], loc="lower right")
    plt.show()
    plt.clf()
    
'''