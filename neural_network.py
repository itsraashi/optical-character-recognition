import numpy as np

def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An (N_train, M) ndarray containing the training data (N_train examples, M features each)
        y_train: An (N_train,) ndarray contraining the labels
        X_val: An (N_val, M) ndarray containing the validation data (N_val examples, M features each)
        y_val: An (N_val,) ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


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


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    '''
    D, N = p.shape # n = m+1
    def summation(i):
        count = 0
        for m in range(0, N):
            count += (p[i, m] * input[m])

        return count

    veclin = np.vectorize(summation)
    result = veclin(np.arange(D))
    (a, ) = result.shape
    new_result = np.reshape(result, (a, 1))
    '''
    return np.matmul(p, input)

def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    '''
    def sig(i):
        return 1 / (1 + np.exp(-i))

    vecsig = np.vectorize(sig)
    '''

    return 1. / (1. + np.exp(-a))


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    '''
    def smax(i):
        efunc = lambda e: np.exp(e)
        evec = efunc(b)
        bsum = np.sum(evec)
        return np.exp(b[i]) / bsum

    vecy = np.vectorize(smax)
    result = vecy(np.arange(np.size(b)))
    (n,) = result.shape
    new_y = np.reshape(result, (n, 1))
    '''

    exp_b = np.exp(b)
    return exp_b / np.sum(exp_b)


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    '''
    def entropy(i):
        return hot_y[i] * np.log(y_hat[i])

    vecent = np.vectorize(entropy)
    print(hot_y)
    ents = vecent(np.arange(size(hot_y)))
    '''
    return -np.log(y_hat[hot_y])[0]


def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: beta WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    a = linearForward(x, alpha)
    z = sigmoidForward(a)
    new_z = np.insert(z, 0, 1)
    (n,) = new_z.shape
    new_z = np.reshape(new_z, (n, 1))
    b = linearForward(new_z, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y, y_hat)
    return x, a, new_z, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """

    def diff(i):
        if hot_y == i:
            return y_hat[i] - 1
        else:
            return y_hat[i]

    (a, b) = y_hat.shape
    vecgb = np.vectorize(diff)
    result = vecgb(np.arange(a))
    (n,) = result.shape
    new_gb = np.reshape(result, (n, 1))
    return new_gb


def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    n, m = p.shape
    a, _ = prev.shape
    b, _ = grad_curr.shape # may need to transpose

    g_bias = np.matmul(prev, np.transpose(grad_curr))
    p = np.delete(p, 0, 1)
    g_z = np.matmul(np.transpose(p), grad_curr)

    return np.transpose(g_bias), g_z


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """

    def ind(j):
        z_j = curr[j+1][0]
        grad_zj = grad_curr[j][0]
        return grad_zj * z_j * (1-z_j)

    (a, b) = curr.shape
    veca = np.vectorize(ind)
    result = veca(np.arange(a-1))
    (n,) = result.shape
    new_ga = np.reshape(result, (n, 1))
    return new_ga


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    TIP: Make sure you're accounting for the changes due to the bias term
    """
    g_b = softmaxBackward(y, y_hat)
    g_beta, g_z = linearBackward(z, beta, g_b)
    g_a = sigmoidBackward(z, g_z)
    g_alpha, g_x = linearBackward(x, alpha, g_a)
    return g_alpha, g_beta, g_b, g_z, g_a



def SGD(X_train, y_train, X_val, y_val, hidden_units, num_epochs, init_rand, learning_rate):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param hidden_units: Number of hidden units
    :param num_epochs: Number of epochs
    :param init_rand:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    (N_train, M) = X_train.shape
    (N_valid, _) = X_val.shape

    if init_rand:
        alpha = np.random.uniform(-0.1, 0.1, (hidden_units, M+1))
        alpha[:, 0] = 0
        beta = np.random.uniform(-0.1, 0.1, (10, hidden_units + 1))
        beta[:, 0] = 0
    else:
        alpha = np.zeros((hidden_units, M+1))
        beta = np.zeros((10, hidden_units + 1))
    losses_train = []
    losses_val = []

    for e in range(num_epochs):
        for ind in range(N_train):
            (x, y) = (X_train[ind], y_train[ind])

            new_x = np.reshape(x, (M, 1))
            new_x = np.concatenate([np.array([[1]]), new_x])
            (x, a, z, b, y_hat, J) = NNForward(new_x, y, alpha, beta)
            (g_alpha, g_beta, _, _, _) = NNBackward(x, y, alpha, beta, z, y_hat)

            learn_alpha = g_alpha * learning_rate
            learn_beta = g_beta * learning_rate
            alpha = alpha - learn_alpha
            beta = beta - learn_beta

        def vector(i):
            (x, y) = (X_train[i], y_train[i])

            new_x = np.reshape(x, (M, 1))
            new_x = np.concatenate([np.array([[1]]), new_x])
            (x, a, z, b, y_hat, J) = NNForward(new_x, y, alpha, beta)
            return J
        vecJ = np.vectorize(vector)
        result = vecJ(np.arange(N_train))
        losses_train.append(np.sum(result) / N_train)

        def vector(i):
            (x, y) = (X_val[i], y_val[i])

            new_x = np.reshape(x, (M, 1))
            new_x = np.concatenate([np.array([[1]]), new_x])
            (x, a, z, b, y_hat, J) = NNForward(new_x, y, alpha, beta)
            return J

        vecV = np.vectorize(vector)
        resultV = vecV(np.arange(N_valid))
        losses_val.append(np.sum(resultV) / N_valid)

    return alpha, beta, losses_train, losses_val


def prediction(X_train, y_train, X_val, y_val, tr_alpha, tr_beta):
    """
    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data (list)
        - y_hat_valid: predicted labels for validation data (list)
    """
    train_pred = []
    train_error = 0
    (N_train, _) = X_train.shape
    (N_valid, _) = X_val.shape
    for ind in range(N_train):
        (x, y) = (X_train[ind], y_train[ind])

        new_x = np.insert(x, 0, 1)
        (n,) = new_x.shape
        new_x = np.reshape(new_x, (n, 1))
        (x, a, z, b, y_hat, J) = NNForward(new_x, y, tr_alpha, tr_beta)
        l = np.argmax(y_hat)
        train_pred.append(l)
        if l != y:
            train_error += 1

    valid_pred = []
    valid_error = 0
    for ind in range(N_valid):
        (x, y) = (X_val[ind], y_val[ind])

        new_x = np.insert(x, 0, 1)
        (n,) = new_x.shape
        new_x = np.reshape(new_x, (n, 1))
        (x, a, z, b, y_hat, J) = NNForward(new_x, y, tr_alpha, tr_beta)
        l = np.argmax(y_hat)
        valid_pred.append(l)
        if l != y:
            valid_error += 1

    return (train_error/N_train), (valid_error/N_valid), train_pred, valid_pred


### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epochs, num_hidden, init_rand, learning_rate):
    """ 
    Main function to train and validate your neural network implementation.

    :param X_train: Training data input (ndarray with shape (N_train, M))
    :param y_train: Training labels (1D column vector with shape (N_train,))
    :param X_val: Validation data input (ndarray with shape (N_valid, M))
    :param y_val: Validation labels (1D column vector with shape (N_valid,))
    :param num_epochs: Number of epochs to train (i.e. number of loops through the training data).
    :param num_hidden: Number of hidden units.
    :param init_rand: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Float value specifying the learning rate for SGD.

    :return: a tuple of the following six objects, in order:
        - loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        - loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        - err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        - err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        - y_hat_train: A list of integers representing the predicted labels for training data
        - y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    (alpha, beta, losses_train, losses_val) = SGD(X_train, y_train, X_val, y_val, num_hidden, num_epochs, init_rand, learning_rate)
    (train_error, valid_error, train_pred, valid_pred) = prediction(X_train, y_train, X_val, y_val, alpha, beta)

    return losses_train, losses_val, train_error, valid_error, train_pred, valid_pred

