import numpy as np
import sklearn.metrics

from ann import *

def vector_to_weights(vector, shape):
    weights = []
    idx = 0
    for i in range(len(shape) - 1):
        r = shape[i + 1]
        c = shape[i]
        idx_min = idx
        idx_max = idx + r * c
        W = vector[idx_min:idx_max].reshape(r, c)
        weights.append(W)
        idx = idx_max
    return weights


def pred_arr_to_int(y):
    for index, _ in enumerate(y):
        if _ == 1:
            return index
    return -1


def eval_accuracy(weights, shape, X, y):
    corrects, wrongs = 0, 0
    nn = MultiLayerPerceptron(shape, weights=weights)
    predictions = []
    for i in range(len(X)):
        out_vector = nn.run(X[i])
        y_pred = np.argmax(out_vector)
        predictions.append(y_pred)
        
        if y_pred ==  pred_arr_to_int(y[i]):
            corrects += 1
        else:
            wrongs += 1
    return corrects, wrongs, predictions


def eval_neural_network(weights, shape, X, y):
    weights = vector_to_weights(weights, shape)
    nn = MultiLayerPerceptron(shape, weights=weights)
    y_pred = nn.run(X)

    mse = sklearn.metrics.mean_squared_error(y, y_pred.T)
    return mse


def print_best_particle(i, mse):
    print("New best weights found at iteration #{i} with mean squared error: {score}".format(i=i, score=mse[1]))
