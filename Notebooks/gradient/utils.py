import numpy as np
import sklearn.metrics

from gradient.mlp import *

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
    nn = MultiLayerPerceptron(shape, weights=weights)
    y_pred = nn.run(X)
    mse = sklearn.metrics.mean_squared_error(y, y_pred.T)
    return mse


def print_best_particle(i, mse):
    print("New best weights found at iteration #{i} with mean squared error: {score}".format(i=i, score=mse[1]))
