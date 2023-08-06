from MLP_w_GD import *
from MLP import *
import sklearn.metrics
import sklearn.datasets

def dim_weights(shape):
    dim = 0
    for i in range(len(shape) - 1):
        dim = dim + (shape[i] + 1) * shape[i + 1]
    return dim

def weights_to_vector(weights):
    w = np.asarray([])
    for i in range(len(weights)):
        v = weights[i].flatten()
        w = np.append(w, v)
    return w


def vector_to_weights(vector, shape):
    weights = []
    idx = 0
    for i in range(len(shape) - 1):
        r = shape[i + 1]
        c = shape[i]
        idx_min = idx
        idx_max = idx + r * c
        W = np.array(vector[idx_min:idx_max]).reshape(r, c)
        weights.append(W)
        idx = idx_max
    return weights


def eval_accuracy(weights, shape, X, y):
    corrects, wrongs = 0, 0
    nn = MultiLayerPerceptron(shape)
    nn.weights = weights
    predictions = []
    for i in range(len(X)):
        out_vector = nn.run(X[i])
        y_pred = np.argmax(out_vector)
        predictions.append(y_pred)
        if y_pred == y[i]:
            corrects += 1
        else:
            wrongs += 1
    return corrects, wrongs, predictions


def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    for w in weights:
        weight = vector_to_weights(w, shape)
        nn = MultiLayerPerceptron(shape)
        nn.weights = weight
        y_pred = nn.run(X)
        mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
    return mse


def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))

