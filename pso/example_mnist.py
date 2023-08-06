import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

import sa
import ann
import grad

def dim_weights(shape):
    dim = 0
    for i in range(len(shape) - 1):
        dim = dim + (shape[i] + 1) * shape[i + 1]
    return dim

def eval_accuracy(weights, shape, X, y):
    corrects, wrongs = 0, 0
    nn = ann.MultiLayerPerceptron(shape, weights=weights)
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

def weights_to_vector(weights):
    w = np.asarray([])
    for i in range(len(weights) + 1):
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
        W = vector[idx_min:idx_max].reshape((r, c))
        weights.append(W)
        idx = idx_max
    return weights

def eval_neural_network(weights, shape, X, y):
    mse = np.asarray([])
    weight = vector_to_weights(weights, shape)
    nn = ann.MultiLayerPerceptron(shape, weights=weight)
    y_pred = nn.run(X)
    mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
    return mse


def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))


# Load MNIST digits from sklearn
num_classes = 10
mnist = sklearn.datasets.load_digits(n_class=num_classes)
X, X_test, y, y_test = sklearn.model_selection.train_test_split(mnist.data, mnist.target)

num_inputs = X.shape[1]

y_true = np.zeros((len(y), num_classes))
for i in range(len(y)):
    y_true[i, y[i]] = 1

y_test_true = np.zeros((len(y_test), num_classes))
for i in range(len(y_test)):
    y_test_true[i, y_test[i]] = 1

# Set up
accuracies = []
shape = (num_inputs, 16, 16, num_classes)  # Increase the number of neurons in the hidden layers

num_dimensions = sum(shape[i] * shape[i + 1] + shape[i + 1] for i in range(len(shape) - 1))
initial_temperature = 100.0
final_temperature = 1
cooling_rate = 0.99

obj_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)

sa_ = sa.SimulatedAnnealing(obj_func, num_dimensions, initial_temperature, final_temperature, cooling_rate)
result = sa_.minimize(shape, X, y)



# Plot
error = [tup[1][0] for tup in result.best_scores]
iters = [tup for tup in range(len(result.best_scores))]
figure = plt.figure()
errorplot = plt.subplot(2, 1, 1)
errorplot.plot(iters, error)
plt.title("Simulated Annealing")
plt.ylabel("Mean squared error")

accuracyplot = plt.subplot(2, 1, 2)
accuracyplot.plot(result.accuracies)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")

plt.show()

# Test...
best_weights = vector_to_weights(sa_.best_state, shape)
best_nn = ann.MultiLayerPerceptron(shape, weights=best_weights)
y_test_pred = np.round(best_nn.run(X_test))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))
