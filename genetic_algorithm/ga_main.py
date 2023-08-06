import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

import ga
from MLP_w_GD import *
from utils import *

def print_best_individual(best_individual):
    print("New best individual found at iteration #{i} with mean squared error: {score}".format(i=best_individual[0], score=best_individual[1]))


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

obj_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)

population_size = 128
mutation_rate = 0.8
crossover_rate = 0.8
num_generations = 1000

genetic_algo = ga.GeneticAlgorithm(obj_func, dim_weights(shape), population_size, X, y, 0.01, shape,
                                  mutation_rate=mutation_rate, crossover_rate=crossover_rate)

# Train...
i = 0
best_scores = [(i, 1)]
while i < num_generations:
    print(i)
    genetic_algo._update()
    i = i + 1
    if genetic_algo.best_score < best_scores[-1][1]:
        corrects, wrongs, predictions = eval_accuracy(vector_to_weights(genetic_algo.best_individual, shape), shape, X_test, y_test)
        accuracy = corrects / (corrects + wrongs)
        best_scores.append((i, genetic_algo.best_score))
        print_best_individual(best_scores[-1])
        print("With accuracy: {accuracy}".format(accuracy=accuracy))
        accuracies.append(accuracy)

# Plot
error = [tup[0] for tup in best_scores]
iters = [tup[1] for tup in best_scores]
figure = plt.figure()
errorplot = plt.subplot(2, 1, 1)
errorplot.plot(error, iters)
plt.title("Genetic Algorithm")
plt.ylabel("Mean squared error")

accuracyplot = plt.subplot(2, 1, 2)
accuracyplot.plot(accuracies)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")

plt.show()

# Test...
best_weights = vector_to_weights(genetic_algo.best_individual, shape)
best_nn = MultiLayerPerceptronWGradientDescent(shape)
best_nn.weights = best_weights
y_test_pred = np.round(best_nn.run(X_test))
print(sklearn.metrics.classification_report(y_test_true, y_test_pred.T))
