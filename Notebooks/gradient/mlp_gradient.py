import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.special

from simulated.utils import *

# Multi-layer Perceptron with Gradient Descent
class MultiLayerPerceptron:
    def __init__(self, shape, learning_rate=0.1, max_epochs=1000, print_epochs=True):
        self.shape = shape
        self.num_layers = len(shape)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.print_epochs = print_epochs
        self.weights = []

    def initialize_weights(self):
        self.weights = []
        for i in range(self.num_layers - 1):
            W = np.random.uniform(size=(self.shape[i + 1], self.shape[i]))
            self.weights.append(W)

    def forward_propagation(self, X):
        activations = [X]
        for i in range(self.num_layers - 2):
            activation = scipy.special.expit(np.dot(activations[-1], self.weights[i]))
            activations.append(activation)

        output = scipy.special.expit(np.dot(activations[-1], self.weights[-1].T))
        activations.append(output)
        return activations

    def backward_propagation(self, X, y, activations):
        error = activations[-1] - y
        delta = error * activations[-1] * (1 - activations[-1])

        for i in range(self.num_layers - 2, 0, -1):
            self.weights[i] -= self.learning_rate * np.dot(delta.T, activations[i])
            hidden_error = np.dot(delta, self.weights[i])
            delta = hidden_error * activations[i] * (1 - activations[i])

        self.weights[0] -= self.learning_rate * np.dot(delta.T, X)

    def train(self, X, y):
        self.initialize_weights()
        i = 0
        accuracies = []
        best_scores = [(i, 1)]
        if self.print_epochs:
            print_best_particle(i, best_scores[-1])
        for epoch in range(self.max_epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations)
            i = i + 1
            score = eval_neural_network(self.weights, self.shape, X, y)
            corrects, wrongs, predictions = eval_accuracy(self.weights, self.shape, X, y)
            accuracy = corrects / (corrects + wrongs)
            best_scores.append((i, score))
            if self.print_epochs:
                print_best_particle(i, best_scores[-1])
                print("With accuracy: {accuracy}".format(accuracy=accuracy))
            accuracies.append(accuracy)
        
        return [
            self.weights,
            best_scores,
            accuracies,
            self.max_epochs
        ]
        
    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)