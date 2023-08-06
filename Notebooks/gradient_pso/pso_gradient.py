import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import functools
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

from MLP_w_GD import *
from MLP import *
from PSO import *
from utils import *

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert target labels to one-hot encoding
num_classes = len(np.unique(y))
y_train_onehot = np.eye(num_classes)[y_train]


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
shape = (num_inputs, 128, 64, num_classes)  # Increase the number of neurons in the hidden layers

obj_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)
# # Train using Particle Swarm Optimization
swarm = ParticleSwarm(obj_func, num_dimensions=dim_weights(shape), num_particles=20)

# Train...
i = 0
best_scores = [(i, swarm.best_score)]
print_best_particle(best_scores[-1])
while i < 1000:  # Increase the maximum number of iterations
    swarm._update()
    i = i + 1
    if swarm.best_score < best_scores[-1][1]:
        corrects, wrongs, predictions = eval_accuracy(vector_to_weights(swarm.Gbest, shape), shape, X_test, y_test)
        accuracy = corrects / (corrects + wrongs)
        best_scores.append((i, swarm.best_score))
        print_best_particle(best_scores[-1])
        print("With accuracy: {accuracy}".format(accuracy=accuracy))
        accuracies.append(accuracy)

# Define the MLP shape and hyperparameters
mlp_shape = [X_train.shape[1], 64, num_classes]
learning_rate = 0.01
max_epochs = 1000

# Instantiate and train the MLP
mlp = MultiLayerPerceptronWGradientDescent(shape=mlp_shape, learning_rate=learning_rate, max_epochs=max_epochs)
mlp.train(X_train, y_train_onehot)
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
