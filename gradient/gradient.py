import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.special

# Multi-layer Perceptron with Gradient Descent
class MultiLayerPerceptron:
    def __init__(self, shape, learning_rate=0.1, max_epochs=1000):
        self.shape = shape
        self.num_layers = len(shape)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
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

        for epoch in range(self.max_epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations)

    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

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

# Define the MLP shape and hyperparameters
mlp_shape = [X_train.shape[1], 64, num_classes]
learning_rate = 0.01
max_epochs = 1000

# Instantiate and train the MLP
mlp = MultiLayerPerceptron(shape=mlp_shape, learning_rate=learning_rate, max_epochs=max_epochs)
mlp.train(X_train, y_train_onehot)
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
