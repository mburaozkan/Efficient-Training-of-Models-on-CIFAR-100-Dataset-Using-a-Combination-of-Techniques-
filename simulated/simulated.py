import functools
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ann import MultiLayerPerceptron
from utils import *
import scipy.special


class SAResult(object):
    def __init__(self, best_solution, best_score, num_iterations):
        self.best_solution = best_solution
        self.best_score = best_score
        self.num_iterations = num_iterations

class SimulatedAnnealing(object):
    def __init__(self, obj_func, num_dimensions, initial_solution, initial_temperature, final_temperature, cooling_rate):
        self.obj_func = obj_func
        self.num_dimensions = num_dimensions
        self.initial_solution = initial_solution
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        
        
        self.current_solution = self.initial_solution.copy()
        self.best_solution = self.current_solution.copy()
        self.current_score = self.obj_func(self.current_solution)
        self.best_score = self.current_score
        self.temperature = self.initial_temperature

    def _acceptance_probability(self, new_score):
        if new_score < self.current_score:
            return 1.0
        else:
            return np.exp((self.current_score - new_score) / self.temperature)

    def _update(self):
        if self.temperature > self.final_temperature:
            new_solution = self.current_solution + np.random.normal(scale=1.0, size=self.num_dimensions)
            new_score = self.obj_func(new_solution)

            acceptance_prob = self._acceptance_probability(new_score)
            if acceptance_prob > np.random.uniform():
                self.current_solution = new_solution
                self.current_score = new_score

            if new_score < self.best_score:
                self.best_solution = new_solution
                self.best_score = new_score

            self.temperature *= self.cooling_rate

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
num_inputs = X.shape[1]
num_classes = len(np.unique(y))
y_train_onehot = np.eye(num_classes)[y_train]

# Set up
mlp_shape = (num_inputs, 128, 64, num_classes)  # Increase the number of neurons in the hidden layers

# Define the MLP shape and hyperparameters
learning_rate = 0.01
max_epochs = 1000

# Define the objective function
obj_func = functools.partial(eval_neural_network, shape=mlp_shape, X=X_train, y=y_train_onehot)

# Define the parameters for simulated annealing
num_dimensions = sum(mlp_shape[i] * mlp_shape[i + 1] + mlp_shape[i + 1] for i in range(len(mlp_shape) - 1))
initial_solution = np.random.randn(num_dimensions)
initial_temperature = 1.0
final_temperature = 0.01
cooling_rate = 0.99

# Instantiate and train the MLP using simulated annealing
sa = SimulatedAnnealing(obj_func, num_dimensions, initial_solution, initial_temperature, final_temperature,
                                   cooling_rate)

for i in range(1000):
    sa._update()

# Create the MLP using the best solution
best_weights = vector_to_weights(sa.best_solution, mlp_shape)
best_nn = MultiLayerPerceptron(mlp_shape, weights=best_weights)
y_test_pred = np.round(best_nn.run(X_test))

y_pred = []

for i, preds in enumerate(y_test_pred.T):
    y_pred.append(pred_arr_to_int(preds))

print(sklearn.metrics.classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)