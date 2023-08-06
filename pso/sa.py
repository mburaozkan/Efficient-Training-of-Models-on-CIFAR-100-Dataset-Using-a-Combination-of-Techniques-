import numpy as np

import ann

def eval_accuracy(weights, shape, X, y):
    corrects, wrongs = 0, 0
    nn = ann.MultiLayerPerceptron(shape, weights=weights)
    predictions = []
    for i in range(len(X)):
        out_vector = nn.run(X[i])
        # print(out_vector)
        y_pred = np.argmax(out_vector)
        # print(y_pred)
        # print(y[i])
        predictions.append(y_pred)
        if y_pred == y[i]:
            corrects += 1
        else:
            wrongs += 1
    return corrects, wrongs, predictions

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

def print_best_particle(best_particle):
    print("New best particle found at iteration #{i} with mean squared error: {score}".format(i=best_particle[0], score=best_particle[1]))


class SAResult(object):
    def __init__(self, best_particle, best_scores, accuracies, num_iterations):
        self.best_particle = best_particle
        self.best_scores = best_scores
        self.accuracies = accuracies
        self.num_iterations = num_iterations


class SimulatedAnnealing(object):
    def __init__(self, obj_func, num_dimensions, initial_temperature, final_temperature, cooling_rate):
        self.obj_func = obj_func
        self.num_dimensions = num_dimensions
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate

        self.current_state = np.random.rand(num_dimensions)
        self.best_state = self.current_state.copy()
        self.current_score = self.obj_func(self.current_state)
        self.best_score = self.current_score
        self.num_iterations = 0

        self.best_scores = []
        self.accuracies = []

    def _transition(self, temperature, shape, X, y):
        candidate_state = self.current_state + np.random.normal(size=self.num_dimensions)
        candidate_score = self.obj_func(candidate_state)

        if candidate_score < self.current_score and np.random.rand() < np.exp(-(candidate_score - self.current_score) / temperature):
            self.current_state = candidate_state
            self.current_score = candidate_score

            if candidate_score < self.best_score:
                self.best_state = candidate_state
                self.best_score = candidate_score

                corrects, wrongs, predictions = eval_accuracy(vector_to_weights(self.best_state, shape), shape, X, y)
                accuracy = corrects / (corrects + wrongs)
                self.best_scores.append((self.num_iterations, self.best_score))
                print_best_particle(self.best_scores[-1])
                print("With accuracy: {accuracy}".format(accuracy=accuracy))
                self.accuracies.append(accuracy)
        
    def minimize(self, shape, X, y):
        temperature = self.initial_temperature

        while temperature > self.final_temperature:
            print(temperature)
            for _ in range(100):
                self._transition(temperature, shape, X, y)
                self.num_iterations += 1
            temperature *= self.cooling_rate

        
        return SAResult(
            best_particle=self.best_state,
            best_scores=self.best_scores,
            accuracies=self.accuracies,
            num_iterations=self.num_iterations
        )
