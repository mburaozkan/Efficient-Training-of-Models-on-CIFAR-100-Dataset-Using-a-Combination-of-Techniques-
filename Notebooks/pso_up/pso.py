import numpy as np
from pso_up.utils import *


class PSOResult(object):
    def __init__(self, best_particle, best_scores, accuracies, num_iterations):
        self.best_particle = best_particle
        self.best_scores = best_scores
        self.accuracies = accuracies
        self.num_iterations = num_iterations

class ParticleSwarm(object):
    def __init__(self, obj_func, num_dimensions, num_particles, inertia=0.72984, c1=2.05, c2=2.05, print_epochs=False):
        self.obj_func = obj_func
        self.num_dimensions = num_dimensions
        self.print_epochs = print_epochs

        self.num_particles = num_particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

        self.X = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        self.V = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        self.Pbest = self.X.copy()
        self.fitness_value = self.obj_func(self.X)
        self.Gbest = self.Pbest[self.fitness_value.argmin()]
        self.best_score = self.fitness_value.min()

        self.Vmax = np.abs(np.amax(self.X) - np.amin(self.X))

    def _update(self):
        # Velocities update
        r1 = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        r2 = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        self.V = self.inertia * (self.V
                                 + self.c1 * r1 * (self.Pbest - self.X)
                                 + self.c2 * r2 * (self.Gbest - self.X))

        # Apply velocity limits
        self.V = np.clip(self.V, -self.Vmax, self.Vmax)

        # Positions update
        self.X = self.X + self.V

        # Best scores
        scores = self.obj_func(self.X)

        better_scores_idx = scores < self.fitness_value
        self.Pbest[better_scores_idx] = self.X[better_scores_idx]
        self.fitness_value[better_scores_idx] = scores[better_scores_idx]

        self.Gbest = self.Pbest[self.fitness_value.argmin()]
        self.best_score = self.fitness_value.min()

    def minimize(self, max_iter, X_test, y_test, shape):
        i = 0
        accuracies = []
        best_scores = [(i, self.best_score)]
        if self.print_epochs:    
            print_best_particle(best_scores[-1])
        for _ in range(max_iter):
            self._update()
            i = i + 1
            if self.best_score < best_scores[-1][1]:
                corrects, wrongs, predictions = eval_accuracy(vector_to_weights(self.Gbest, shape), shape, X_test, y_test)
                accuracy = corrects / (corrects + wrongs)
                best_scores.append((i, self.best_score))
                if self.print_epochs:
                    print_best_particle(best_scores[-1])
                    print("With accuracy: {accuracy}".format(accuracy=accuracy))
                accuracies.append(accuracy)
        return PSOResult(
            best_particle = self.Gbest,
            best_scores = best_scores,
            accuracies = accuracies,
            num_iterations = max_iter
        )
