import numpy as np
from sklearn.metrics import accuracy_score

from ga.utils import *
from ga.MLP import *

class GeneticAlgorithmResult:
    def __init__(self, best_individual, best_scores, accuracies, num_iterations):
        self.best_individual = best_individual
        self.best_scores = best_scores,
        self.accuracies = accuracies,
        self.num_iterations = num_iterations
            

class GeneticAlgorithm:
    def __init__(self, obj_func, num_dimensions, population_size, X, Y, shape, mutation_rate=0.01, crossover_rate=0.8, print_epochs=False):
        self.obj_func = obj_func
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.X = X
        self.Y = Y
        self.shape = shape
        self.print_epochs = print_epochs

        self.population = np.random.uniform(size=(self.population_size, self.num_dimensions))
        self.fitness_values = self.obj_func(self.population)

        self.best_individual = self.population[self.fitness_values.argmin()]
        self.best_score = self.fitness_values.min()

    def _update(self):
        old_pop = self.population.copy()  # Make a copy of the population
        self.crossover()
        self.mutate()
        offspring_scores = self.obj_func(self.population)

        if np.min(offspring_scores) < self.best_score:
            # Update best individual and best score
            ind = np.argmin(offspring_scores)
            self.best_individual = self.population[ind]
            self.best_score = offspring_scores[ind]
        else:
            self.population = old_pop  # Restore the population from the copy

    def crossover(self):
        for i in range(self.population_size):
            if np.random.random() <= self.crossover_rate:
                parent1 = self.best_individual
                parent2_idx = np.random.randint(0, self.population_size)
                crossover_point = np.random.randint(0, self.num_dimensions)
                
                self.population[i, crossover_point:] = parent1[crossover_point:]
                self.population[i, :crossover_point] = self.population[parent2_idx, :crossover_point]

    def mutate(self):
        for i in range(self.population_size):
            for j in range(self.num_dimensions):
                if np.random.random() < self.mutation_rate:
                    self.population[i, j] = np.random.uniform()

    def minimize(self, max_iter, X_test, y_test, shape):
        i = 0
        accuracies = []
        best_scores = [(i, 1)]
        while i < max_iter:
            self._update()
            i = i + 1
            if self.best_score < best_scores[-1][1]:
                corrects, wrongs, predictions = eval_accuracy(vector_to_weights(self.best_individual, shape), shape, X_test, y_test)
                accuracy = corrects / (corrects + wrongs)
                best_scores.append((i, self.best_score))
                if self.print_epochs:
                    print_best_individual(best_scores[-1])
                    print("With accuracy: {accuracy}".format(accuracy=accuracy))
                accuracies.append(accuracy)

        return GeneticAlgorithmResult(
            best_individual=self.best_individual,
            best_scores = best_scores,
            accuracies = accuracies,
            num_iterations=max_iter    
        )
