import numpy as np
from sklearn.metrics import accuracy_score

from utils import *
from MLP import *

class GeneticAlgorithmResult:
    def __init__(self, best_individual, best_score, num_iterations):
        self.best_individual = best_individual
        self.best_score = best_score
        self.num_iterations = num_iterations

class GeneticAlgorithm:
    def __init__(self, obj_func, num_dimensions, population_size, X, Y, learning_rate, shape, mutation_rate=0.01, crossover_rate=0.8):
        self.obj_func = obj_func
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.X = X
        self.Y = Y
        self.shape = shape
        self.learning_rate = learning_rate

        self.population = np.random.uniform(size=(self.population_size, self.num_dimensions))
        self.fitness_values = self.obj_func(self.population)

        self.best_individual = self.population[self.fitness_values.argmin()]
        self.best_score = self.fitness_values.min()

    def _update(self):
        old_pop = self.population
        # Crossover
        self.crossover()

        # Mutation
        self.mutate()

        # Evaluate fitness of offspring
        offspring_scores = self.obj_func(self.population)
        ind = np.argmin(offspring_scores)

        if offspring_scores.argmin() < self.best_score:
            # Update best individual and best score
            self.best_individual = self.population[ind]
            self.best_score = offspring_scores[ind]
        else:
            self.population = old_pop

    def crossover(self):
        for _ in range(self.num_dimensions):
            parent1 = self.best_individual
            parent2_idx = np.random.randint(0, self.population_size)
            val = np.random.randint(0, self.num_dimensions)
            
            x = np.random.random(1)
            if (x <= self.crossover_rate):
                parent1[val] = self.population[parent2_idx][val]
                self.population[parent2_idx][val] = parent1[val]

    def mutate(self):
        x = np.random.random(1)
        if (x < self.mutation_rate):
            ind1 = np.random.randint(0, self.num_dimensions)
            mutation = np.random.uniform(1)
            self.best_individual[ind1] = mutation

    def minimize(self, max_iter):
        for i in range(max_iter):
            self._update()

        return GeneticAlgorithmResult(
            best_individual=self.best_individual,
            best_score=self.best_score,
            num_iterations=max_iter
        )
    