import numpy as np
import scipy.special


class MultiLayerPerceptron:
    def __init__(self, shape, weights=None):
        self.shape = shape
        self.num_layers = len(shape)
        self.weights = weights

    def run(self, data):
        layer = data.T
        for i in range(self.num_layers - 1):
            prev_layer = layer
            o = np.dot(self.weights[i], prev_layer)
            # sigmoid
            layer = scipy.special.expit(o)
        return layer
