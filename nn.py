"""
Feed forward perceptron
"""
import numpy as np


class NeuralNetwork(object):

    def __init__(self, input_size, layers, output_size):
        self.all_Layers = [input_size] + layers + [output_size]
        self.input_layer_size = input_size
        self.hidden_layer_size = len(layers)
        self.output_layer_size = output_size
        self.weights = []
        self.weights.append(np.random.randn(self.input_layer_size, layers[0]))
        for i in range(self.hidden_layer_size - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))

        self.weights.append(np.random.randn(self.weights[-1].shape[1], self.output_layer_size))
        self.Layers = []

    def forward(self, X):
        self.Layers.clear()
        self.Layers.append(X)
        self.z_n = np.dot(X, self.weights[0])
        self.a_n = self.sigmoid(self.z_n)

        self.Layers.append(self.a_n)
        for i in range(1, self.hidden_layer_size + 1):
            self.z_n = np.dot(self.a_n, self.weights[i])
            self.a_n = self.sigmoid(self.z_n)
            self.Layers.append(self.a_n)
        yHat = self.a_n
        return yHat

    def train(self, X, y, epooch=1000):
        for i in range(epooch):
            predictions = self.forward(X)
            error = y - predictions
            self.back_prop(error, X)
            if (i % 1000) == 0:
                print("Training Accuracy: ", (100 * (1 - np.round(np.mean(np.abs(error)), 4))), " %")



    def predict(self, X):
        predictions = self.forward(X)
        return predictions
    #error = y - predictions
    #print("Prediction Accuracy: ", (100 * (1 - np.round(np.mean(np.abs(error)), 4))), " %")

    def back_prop(self, total_error, X):
        # Layers: [input_layer, hidden layer1, 2, 3] * realize output_layer is not included in Layers
        # Delta: error w/ respect to change in neuron value

        deltas = [np.array(total_error * self.sigmoid_prime(self.Layers[-1]))]
        for i in range(self.hidden_layer_size):
            new_error = np.dot(deltas[i], self.weights[-(i + 1)].T)
            new_delta = new_error * self.sigmoid_prime(self.Layers[-(i + 2)])
            deltas.append(new_delta)
        adjustments = []
        deltas.reverse()
        for i in range(len(deltas)):
            adjustments.append(np.dot(self.Layers[i].T, deltas[i]))
        for i in range(len(adjustments)):
            self.weights[i] += adjustments[i]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, x):
        return x * (1 - x)

