import numpy as np


##############################################################################
class Neural_Network(object):
	def __init__(self, input_size, layers, output_size):
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

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def back_prop(self, total_error, X):
		# Layers [input layer, hidden layer1, 2, 3]
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

	def sigmoid_prime(self, x):
		return  x * (1-x)



##############################################################################
#############################################################################
###########################################################################
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.utils import np_utils



def main():
	iris_data = pd.read_csv("Iris.csv")
	iris_data = shuffle(iris_data)
	labelencoder = LabelEncoder()

	for col in iris_data.columns:
		iris_data[col] = labelencoder.fit_transform(iris_data[col])
	#print(iris_data.head())

	labels = np.array(iris_data["Species"])

	labelencoder.fit(labels)
	labels = labelencoder.transform(labels)
	labels = np_utils.to_categorical(labels)
	y_train = labels[:100]


	y_test = labels[100:150]

	X = iris_data.iloc[:, 1:5]
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	X_train = np.array(X[:100])
	X_test = np.array(X[100:150])

	hidden_layer_list = [10, 8, 5, 3]
	NN = Neural_Network(4, hidden_layer_list, 3)
	for i in range(500000):
		predictions = NN.forward(X_train)
		error = y_train - predictions
		NN.back_prop(error, X_train)
		if (i % 10000) == 0:
			print("Error: ", np.mean(np.abs(error)))

main()
