import numpy as np


##############################################################################
class Neural_Network(object):
	def __init__(self, input_size, layers, output_size):
		self.input_layer_size = input_size
		self.hidden_layer_size = len(layers)
		self.output_layer_size = output_size
		self.weights = []
		self.weights.append(np.random.randn(self.input_layer_size, layers[0]))
		for i in range(self.hidden_layer_size-1):
			self.weights.append(np.random.randn(layers[i], layers[i+1]))

		self.weights.append(np.random.randn(self.weights[-1].shape[1], self.output_layer_size))
		# for i in range(len(self.weights)):
		# 	print("weights\n\n", self.weights[i])
		self.Layers = []

	def forward(self, X):
		self.Layers.clear()
		self.Layers.append(X)
		self.z_n = np.dot(X, self.weights[0])
		self.a_n = self.sigmoid(self.z_n)

		self.Layers.append(self.a_n)
		for i in range(1,self.hidden_layer_size+1):
			self.z_n = np.dot(self.a_n, self.weights[i])
			self.a_n = self.sigmoid(self.z_n)
			self.Layers.append(self.a_n)
		yHat = self.a_n
		return yHat


	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def back_prop(self, total_error, X):

		delta = np.array(total_error * self.nonlin(self.Layers[-1], deriv=True))


		deltas = [delta]


		for i in range(self.hidden_layer_size):
			new_error = np.dot(deltas[i], self.weights[-(i+1)].T)
			new_delta = new_error * self.nonlin(self.Layers[-(i+2)], deriv=True)
			deltas.append(new_delta)


		adjustments = []
		deltas.reverse()
		for i in range(len(self.Layers)-1):
			adjustments.append(np.dot(self.Layers[i].T, deltas[i]))
		for i in range(len(adjustments)-1):
			self.weights[i] += adjustments[i]


	def nonlin(self, x, deriv=False):
		if (deriv == True):
			return x * (1 - x)

		return 1 / (1 + np.exp(-x))
##############################################################################
##############################################################################
##############################################################################
X = np.array([[0, 0, 1],
			  [0, 1, 1],
			  [1, 0, 1],
			  [1, 1, 1]])

y = np.array([[0],
			  [1],
			  [1],
			  [0]])

NN = Neural_Network(3, [2, 6, 5], 1)



for i in range(600000):
	#print(NN.weights)

	predictions = NN.forward(X)

	error = y - predictions
	#print("precictions: \n\n",predictions)
	if (i % 10000) == 0:
		print("Error:" + str(np.mean(np.abs(error))))

	NN.back_prop(error, X)
