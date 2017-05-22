import numpy as np


##############################################################################
class Neural_Network(object):
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

	def train_with_Sim(self, X, y, epooch=100):
		pygame.init()
		screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption('~~wow~~')
		simulation = Sim()

		for i in range(100000):
			predictions = self.forward(X)
			error = y - predictions
			self.back_prop(error, X)
			if (i % 10000) == 0:
				print("Training Accuracy: ", (100*(1-np.round(np.mean(np.abs(error)),4))), " %")
				simulation.process_events(screen)
				simulation.display_frame(screen, self.all_Layers, self.weights)
		pygame.quit()

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

###########################################################################

#############################  SIMULATION  ################################
###########################################################################
import pygame
FPS = 30
SCREEN_SIZE = (1000, 800)
GREEN = (0, 205, 102)
RED = (220, 20, 60)
BLUE = (0,245,255)
PURPLE = (224,102,255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GOLD = (255,215,0)
GRAY = (110,110,110)

class Sim(object):
	def __init__(self):
		self.run_sim = True
	def process_events(self, screen):
		if self.run_sim is True:
			for event in pygame.event.get():
				if event.type is pygame.KEYDOWN:
					if event.key is pygame.K_x:
						pygame.display.quit()
						pygame.quit()
		pygame.display.flip()

	def display_frame(self, screen, sizes, weights):
		screen.fill(GRAY)
		self.draw_neurons(screen, sizes, weights)
	def draw_neurons(self, screen, sizes, weights):
		initial_Y = 350
		buffer = 100
		x_buffer = 100
		coordinates = []
		column = []
		for x in range(len(sizes)):
			if sizes[x] % 2 is 0:
				offset = -(sizes[x])//2 * buffer
			else:
				offset = -(sizes[x]-1) // 2 * buffer - buffer//2
			for y in range(sizes[x]):

				x_pos = (x + 1) * x_buffer
				y_pos = (y+1)* buffer + initial_Y + offset
				if x is 0:
					color = BLUE
				elif x is len(sizes)-1:
				 	color = GOLD
				else:
					color = PURPLE
				pygame.draw.circle(screen, color, (x_pos, y_pos), buffer//3, buffer//3)
				column.append((x_pos, y_pos))
			coordinates.append(list(column))
			column.clear()

		for i in range(len(sizes)-1):
			current_weights = weights[i]
			for j in range(len(coordinates[i])):
				for k in range(len(coordinates[i+1])):
					x_final = coordinates[i+1][k][0]
					y_final = coordinates[i+1][k][1]
					thickness = int(current_weights[j][k]/4)
					if thickness is 0:
						thickness = 1
					if thickness < 0:
						thickness = abs(thickness)
						color = RED
					else:
						color = GREEN
					#print(thickness)
					pygame.draw.line(screen, color, coordinates[i][j], (x_final, y_final), thickness)
######################################################

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import scale


def main():
	#################################################
	#################################################
	#################  IRIS DATA  ###################
	#################################################
	#################################################
	iris_data = pd.read_csv("Iris.csv")
	iris_data = shuffle(iris_data)
	labelencoder = LabelEncoder()
	for col in iris_data.columns:
		iris_data[col] = labelencoder.fit_transform(iris_data[col])
	labels = np.array(iris_data["Species"])
	labelencoder.fit(labels)
	labels = labelencoder.transform(labels)
	labels = np_utils.to_categorical(labels)
	y_train = labels[:150]
	y_test = labels[100:150]
	X = iris_data.iloc[:, 1:5]
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X_train = np.array(X[:150])
	X_test = np.array(X[100:150])
	#################################################
	#################################################
	#################################################

	#################################################
	#################  BREAST CANCER DATA  ##########
	#################################################
	Breast_Cancer = pd.read_csv("breastcancer.csv")
	print(Breast_Cancer)
	Breast_Cancer.drop(Breast_Cancer.columns[[0, 5]], axis=1, inplace=True)
	X_Breast = np.array(Breast_Cancer.iloc[:, [0, 1, 2, 3, 5, 6, 7]])
	X_Breast = scale(X_Breast)
	print(X_Breast)
	y_Breast = np.array(Breast_Cancer.iloc[:, [8]])

	y_Breast = y_Breast.ravel()
	labelencoder.fit(y_Breast)
	y_Breast = labelencoder.transform(y_Breast)
	y_Breast = np_utils.to_categorical(y_Breast)



	X_Breast_Train = X_Breast[:550,:]
	X_Breast_Test = X_Breast[550:, :]
	y_Breast_Train = y_Breast[:550, :]
	y_Breast_Test = y_Breast[550:, :]

	#################################################

	#################################################
	#################################################


	# hidden_layer_list = [int(x) for x in input("Enter Hidden Layers: ").split()]
	#hidden_layer_list = [7]
	#NN = Neural_Network(7, hidden_layer_list, 2)
	#NN.train(X_Breast_Train, y_Breast_Train, 10000)
	#NN.predict(X_Breast_Test, y_Breast_Test)
	#NN.train_with_Sim(X_Breast_Test, y_Breast_Test, 10000000000)
	#NN.predict(X_Breast_Test, y_Breast_Test)


	hidden_layer_list = [4,10,12,2,1,4,5]
	NN = Neural_Network(4, hidden_layer_list, 3)

	NN.train_with_Sim(X_train, y_train, 10000000000000000000)

main()
