import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# A Counter is a container that keeps track of how many times equivalent values are added.



LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200
score_requirement = 70
initial_games = 10000

def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for i in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0: # for first iter?
				game_memory.append([prev_observation, action])
			prev_observation = observation
			score += reward
			if done:
				break
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				# ONE HOT ENCODING !!!
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				training_data.append([data[0], output])
		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	print('Average accepted Score: ', np.mean(accepted_scores))
	print('Median accepted Score:  ', np.median(accepted_scores))
	print(Counter(accepted_scores))

	return np.vstack(training_data)

def neural_network_model(input_size):
	# Input Layer
	network = input_data(shape = [None, input_size, 1], name = 'input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	# Output Layer
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size=len(X[0]))
	model.fit({'input':X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openaistuff')
	return model


import numpy as np



#training_data = initial_population()
from sklearn.preprocessing import scale

#print(training_data)
training_data = np.load('saved.npy')
# print("TRAINING DATA: ", training_data)
X =  training_data[:, 0]
X = np.vstack(X)
X = scale(X)
X = X[:, 2:3]
y = np.vstack(np.ravel(training_data[:, 1]))
# import Neural_Network
# NN = Neural_Network.Neural_Network(1,[1, 200, 100],2)
# NN.train(X, y, 1000)
model = train_model(training_data)


#model.save('iloveit.model')
#model.load('iloveit.model')
#
scores = []
choices = []

for each_game in range(100):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(500):
		env.render()
		if len(prev_obs) is 0:
			action = random.randrange(0,2)
		else:
			#print(prev_obs)
			#print(prev_obs.reshape(-1, len(prev_obs), 1))
			#print("prediction",  model.predict(prev_obs))
			#print( np.argmax(model.predict(prev_obs)))
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)))

		choices.append(action) # to check wtf its thinking
		new_obs, reward, done, info = env.step(action)
		prev_obs = new_obs
		game_memory.append([new_obs, action])
		score += reward
		if done:
			#pass
			break

	scores.append(score)
#print('Avg Scores: ', sum(score)/len(scores))
# percent of each choice
#print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices)), choices.count(0)/len(choices))


		# we can keep saving and learning and getting better and better and better


