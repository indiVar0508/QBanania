import numpy as np
from collections import deque
import random

class NeuralNet():

	def __init__(self, layers = [2, 2, 1], learningRate = 0.09, activationFunc = 'sigmoid', Gaussian = True):
		self.layers = layers
		self.learningRate = learningRate
		self.Gaussian = Gaussian
		if Gaussian:
			self.biasses = [np.random.randn(1, l) for l in self.layers[1:]] # n * 3
			self.weights = [np.random.randn(i, o) for i, o in zip(self.layers[:-1], self.layers[1:])] # n X 2 * 2 X 3 = n * 3
		else:
			self.biasses = [np.random.randn(1, l) for l in self.layers[1:]] # n * 3
			self.weights = [np.random.randn(i, o) for i, o in zip(self.layers[:-1], self.layers[1:])] # n X 2 * 2 X 3 = n * 3
		self.activationFunc = activationFunc

	def sigmoid(self, z):
		return (1 / (1. + np.exp(-z)))

	def sigmoidPrime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	def RelU(self, z):
		z[z<0] = 0
		return z
	def ReLUPrime(self, z):
		z[z!=0] = 1
		return z

	def feedForward(self, X):
		X = np.array(X).reshape(1, -1)
		assert self.layers[0] == X.shape[1]
		activations = [X]
		for w, b in zip(self.weights, self.biasses):
			if self.activationFunc == 'sigmoid': X = self.sigmoid(np.matmul(X, w) + b)
			elif self.activationFunc == 'relu': X = self.RelU(np.matmul(X, w) + b)
			activations.append(X)
		return activations

	def softmax(self, z):
		exps = [np.exp(e) for e in z]
		sum_of_exps = np.sum(exps)
		softmax = [e / sum_of_exps for e in exps]
		return softmax

	def predict(self, X, show = 'probability'):
		if show == 'round': return np.round(self.softmax(self.feedForward(X)[-1]))
		elif show == 'softmax': 
			softmax = []
			preds = self.feedForward(X)[-1]
			for pred in preds: softmax.append(self.softmax(pred))
			return np.argmax(softmax, 1)
		elif show == 'probability': 
			softmax = []
			preds = self.feedForward(X)[-1]
			for pred in preds: softmax.append(self.softmax(pred))
			return softmax



	def backPropogate(self, x, y):
		bigDW = [np.zeros(w.shape) for w in self.weights]
		bigDB = [np.zeros(b.shape) for b in self.biasses]
		activations = self.feedForward(x)
		delta = activations[-1] - y
		for layer in range(2, len(self.layers) + 1):
			# print(bigDW[-layer + 1].shape, np.dot(activations[-layer].T, delta).shape)
			# print(delta.shape, self.weights[-layer + 1].shape)
			bigDW[-layer + 1] = (1 / len(x)) * np.dot(activations[-layer].T, delta)
			bigDB[-layer + 1] = (1 / len(x)) * np.sum(delta)
			if self.activationFunc == 'sigmoid': delta = np.dot(delta, self.weights[-layer + 1].T) * self.sigmoidPrime(activations[-layer])
			elif self.activationFunc == 'relu': delta = np.dot(delta, self.weights[-layer + 1].T) * self.ReLUPrime(activations[-layer])

		for w, dw in zip(self.weights, bigDW): w -= self.learningRate * dw
		for b, db in zip(self.biasses, bigDB): b -= self.learningRate * db


	def fit(self, x, y, epochs = 100):
		for _ in range(epochs): self.backPropogate(x, y)


class mutableBrain(NeuralNet):

	def __init__(self, layers = [2, 2, 1], learningRate = 0.09, activationFunc = 'relu', Gaussian = False, weights = None, biasses = None):
		super().__init__(layers, learningRate, activationFunc, Gaussian)
		if weights != None: self.weights = weights
		if biasses != None: self.biasses = biasses

	def mutate(self, mutationRate = 0.01):
		for i in range(len(self.weights)):
			for row in range(len(self.weights[i])):
				for col in range(len(self.weights[i][row])):
					if np.random.random() < mutationRate: 
						if self.Gaussian: self.weights[i][row][col] = np.random.randn()
						else: self.weights[i][row][col] = np.random.random()
		for i in range(len(self.biasses)):
			for row in range(len(self.biasses[i])):
					if np.random.random() < mutationRate: 
						if self.Gaussian: self.biasses[i][row] = np.random.randn()
						else: self.biasses[i][row] = np.random.random()

	def giveMeChildBrainBY(parent):
		return mutableBrain(layers = parent.layers, learningRate = parent.learningRate, activationFunc = parent.activationFunc, Gaussian = parent.Gaussian,\
					 weights = parent.weights, biasses = parent.biasses)

class QBrain:
	def __init__(self, layers = [2, 2, 1], learningRate = 0.09, activationFunc = 'relu', Gaussian = False, gamma = 0.9, epsilon = 1.0, epsilonDecay = 0.995, min_epsilon = 0.1):
		self.brain = NeuralNet(layers = layers, learningRate =learningRate, activationFunc = activationFunc, Gaussian = Gaussian)
		self.epsilon = epsilon
		self.gamma = 0.9
		self.epsilonDecay = epsilonDecay
		self.min_epsilon = min_epsilon
		self.memory = deque(maxlen = 20_000)
		self.stateSize = layers[0]
		self.actionSize = layers[-1]


	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, X, show = 'softmax'):
		if np.random.random() < self.epsilon: return np.random.randint(self.actionSize)
		return self.brain.predict(X = X, show = 'softmax')

	def replay(self, batch_size):

		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			# print(action)
			if not done:
				target = (reward + self.gamma * np.amax(self.brain.predict(X = next_state, show = 'probability')))
			target_f = self.brain.predict(X = state, show = 'probability')
			target_f[0][action] = target
			self.brain.backPropogate(state, target_f)
		if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilonDecay			





if __name__ == '__main__':
	nn = mutableBrain(layers = [2, 2, 2], learningRate = 0.25, activationFunc = 'relu', Gaussian = False)
	datasetX = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	datasetY = [[1 - (x[0] ^ x[1]), x[0] ^ x[1]] for x in datasetX]
	print(datasetY)
	print(nn.predict(datasetX, show = 'probability'))
	print(nn.predict(datasetX, show = 'softmax'))
	nn.fit(datasetX, datasetY, epochs = 1000)
	print(nn.predict(datasetX, show = 'probability'))
	print(nn.predict(datasetX, show = 'softmax'))

