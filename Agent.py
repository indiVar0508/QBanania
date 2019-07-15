import pygame
from Brain import *
import numpy as np

class GeneralPlayer():

	def __init__(self, gameDisplay, gameWidth = 600, gameHeight = 400, visionLimit = 50):
		self.gameWidth = gameWidth
		self.gameHeight = gameHeight
		self.gameDisplay = gameDisplay
		self.imgItr = 1
		self.x, self.y = 50, self.gameHeight // 2
		self.width, self.height = 30, 35
		self.step = 5
		self.characterDefault = pygame.image.load(r'Resources\Character\StandBy\{}.png'.format(self.imgItr))
		self.left = self.right = self.up = self.down = False
		self.visionLimit = visionLimit
		self.leftVision = self.rightVision = self.upVision = self.downVision = (255, 255 ,255)
		self.leftVisionBlock = (self.x + self.width // 2 - self.visionLimit, self.y + self.height // 2)
		self.rightVisionBlock = (self.x + self.width // 2 + self.visionLimit, self.y + self.height // 2)
		self.upVisionBlock = (self.x + self.width // 2, self.y + self.height // 2 - self.visionLimit)
		self.downVisionBlock = (self.x + self.width // 2, self.y + self.height // 2 + self.visionLimit)
		self.leftDistance = self.rightDistace = self.upDistance = self.downDistance = self.visionLimit
		self.image = 1
		self.u = self.d = self.l = self.r = 1

	def takeStep(self, dir_):
		reward = 0
		if dir_ == 'Left':
			self.x -= self.step
			if self.x < 0: 
				self.x = 0
				self.left = False
				reward = -0.5
				# self.leftVisionBlock = (self.leftVisionBlock[0] + self.step, self.leftVisionBlock[1]) # i was lazy enough to change it every where to list :P
		elif dir_ == 'Right':
			self.x += self.step
			if self.x +self.width > self.gameWidth: 
				self.x = self.gameWidth - self.width
				self.right = False
				self.fitness -= 0.1
				reward = -0.5
				# self.rightVisionBlock = (self.rightVisionBlock[0] - self.step, self.rightVisionBlock[1])
		elif dir_ == 'Up':
			self.y -= self.step
			if self.y < 0: 
				self.y = 0
				self.up = False
				reward = -0.5
				# self.upVisionBlock = (self.upVisionBlock[0], self.upVisionBlock[1] + self.step)
		else:
			self.y += self.step
			if self.y + self.height > self.gameHeight: 
				self.y = self.gameHeight - self.height
				self.down = False
				reward = -0.5
				# self.downVisionBlock = (self.downVisionBlock[0], self.downVisionBlock[1] - self.step)
		self.handleVision()
		return reward

	
	def decideMovement(self, cords):
		if self.up: 
			self.d = self.l = self.r = 1
			yield self.showMovement('Up', cords, self.u)
		elif self.down: 
			self.u = self.l = self.r = 1
			yield self.showMovement('Down', cords, self.d)
		elif self.left: 
			self.u = self.d = self.r = 1
			yield self.showMovement('Left', cords, self.l)
		elif self.right: 
			self.u = self.l = self.d = 1
			yield self.showMovement('Right', cords, self.r)



	def hurdleContact(self, cords):
		self.handleHurdleVision(cords)
		condn, cords = self.contactingHurdle(cords)
		reward = 0
		if condn:
			reward -= 0.25
			if self.left: 
				self.x = cords[0] + cords[2]
				self.left = False
				self.upVisionBlock = (self.upVisionBlock[0] + self.step, self.upVisionBlock[1])
				self.downVisionBlock = (self.downVisionBlock[0] + self.step, self.downVisionBlock[1])
			elif self.right: 
				self.x = cords[0] - self.width
				self.right = False
				self.upVisionBlock = (self.upVisionBlock[0] - self.step, self.upVisionBlock[1])
				self.downVisionBlock = (self.downVisionBlock[0] - self.step, self.downVisionBlock[1])
			elif self.up: 
				self.y = cords[1] + cords[3]
				self.up = False
				self.leftVisionBlock = (self.leftVisionBlock[0], self.leftVisionBlock[1] + self.step)
				self.rightVisionBlock = (self.rightVisionBlock[0], self.rightVisionBlock[1] + self.step)
			elif self.down: 
				self.y = cords[1] - self.height	
				self.down = False
				self.leftVisionBlock = (self.leftVisionBlock[0], self.leftVisionBlock[1] - self.step)
				self.rightVisionBlock = (self.rightVisionBlock[0], self.rightVisionBlock[1] - self.step)
			return (True, reward)
		return (False, reward)

	def contactingHurdle(self, cordinates):
		for cords in cordinates:
			if (cords[0] < self.x < cords[0] + cords[2] or cords[0] < self.x + self.width < cords[0] + cords[2] or cords[0] < self.x + self.width // 2 < cords[0] + cords[2]) and\
			(cords[1] < self.y < cords[1] + cords[3] or cords[1] < self.y + self.height < cords[1] + cords[3] or  cords[1] < self.y + self.height // 2 < cords[1] + cords[3]): return True, cords
		return False, None


	def handleHurdleVision(self, cordinates):
		# self.leftVision = self.rightVision = self.upVision = self.downVision = (255, 255 ,255)
		for idx, cords in enumerate(cordinates):
			if self.x + self.width // 2 >= cords[0] + cords[2]  and cords[1] <= self.y + self.height // 2 <= cords[1] + cords[3] and self.x + self.width // 2 - (cords[0] + cords[2]) <= self.visionLimit:
				self.leftDistance = self.x + self.width // 2 - (cords[0] + cords[2])
				self.leftVisionBlock = (cords[0] + cords[2], self.y + self.height // 2)
				self.leftVision = (255, 0 , 0)
			elif self.x + self.width // 2 <= cords[0] and cords[1] <= self.y + self.height // 2 <= cords[1] + cords[3] and cords[0] - (self.x + self.width // 2) <= self.visionLimit:
				self.rightDistace =  cords[0] - (self.x + self.width // 2)
				self.rightVisionBlock = (cords[0], self.y + self.height // 2)
				self.rightVision = (255, 0 , 0)
			elif self.y + self.height // 2 >= cords[1] + cords[3] and cords[0] <= self.x + self.width // 2 <= cords[0] + cords[2] and self.y + self.height // 2 - (cords[1] + cords[3]) <= self.visionLimit:
				self.upDistance =  self.y + self.height // 2 - (cords[1] + cords[3])
				self.upVisionBlock = (self.x + self.width // 2, cords[1] + cords[3])
				self.upVision = (255, 0 , 0)
			elif self.y + self.height // 2 <= cords[1] and cords[0] <= self.x + self.width // 2 <= cords[0] + cords[2] and cords[1] - (self.y + self.height // 2) <= self.visionLimit:
				self.downDistance =  cords[1] - (self.y + self.height // 2)
				self.downVisionBlock = (self.x + self.width // 2, cords[1])
				self.downVision = (255, 0 , 0)



	def showVision(self):
		pygame.draw.line(self.gameDisplay, self.leftVision, (self.x + self.width // 2, self.y + self.height // 2), self.leftVisionBlock, 2)
		pygame.draw.line(self.gameDisplay, self.rightVision, (self.x + self.width // 2, self.y + self.height // 2), self.rightVisionBlock, 2)
		pygame.draw.line(self.gameDisplay, self.upVision, (self.x + self.width // 2, self.y + self.height // 2), self.upVisionBlock, 2)
		pygame.draw.line(self.gameDisplay, self.downVision, (self.x + self.width // 2, self.y + self.height // 2), self.downVisionBlock, 2)

	def handleVision(self):
		# self.leftVision = self.rightVision = self.upVision = self.downVision = (255, 255 ,255)
		if (self.x + self.width // 2) - self.visionLimit < 0: 
			self.leftVisionBlock = (0, self.y + self.height // 2)
			self.leftVision = (255, 0, 0)
			self.leftDistance = self.x + self.width // 2
		else: 
			self.leftVisionBlock = (self.x + self.width // 2 - self.visionLimit, self.y + self.height // 2)
			self.leftDistance = self.visionLimit

		if self.x + self.width // 2 + self.visionLimit > self.gameWidth: 
			self.rightVisionBlock = (self.gameWidth, self.y + self.height // 2)
			self.rightVision = (255, 0, 0)
			self.rightDistace =  self.gameWidth - (self.x + self.width // 2)
		else: 
			self.rightDistace = self.visionLimit
			self.rightVisionBlock = (self.x + self.width // 2 + self.visionLimit, self.y + self.height // 2)

		if self.y + self.height // 2 - self.visionLimit < 0: 
			self.upDistance =  self.y + self.height // 2
			self.upVisionBlock = (self.x + self.width // 2, 0)
			self.upVision = (255, 0, 0)
		else: 
			self.upDistance = self.visionLimit
			self.upVisionBlock = (self.x + self.width // 2, self.y + self.height // 2 - self.visionLimit)

		if self.y + self.height // 2 + self.visionLimit > self.gameHeight: 
			self.downDistance =   self.gameHeight - (self.y + self.height // 2)
			self.downVisionBlock = (self.x + self.width // 2, self.gameHeight)
			self.downVision = (255, 0, 0)
		else: 
			self.downDistance = self.visionLimit
			self.downVisionBlock = (self.x + self.width // 2, self.y + self.height // 2 + self.visionLimit)


	def showPlayerStandBy(self):
		self.gameDisplay.blit(self.characterDefault, (self.x, self.y))

	def showMovement(self, dir, cords, image):
		movDir = 'Resources\\Character\\Movements\\' + dir + 'Movement\\'
		# self.fitness += 0.000005
		reward = self.takeStep(dir)
		condn, r = self.hurdleContact(cords)
		reward += r
		if not (self.left or self.up or self.right or self.down) or condn: return (None, reward)
		img = pygame.image.load(movDir + '{}.png'.format(image))
		image = 1 +  (self.image + 1) % 4
		return (img, reward)

	def gotFood(self, foodLoc, foodSize):
		if (foodLoc[0] <= self.x <= foodLoc[0] + foodSize or foodLoc[0] <= self.x + self.width <= foodLoc[0] + foodSize) \
		and (foodLoc[1] <= self.y <= foodLoc[1] + foodSize or foodLoc[1] <= self.y + self.height <= foodLoc[1] + foodSize):
			return True, 10
		return False, 0


class MutablePlayer(GeneralPlayer):

	def __init__(self, gameDisplay, gameWidth = 600, gameHeight = 400, layers = [4, 2, 4], learningRate = 0.09, activationFunc = 'relu', Gaussian = False, weights = None, biasses = None):
		super().__init__(gameDisplay, gameWidth = 600, gameHeight = 400)
		self.Brain = mutableBrain(layers = layers, learningRate = learningRate, activationFunc = activationFunc, Gaussian = Gaussian, weights = weights, biasses = biasses)
		self.alive = True
		self.steps = 5000

	def isAlive(self):
		# print(self.fitness)
		if self.fitness < -2 or self.steps < 0: 
			self.alive = False


	def getFitness(self, foodCords, hurdles):
		self.fitness += (1 / np.sqrt((self.x - foodCords[0]) ** 2 + (self.y - foodCords[1]) ** 2))
		i = len(hurdles) - 1
		while i > 0 and hurdles[i][0] > self.x: i -= 1
		self.fitness += i * 5
		return self.fitness

	def biCrossOver(parentOne, parentTwo):
		child = MutablePlayer(gameDisplay = parentOne.gameDisplay, gameWidth = parentOne.gameWidth, gameHeight = parentOne.gameHeight, layers = parentOne.Brain.layers, \
			activationFunc = parentOne.Brain.activationFunc, Gaussian = parentTwo.Brain.Gaussian,\
			 weights = parentOne.Brain.weights, biasses = parentTwo.Brain.biasses)
		for idx, _ in enumerate(child.Brain.weights):
			for row, __ in enumerate(child.Brain.weights[idx]):
				for col, ___ in enumerate(child.Brain.weights[idx][row]):
					if np.random.random() < 0.5: child.Brain.weights[idx][row][col] = np.copy(parentOne.Brain.weights[idx][row][col])
					else: child.Brain.weights[idx][row][col] = np.copy(parentTwo.Brain.weights[idx][row][col])
		for idx, _ in enumerate(child.Brain.biasses):
			for row, __ in enumerate(child.Brain.biasses[idx]):
				for col, ___ in enumerate(child.Brain.biasses[idx][row]):
					if np.random.random() < 0.5: child.Brain.biasses[idx][row][col] = np.copy(parentOne.Brain.biasses[idx][row][col])
					else: child.Brain.biasses[idx][row][col] = np.copy(parentTwo.Brain.biasses[idx][row][col])
		# quit()

		return child

	def uniCrossOver(parentOne):
		child = MutablePlayer(gameDisplay = parentOne.gameDisplay, gameWidth = parentOne.gameWidth, gameHeight = parentOne.gameHeight, layers = parentOne.Brain.layers, \
			activationFunc = parentOne.Brain.activationFunc, Gaussian = parentOne.Brain.Gaussian,\
			 weights = parentOne.Brain.weights, biasses = parentOne.Brain.biasses)
		for idx, _ in enumerate(child.Brain.weights):
			child.Brain.weights[idx] = np.copy(parentOne.Brain.weights[idx])
			child.Brain.biasses[idx] = np.copy(parentOne.Brain.biasses[idx])
		return child



	def think(self, foodCords):
		foodDistance = np.sqrt((self.x - foodCords[0]) ** 2 + (self.y - foodCords[1]) ** 2) / self.gameWidth
		state = np.array([self.leftDistance / self.visionLimit, self.rightDistace / self.visionLimit, self.upDistance / self.visionLimit, self.downDistance / self.visionLimit,\
						self.left, self.right, self.up, self.down, foodDistance])
		action = self.Brain.predict(X = state, show = 'softmax')

		if action == 0: 
			self.up = True
			self.left = self.right = self.down = False
		elif action == 1:
			self.down = True
			self.left = self.right = self.up = False
		elif action == 2:
			self.left = True
			self.right = self.up = self.down = False
		elif action == 3:
			self.right = True
			self.left = self.up = self.down = False


class QPlayer(GeneralPlayer):

	def  __init__(self, gameDisplay, gameWidth = 600, gameHeight = 400, layers = [2, 2, 1], learningRate = 0.09, activationFunc = 'relu', Gaussian = False,\
	gamma = 0.9, epsilon = 1.0, epsilonDecay = 0.995, min_epsilon = 0.1):
		super().__init__(gameDisplay = gameDisplay, gameWidth = gameWidth, gameHeight = gameHeight)
		self.brain = QBrain(layers = layers, learningRate = learningRate, activationFunc = activationFunc, Gaussian = Gaussian, gamma = gamma, epsilon = epsilon, epsilonDecay = epsilonDecay, min_epsilon = min_epsilon)

	def getState(self, foodCords):
		foodDistance = np.sqrt((self.x - foodCords[0]) ** 2 + (self.y - foodCords[1]) ** 2) / self.gameWidth
		return np.array([self.leftDistance / self.visionLimit, self.rightDistace / self.visionLimit, self.upDistance / self.visionLimit, self.downDistance / self.visionLimit,\
						self.left, self.right, self.up, self.down, foodDistance, self.x / self.gameWidth, self.y / self.gameHeight])

	def move(self, foodCords):
		state = self.getState(foodCords)
		action = int(self.brain.act(state))
		if action == 0: 
			self.up = True
			self.upVision = (0, 255, 0)
			self.leftVision = self.rightVision = self.downVision = (0, 0, 255)
			self.left = self.right = self.down = False
		elif action == 1:
			self.down = True
			self.downVision = (0, 255, 0)
			self.leftVision = self.rightVision = self.upVision = (0, 0, 255)
			self.left = self.right = self.up = False
		elif action == 2:
			self.left = True
			self.leftVision = (0, 255, 0)
			self.upVision = self.rightVision = self.downVision = (0, 0, 255)
			self.right = self.up = self.down = False
		elif action == 3:
			self.right = True
			self.rightVision = (0, 255, 0)
			self.leftVision = self.up = self.downVision = (0, 0, 255)
			self.left = self.up = self.down = False

		return state, action


	def remember(self, state, action, reward, next_state, done):
		self.brain.remember(state, action, reward, next_state, done)

	def learn(self, batch_size):
		self.brain.replay(batch_size)

	def reset(self):
		self.x, self.y = 50, self.gameHeight // 2
		self.left = self.right = self.up = self.down = False
		self.leftVision = self.rightVision = self.upVision = self.downVision = (255, 255 ,255)
		self.leftVisionBlock = (self.x + self.width // 2 - self.visionLimit, self.y + self.height // 2)
		self.rightVisionBlock = (self.x + self.width // 2 + self.visionLimit, self.y + self.height // 2)
		self.upVisionBlock = (self.x + self.width // 2, self.y + self.height // 2 - self.visionLimit)
		self.downVisionBlock = (self.x + self.width // 2, self.y + self.height // 2 + self.visionLimit)
		self.leftDistance = self.rightDistace = self.upDistance = self.downDistance = self.visionLimit
		self.image = 1
		self.u = self.d = self.l = self.r = 1