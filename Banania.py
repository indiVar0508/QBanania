import pygame
import numpy as np
from Agent import QPlayer

pygame.display.set_caption('Banania')
pygame.init()

class Environment():

	def __init__(self, gameWidth = 600, gameHeight = 400):
		self.gameHeight = gameHeight
		self.gameWidth = gameWidth
		self.gameDisplay = pygame.display.set_mode((gameWidth, gameHeight))
		self.backGroundColor = (150, 0, 0)
		self.hurdleCords = []
		self.hurdleColor = (0, 255, 0)
		relDist, increment = 0.2, 0.25
		for _ in range(3):
			self.hurdleCords.append((int(relDist* self.gameWidth), int(0.1 * self.gameHeight), 20, 320))
			self.hurdleCords.append((int((relDist + 0.1) * self.gameWidth), 0, 20, 175))
			self.hurdleCords.append((int((relDist + 0.1) * self.gameWidth), 225, 20, 175))
			relDist += increment
		self.foodLoc = (int(0.9 * self.gameWidth), self.gameHeight // 2 - 25)
		self.foodSize = 30
		self.food = pygame.image.load(r'Resources\Food\banana.png')
		self.player = QPlayer(gameDisplay = self.gameDisplay, gameWidth = self.gameWidth, gameHeight = self.gameHeight, layers = [11, 24, 4], learningRate = 0.45, \
						activationFunc = 'relu', Gaussian = True, gamma = 0.9, epsilon = 1.0, epsilonDecay = 0.995, min_epsilon = 0.1)



	def makeObjMsg(self, msg, fontDefination, color = (0, 0, 0)):
		msgObj = fontDefination.render(msg, True, color)
		return msgObj, msgObj.get_rect()

	def message(self, msg, color = (0, 0, 0), fontType = 'freesansbold.ttf', fontSize = 15, xpos = 10, ypos = 10):
		fontDefination = pygame.font.Font(fontType, fontSize)
		msgSurface, msgRectangle = self.makeObjMsg(msg, fontDefination, color)
		msgRectangle = (xpos, ypos)
		self.gameDisplay.blit(msgSurface, msgRectangle)

	def makeGrids(self):
		for x in range(0,self.gameWidth, 5): pygame.draw.line(self.gameDisplay, (10, 10, 10), (x, 0), (x, self.gameHeight))
		for y in range(0, self.gameHeight, 5): pygame.draw.line(self.gameDisplay, (10, 10, 10), (0, y), (self.gameWidth, y))

	def makeHurdles(self):
		for cords in self.hurdleCords: pygame.draw.rect(self.gameDisplay, self.hurdleColor, cords)

	def pauseGame(self):
		
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					quit()
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_s:
						return


			self.gameDisplay.fill((200, 200, 200))
			self.message(msg = "Paused.! Press S to continue...", fontSize = 30,\
				xpos = self.gameWidth // 2 - 200, ypos = self.gameHeight // 2)
			pygame.display.update()

	def defaultDisplays(self):
		self.gameDisplay.fill(self.backGroundColor)
		self.makeGrids()
		self.gameDisplay.blit(self.food, self.foodLoc)
		self.makeHurdles()

	def gotFood(self):
		if (self.foodLoc[0] <= self.player.x <= self.foodLoc[0] + self.foodSize or self.foodLoc[0] <= self.player.x + self.player.width <= self.foodLoc[0] + self.foodSize) \
		and (self.foodLoc[1] <= self.player.y <= self.foodLoc[1] + self.foodSize or self.foodLoc[1] <= self.player.y + self.player.height <= self.foodLoc[1] + self.foodSize):
			return True, 10
		return False, -0.1

	def getReward(self):
		# fitness = (1 / np.sqrt((self.player.x - self.foodLoc[0]) ** 2 + (self.player.y - self.foodLoc[1]) ** 2))
		fitness = 0
		i = len(self.hurdleCords) - 2
		while i >= 0 and self.hurdleCords[i][0] > self.player.x: i -= 1
		if i % 3 == 0:
			fitness += 100*max((1 / abs(1 + self.hurdleCords[i][1] - self.player.y)), (1 / abs((1 + self.hurdleCords[i][1] + self.hurdleCords[i][3]) - self.player.y)))
		else:
			fitness += 100*max((1 / abs(self.hurdleCords[i + 1][1] - self.player.y + 1)), (1 / abs((self.hurdleCords[i][1] + self.hurdleCords[i][3]) - self.player.y + 1)))
		fitness = (i + 1) * 2
		return fitness

	def showGame(self, Episode):

		steps = 0
		while not self.gotFood()[0] and steps < 10000: ######
			steps += 1
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return
			
			state, action = self.player.move(self.foodLoc)
			for v in self.player.decideMovement(self.hurdleCords):
				reward = v[1]
				next_state = self.player.getState(self.foodLoc)
				done, r = self.gotFood()
				# print(steps, v[1])
				reward += r + self.getReward()
				self.player.remember(state, action, reward, next_state, done)
				if v[0] == None: break
				self.defaultDisplays()
				self.gameDisplay.blit(v[0], (self.player.x, self.player.y))
				self.player.showVision()
				self.message(msg = 'Episode : {} Exploration : {}'.format(Episode, self.player.brain.epsilon), color = (255, 255, 255),fontSize = 12)
				pygame.display.update()
				# pygame.time.wait(25)
				

		self.player.reset()
		self.player.learn(batch_size = 1024)


	def run(self):
		Episode = 0
		self.pauseGame()
		while True: 
			self.showGame(Episode)
			Episode += 1
		pygame.quit()


if __name__ == '__main__':
	env = Environment()
	env.run()