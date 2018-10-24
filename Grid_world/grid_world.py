#### Grid

import numpy as np
import pygame
import itertools

class Grid:

	def __init__(self,width,height, start, step_cost, obey_prob, epsilon, gamma, learning_rate, Lambda):

		self.width = width
		self.height = height
		self.x = start[0]
		self.y = start[1]
		self.start = start
		self.step_cost = step_cost
		self.obey_prob = obey_prob
		self.epsilon = epsilon
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.Lambda = Lambda

	def set_state(self,s):
		self.x = s[0]
		self.y = s[1]

	def current_state(self):
		return (self.x , self.y)

	def set(self, rewards, actions, obey_prob):
		self.rewards = rewards
		self.actions = actions
		self.obey_prob = obey_prob

	def states(self):
		return self.actions.keys()

	def is_terminal(self,s):
		return s not in self.actions

	def random_action(self,action):
		new_action = action
		if  np.random.random() > self.obey_prob:
			while new_action == action:
				new_action = np.random.choice(self.action)
			return new_action
		else:
			return action

	def move(self,action):

		action = self.random_action(action)
		if action in self.actions[(self.x,self.y)]:
			if action == 'U':
				self.y -= 1
			elif action == 'D':
				self.y += 1
			elif action == 'L':
				self.x -= 1
			elif action == 'R':
				self.x += 1
		reward = self.rewards.get((self.x, self.y), 0)#self.rewards[self.x,self.y]
		return reward

	def game_over(self):
		return (self.x,self.y) not in self.actions

	def draw_grid(self):

		field_size = 50
		screen_size = (field_size*self.width, field_size*self.height)
		
		screen = pygame.display.set_mode(screen_size)

		for n in range(self.height):

			pygame.draw.line(screen, (150,150,150), (0,n*field_size),(screen_size[0],n*field_size))

		for n in range(self.width):

			pygame.draw.line(screen, (150,150,150), (n*field_size,0),(n*field_size,screen_size[1]))

		# draw agent
		pygame.draw.circle(screen,(0,0,250),(int(field_size*(0.5+self.x)),int(field_size*(0.5+self.y))),17)

		# draw terminal states  
		terminal_states = np.nonzero(np.subtract(list(self.rewards.values()),self.step_cost))

		for i in np.nditer(terminal_states):

			if list(self.rewards.values())[i] < 0:

				pygame.draw.line(screen,(250,0,0),(list(self.rewards.keys())[i][0]*field_size, list(self.rewards.keys())[i][1]*field_size),((list(self.rewards.keys())[i][0]+1)*field_size, (list(self.rewards.keys())[i][1]+1)*field_size))
				pygame.draw.line(screen,(250,0,0),(list(self.rewards.keys())[i][0]*field_size, (list(self.rewards.keys())[i][1]+1)*field_size),((list(self.rewards.keys())[i][0]+1)*field_size, list(self.rewards.keys())[i][1]*field_size))
			else:

				points = []
				for ang in range(8):    
						points.extend([(list(self.rewards.keys())[i][0]*field_size+int(field_size/2+np.cos(np.pi/4*ang)*16), list(self.rewards.keys())[i][1]*field_size+int(field_size/2+np.sin(np.pi/4*ang)*16)),
							(list(self.rewards.keys())[i][0]*field_size+int(field_size/2+np.cos(np.pi/8+np.pi/4*ang)*8), list(self.rewards.keys())[i][1]*field_size+int(field_size/2+np.sin(np.pi/8+np.pi/4*ang)*8))])
				pygame.draw.polygon(screen,(0,250,0),points)

		for i,j in itertools.product(range(self.width),range(self.height)):
			if (i,j) not in self.actions and self.rewards[(i,j)]==self.step_cost:
				pygame.draw.rect(screen,(150,150,150), (i*field_size, j*field_size,field_size,field_size))

def make_grid(width, height, start, actions, rewards, step_cost = 0, obey_prob=1, epsilon=0.8, gamma=0.9, learning_rate=0.01, Lambda=0.6):


	grid = Grid(width,height,start,step_cost,obey_prob,epsilon,gamma,learning_rate,Lambda)
	grid.set(rewards, actions, obey_prob)

	return grid

