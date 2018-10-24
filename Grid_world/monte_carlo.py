## Monte Carlo

from grid_world import make_grid
import pygame
import numpy as np
import itertools
import matplotlib.pyplot as plt

possible_actions = ('U','D','L','R')
n_episodes = 10001

def max_dict(d):
	max_key = None
	max_val = float('-inf')
	for k, v in d.items():
		if v > max_val:
			max_val = v
			max_key = k
	return max_key, max_val

def epsilon_action(a, epsilon=0.1):
	p = np.random.random()
	if p < (1 - epsilon):
		return a
	else:
		return np.random.choice(possible_actions)


def play_game(grid, policy, epsilon, visibility = False):

	s = grid.start

	grid.set_state(s) 

	a = epsilon_action(policy[s], epsilon)

	states_actions_rewards = [(s, a, 0)]

	if visibility:
		pygame.display.init()

	running = True

	while running:
		if visibility:				
			grid.draw_grid()
			pygame.display.flip()
			pygame.time.delay(300)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False
				#sys.exit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						running = False

		r = grid.move(a)
		s = grid.current_state()

		if grid.game_over():
			states_actions_rewards.append((s, None, r))
			running = False
		else:
			a = epsilon_action(policy[s],epsilon)
			states_actions_rewards.append((s, a, r))
		

	if visibility:
		pygame.display.quit()

	G = 0
	states_actions_returns = []
	first = True

	for s,a,r in reversed(states_actions_rewards):

		if first:
			first=False
		else:
			states_actions_returns.append((s,a,G))
		G = r + gamma * G

	states_actions_returns.reverse() 
	return states_actions_returns

def monte_carlo(grid):

	policy = {}
	for s in grid.actions.keys():
			policy[s] = np.random.choice(possible_actions)

	#print(policy)
	Q = {}

	returns = {}

	delta = []

	states = grid.states()

	for s in states:
		Q[s] = {}
		for a in possible_actions:
			Q[s][a] = 0
			returns[(s,a)] = []

	for n in range(n_episodes):

		epsilon = np.divide(grid.epsilon, (1+n*0.01))

		if n % 1000 == 0:
			print(n)
		if n % 10000 == 0 and not n == 0:
			visibility = True
		else: 
			visibility = False

		states_actions_returns = play_game(grid, policy, epsilon, visibility)

		seen_states_actions = set()
		max_change = 0

		for s,a,G in states_actions_returns:

			if (s,a) not in seen_states_actions:
				Q_old = Q[s][a] 
				returns[(s,a)].append(G) 
				Q[s][a] = np.mean(returns[(s,a)])

				max_change = max(max_change,np.abs(Q_old-Q[s][a]))
				seen_states_actions.add((s,a))

		delta.append(max_change)

		for s in policy.keys():
			a, _ = max_dict(Q[s])
			policy[s] = a
	
	V = {}
	for s in policy.keys():
		V[s] = max_dict(Q[s])[1]		

	return policy, V, delta
			
def print_values(V, g):
	for i in range(g.height):
		print("---------------------------")
		for j in range(g.width):
			v = V.get((j,i), 0)
			if v >= 0:
				print(" %.2f|" % v, end="")
			else:
				print("%.2f|" % v, end="") # -ve sign takes up an extra space
		print("")

def print_policy(P, g):
	for i in range(g.height):
		print("---------------------------")
		for j in range(g.width):
			a = P.get((j,i), ' ')
			print("  %s  |" % a, end="")
		print("")

if __name__ == '__main__':

	gamma = 0.9
	epsilon = 1
	
	height = 4
	width = 6
	
	step_cost = -0.2

	start = (0,3)
	actions = {	(0, 0): ('R','D'), 			(1, 0): ('L','R'), 									 			(3, 0): ('L','R', 'D'), 		(4, 0): ('L','R','D'), 		
				(0, 1): ('U','D'), 								 			(2, 1): ('U','R','D'), 			(3, 1): ('U','L','R','D'), 		(4, 1): ('U','L','R', 'D'), 	(5, 1): ('U','L','D'),
				(0, 2): ('U','R','D'),		(1, 2): ('L','R','D'), 			(2, 2): ('U','L','R','D'), 		(3, 2): ('U','L','R'), 											(5, 2): ('U','L','D'),
				(0, 3): ('U','R'), 			(1, 3): ('U','L','R'), 			(2, 3): ('U','L','R'), 			(3, 3): ('U','L','R'), 			(4, 3): ('U','L','R'), 			(5, 3): ('U','R') }

	rewards = {(i,j):step_cost for i,j in itertools.product(range(width),range(height))}
	rewards.update({(2,0):-1,(4,2):-1,(5,0):1})
	obey_prob = 1

	# height = 3
	# width = 4
	# start = (0,2)

	# rewards.update({(3, 0): 1, (3, 1): -1})
	# actions = {	(0, 0): ('R','D'), 			(1,0): ('L','R'), 			(2,0): ('D','L','R'), 			
	#  			(0, 1): ('U','D'), 									 	(2, 1): ('U','D','R'), 		
	#  			(0, 2): ('U','R'),			(1, 2): ('L','R'), 			(2, 2): ('U','L','R'), 		(3, 2): ('U','L')}

	grid = make_grid(width, height, start, actions, rewards, step_cost, obey_prob, epsilon, gamma)

	policy, V, delta = monte_carlo(grid)

	np.savez('monte_carlo_data.npz',V, policy, delta)

	# print rewards
	print("rewards:")
	print_values(grid.rewards, grid)

	

	print("final values:")
	print_values(V, grid)
	print("final policy:")
	print_policy(policy, grid)

	plt.plot(delta)
	plt.show()

