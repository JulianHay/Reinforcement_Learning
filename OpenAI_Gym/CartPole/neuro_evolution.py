## neroevolution

import neat
import gym
import numpy as np
import visualize

num_gen = 1000

def eval_genomes(genomes,config):
	env = gym.make('CartPole-v0')
	for genome_id, genome in genomes:
		genome.fitness = 0
		net = neat.nn.FeedForwardNetwork.create(genome, config)

		done = False
		state = env.reset()
		total_reward = 0

		while not done:
			action = np.argmax(net.activate((state)))
			state, reward, done, _ = env.step(action)
			total_reward += reward

		genome.fitness = total_reward


if __name__ == '__main__':

	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'Neat_Config.txt')

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(False))

	winner = p.run(eval_genomes,num_gen)

	

	# play game
	env = gym.make('CartPole-v0')
	net = neat.nn.FeedForwardNetwork.create(winner, config)
	total_rewards = []
	for i in range(100):
		done = False
		total_reward = 0
		state = env.reset()
		while not done:
			#env.render()
			action = np.argmax(net.activate((state)))
			state, reward, done, _ = env.step(action)
			total_reward += reward

		print('Total Reward: ', total_reward)
		total_rewards.append(total_reward)
	mean_reward = np.sum(total_rewards)/100
	print('Mean Reward: ', mean_reward)

	visualize.draw_net(config, winner, True)