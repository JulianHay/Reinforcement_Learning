# Bipedal Walker Assignment

import numpy as np
import gym

class ARS_Agent():

	def __init__(self,n_steps=1000,episode_length=2000,noise=0.03,n_deltas=16,n_best_deltas=16,learning_rate=0.02,env_name='BipedalWalker-v2',display=False):

		self.n_steps = n_steps 		
		self.noise = noise
		self.n_deltas = n_deltas
		self.n_best_deltas = n_best_deltas
		self.learning_rate = learning_rate
		self.env_name = env_name
		self.env = gym.make(self.env_name)
		self.episode_length = episode_length
		self.input_size = self.env.observation_space.shape[0]
		self.output_size = self.env.action_space.shape[0]
		self.theta = np.zeros((self.output_size,self.input_size))
		self.n = np.zeros(self.input_size)
		self.mean = np.zeros(self.input_size)
		self.mean_diff = np.zeros(self.input_size)
		self.var = np.zeros(self.input_size)
		self.display = display



	def normalize(self, inputs):
		self.n += 1
		last_mean = self.mean.copy()
		self.mean += (inputs - last_mean) / self.n
		self.mean_diff = (inputs - last_mean) * (inputs - self.mean) 					
		self.var = np.sqrt(self.mean_diff / self.n).clip(min = 1e-2)					
		return (inputs - self.mean) / self.var

	def action(self, observation, delta, variation):

		if variation == None:
			return self.theta.dot(observation)
		if variation == '-':
			return (self.theta - self.noise * delta).dot(observation)
		if variation == '+':
			return (self.theta + self.noise * delta).dot(observation)


	def get_rewards(self, delta=None, variation=None):
		observation = self.env.reset()
		done = False
		rewards = 0
		n = 0

		while not done and n < self.episode_length:
			if self.display:
				self.env.render()
			observation = self.normalize(observation)
			action = self.action(observation, delta, variation)
			observation, reward, done, _ = self.env.step(action)
			reward = max(min(reward,1),-1)									
			rewards += reward
			n += 1	
				
		return rewards

	def update_policy(self, rollouts, sigma_rewards):
		step = np.zeros(self.theta.shape)
		for r_pos, r_neg, delta in rollouts:
			step += np.sum(r_pos - r_neg) * delta
		self.theta += self.learning_rate / (self.n_best_deltas * sigma_rewards) * step


	def train(self):

		reward_delta = 50

		for episode in range(1,self.n_steps+1):
			
			# initialize random noise and rewards
			deltas = [np.random.randn(*self.theta.shape) for _ in range(self.n_deltas)]
			pos_rewards = [0] * self.n_deltas
			neg_rewards = [0] * self.n_deltas

			# play episode for each pos/neg delta
			for i in range(self.n_deltas):
				pos_rewards[i] = self.get_rewards(deltas[i], '+')
				neg_rewards[i] = self.get_rewards(deltas[i], '-')

			# calculate standard deviation of all rewards
			sigma_rewards = np.array(pos_rewards + neg_rewards).std()

			# sort rollouts by max(pos_r,neg_r) and select the best n
			rollouts = [(pos_rewards[i],neg_rewards[i],deltas[i]) for i in np.argsort([-max(pos_rewards[i],neg_rewards[i]) for k in range(self.n_deltas)])][:self.n_best_deltas]

			# update policy
			self.update_policy(rollouts, sigma_rewards)

			if episode % 50 == 0:
				self.display = True

			# run episode with new policy
			eval_reward = self.get_rewards()
			print('Episode: ', episode , 'Reward: ', eval_reward)


			if episode % 50 == 0:
				self.display = False
				self.env.close()
				

if __name__ == '__main__':

	load = True
	train = False
	
	ARS_Agent = ARS_Agent(n_steps=1000)

	if load:
		[ARS_Agent.theta, ARS_Agent.n, ARS_Agent.mean, ARS_Agent.mean_diff, ARS_Agent.var] = np.load('BipedalWalker_Data.npy')			

	if train:
		ARS_Agent.train()
		np.save('BipedalWalker_Data.npy',[ARS_Agent.theta, ARS_Agent.n, ARS_Agent.mean, ARS_Agent.mean_diff, ARS_Agent.var])

	else:
		ARS_Agent.display = True
		reward = ARS_Agent.get_rewards()	
		print('Reward: ', reward)
