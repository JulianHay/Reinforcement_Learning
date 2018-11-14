## Policy Gradient


import tensorflow as tf
import numpy as np
from collections import deque
import gym

n_episodes = 500
max_steps = 5000
gamma = 0.9
learning_rate = 0.01

class Net():

	def __init__(self,action_size,state_size,learning_rate):
		# define variables
		self.x = tf.placeholder(tf.float32, [None, state_size], name="inputs")
		self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")		
		self.returns = tf.placeholder(tf.float32, [None,], name="returns")
		self.mean_reward = tf.placeholder(tf.float32 , name="mean_reward")
		# define neural network  
		fc1 = tf.contrib.layers.fully_connected(inputs = self.x, num_outputs = 10, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
		fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = action_size, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
		fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
		self.output = tf.nn.softmax(fc3)

		crossentropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = self.actions)
		self.loss = tf.reduce_mean(crossentropy_loss * self.returns)

		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)	# RMSPropOptimizer

class Agent():

	def __init__(self, learning_rate, gamma):
		self.env = gym.make('CartPole-v0')
		self.render = False
		self.Net = Net(self.env.action_space.n, 4, learning_rate)
		self.sess = tf.Session()
		self.state_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		self.all_rewards =  []
		self.total_reward = 0
		self.gamma = gamma


	def step(self,state):
		action_prob = self.sess.run(self.Net.output, feed_dict = {self.Net.x : state.reshape([1,4])})
		action = np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel())
		next_state, reward, done, _  = self.env.step(action)
		one_hot_action = np.zeros(self.env.action_space.n)
		one_hot_action[action] = 1
		self.state_buffer.append(state)
		self.action_buffer.append(one_hot_action)
		self.reward_buffer.append(reward)
		self.total_reward += reward
		return next_state, done

	def reset_buffer(self):
		self.state_buffer = []
		self.action_buffer = []
		self.reward_buffer = []
		self.total_reward = 0

	def calc_returns(self):
		returns = np.zeros(len(self.reward_buffer))
		cum = 0
		for i in reversed(range(len(self.reward_buffer))):
			cum = cum * self.gamma + self.reward_buffer[i]
			returns[i] = cum  
		mean = np.mean(returns)
		std = np.std(returns)
		returns = (returns-mean)/std
		return returns 

	def train(self):
		self.sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter("/tensorboard/pg/1")
		tf.summary.scalar("Loss", self.Net.loss)
		tf.summary.scalar("Reward_mean", self.Net.mean_reward)
		write = tf.summary.merge_all()
		saver = tf.train.Saver()

		for episode in range(n_episodes):
			state = self.env.reset()
			done = False
			step = 0
			while step < max_steps and not done:
				step += 1
				state, done = self.step(state)

			self.all_rewards.append(self.total_reward)
			mean_reward = np.divide(np.sum(self.all_rewards),episode+1)
			max_reward = np.amax(self.all_rewards)


			returns = self.calc_returns()

			# update 
			loss, _ = self.sess.run([self.Net.loss, self.Net.optimizer], feed_dict = {		self.Net.x : self.state_buffer, 
																							self.Net.actions : self.action_buffer, 
																							self.Net.returns : returns})	

			

			summary = self.sess.run(write, feed_dict = {self.Net.x : np.vstack(np.array(self.state_buffer)), 
														self.Net.actions : np.vstack(np.array(self.action_buffer)), 
														self.Net.returns : returns,
														self.Net.mean_reward : mean_reward})
			writer.add_summary(summary,episode)

			print("==========================================")
			print("Episode: ", episode)
			print("Reward: ", self.total_reward)
			print("Mean Reward", mean_reward)
			print("Max reward so far: ", max_reward)

			if episode % 100 == 0:
				saver.save(self.sess, "./models/model.ckpt")
				print("Model saved")

			self.reset_buffer()





if __name__ == '__main__':

	PG_Agent = Agent(learning_rate, gamma)

	PG_Agent.train()



