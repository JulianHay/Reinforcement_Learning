## Deep Q Learning

import tensorflow as tf
import numpy as np
import gym
import cv2
from collections import deque

# Hyperparameters

env = gym.make('LunarLander-v2')				
state_size = [84,84,4]
action_size = env.action_space.n
learning_rate = 0.001
n_episodes = 500
max_steps = 50000
batch_size = 64
gamma = 0.9
epsilon = 1
epsilon_decay_rate = 10000
stack_size = 4
max_memory_size = 10000

train = True
render = False

class DQN():

	def __init__(self, state_size, action_size, learning_rate, name='DQN'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):

			self.inputs = tf.placeholder(tf.float32, [None, *state_size], name = "inputs")						# self. ???
			self.actions = tf.placeholder(tf.float32, [None, self.action_size], name = "actions")
			self.Q_target = tf.placeholder(tf.float32, [None], name = "target")

			# Define CNN

			self.conv1 = tf.layers.conv2d(inputs = self.inputs, filters = 32, kernel_size = [8,8], strides = [4,4], padding = 'VALID', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "conv1")
			self.conv1_out = tf.nn.relu(self.conv1, name = "conv1_out")
			self.conv2 = tf.layers.conv2d(inputs = self.conv1_out, filters = 64, kernel_size = [4,4], strides = [2,2], padding = 'VALID', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "conv2")
			self.conv2_out = tf.nn.relu(self.conv2, name = "conv2_out")
			self.conv3 = tf.layers.conv2d(inputs = self.conv1_out, filters = 64, kernel_size = [3,3], strides = [2,2], padding = 'VALID', kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "conv3")
			self.conv3_out = tf.nn.relu(self.conv3, name = "conv3_out")
			self.flatten = tf.contrib.layers.flatten(self.conv3_out)
			self.fc = tf.layers.dense(inputs = self.flatten, units = 512, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "fc")
			self.output = tf.layers.dense(inputs = self.fc, units = self.action_size, activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "output")

			self.Q = tf.reduce_sum(tf.multiply(self.output,self.actions))

			self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q))

			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

class Environment():

	def __init__(self, stack_size, max_memory_size):
		self.new_episode = True
		self.stacked_frames = deque([np.zeros((84,84), dtype = np.int) for i in range(stack_size)], maxlen=4)
		self.memory = deque(maxlen=max_memory_size)

	def preprocess(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.normalize(frame, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = -1) 
		# resize
		frame = cv2.resize(frame, (84,84), interpolation = cv2.INTER_LINEAR)

		# stack
		if self.new_episode:
			self.stacked_frames.extend([frame,frame,frame,frame])
			stacked_state = np.stack(self.stacked_frames, axis=2)
		else:
			self.stacked_frames.append(frame)
			stacked_state = np.stack(self.stacked_frames, axis=2)

		return stacked_state

	def add_memory(self, observation):
		self.memory.append(observation)

	def sample_memory(self):
		idx = np.random.choice(np.arange(len(self.memory)), size=batch_size, replace = False)
		return [self.memory[i] for i in idx]

	def initialize_memory(self):

		print('Initialize replay memory...', end="", flush=True)

		rewards = []
		env.reset()

		while len(self.memory) < max_memory_size:
			
			state = env.render(mode='rgb_array')				
			state = self.preprocess(state)
			action = np.random.randint(0,4)
			_ , reward, done, _ = env.step(action)
			next_state = env.render(mode='rgb_array')
			next_state = self.preprocess(next_state)
			one_hot_action = [0,0,0,0]
			one_hot_action[action] = 1
			# add experience to replay memory
			self.add_memory((state,one_hot_action,reward,next_state,done))
			state = next_state

			if done:
				env.reset()

		print('		done', flush=True)


class Agent():

	def __init__(self):
		self.env = Environment(stack_size,max_memory_size)
		self.DQN = DQN(state_size, action_size, learning_rate, name = 'DQN')
		self.DQN_target = DQN(state_size, action_size, learning_rate, name = 'DQN_target')
		self.sess = tf.Session()

	def action(self,state,epsilon):
		
		if np.random.random() < (1 - epsilon):
			Q = self.sess.run(self.DQN.output, feed_dict = {self.DQN.inputs : state.reshape((1, *state.shape))})
			a = np.argmax(Q) 
		else:
			a = np.random.randint(0,4)
		return a

	def train(self):

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter("/tensorboard/dqn/1")
		tf.summary.scalar("Loss", self.DQN.loss)

		self.env.initialize_memory()		




		self.sess.run(tf.global_variables_initializer())
		frame_idx = 0

		for episode in range(1,n_episodes):

			done = False
			step = 0

			rewards = []
			env.reset()
			state = env.render(mode='rgb_array')
			
			state = self.env.preprocess(state)
			self.env.new_episode = False

			while step < max_steps and not done:

				frame_idx += 1
				step += 1

				# decrease epsilon
				epsilon = max(0.02, 1-frame_idx/epsilon_decay_rate)

				action = self.action(state,epsilon)
				_ , reward, done, _ = env.step(action)
				next_state = env.render(mode='rgb_array')
				next_state = self.env.preprocess(next_state)

				if render:
					env.render()

				rewards.append(reward)

				one_hot_action = [0,0,0,0]
				one_hot_action[action] = 1

				# add experience to replay memory
				self.env.add_memory((state,one_hot_action,reward,next_state,done))

				if done:
					self.env.new_episode = True
					total_reward = np.sum(rewards)				
				else:						
					state = next_state

				# update target network every 1000 frames
				if frame_idx % 1000 == 0:
					DQN_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DQN")
					DQN_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DQN_target")
					self.sess.run([i.assign(j) for i,j in zip(DQN_target_params,DQN_params)])

				# get mini batches
				batch = self.env.sample_memory()
				states_mb = np.array([i[0] for i in batch],ndmin = 3)
				actions_mb = np.array([i[1] for i in batch])
				rewards_mb = np.array([i[2] for i in batch])
				next_states_mb = np.array([i[3] for i in batch],ndmin = 3)
				dones_mb = np.array([i[4] for i in batch])

				# calculate Q values with target net
				Q_target_batch = []				
				Q_next_state = self.sess.run(self.DQN_target.output, feed_dict = {self.DQN_target.inputs:next_states_mb})

				for i in range(len(batch)):

					terminal = dones_mb[i]

					if terminal:
						Q_target_batch.append(rewards_mb[i])
					else:
						Q_target_batch.append(rewards_mb[i] + gamma * np.max(Q_next_state[i]))

				Q_target_batch = np.array([i for i in Q_target_batch])

				loss, _ = self.sess.run([self.DQN.loss, self.DQN.optimizer], feed_dict={self.DQN.inputs:states_mb, self.DQN.Q_target:Q_target_batch, self.DQN.actions:actions_mb})


				summary = self.sess.run(tf.summary.merge_all(), feed_dict={self.DQN.inputs:states_mb, self.DQN.Q_target:Q_target_batch, self.DQN.actions:actions_mb})

				writer.add_summary(summary,episode)

				if done:
					print('Episode: {}'.format(episode), 'Total reward: {}'.format(total_reward), 'Epsilon: ', epsilon, 'Training Loss {:.4f}'.format(loss))	

				if episode % 100 == 0:
					save_path = saver.save(self.sess, "./models/model.ckpt")



if __name__ == '__main__':

	DQN_Agent = Agent()

	DQN_Agent.train()
















