import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import gym
import torch
from torch.optim import Adam
import numpy as np
import os
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def get_output_folder(parent_dir, env_name):
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def get_class_attr(Cls):
	import re
	return [a for a, v in Cls.__dict__.items()
				if not re.match('<function.*?>', str(v))
				and not (a.startswith('__') and a.endswith('__'))]

def get_class_attr_val(cls):
	attr = get_class_attr(type(cls))
	attr_dict = {}
	for a in attr:
		attr_dict[a] = getattr(cls, a)
	return attr_dict

class DQN(nn.Module):
	def __init__(self, num_inputs, actions_dim):
		super(DQN, self).__init__()

		self.nn = nn.Sequential(
			nn.Linear(num_inputs, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, actions_dim)
		)

	def forward(self, x):
		return self.nn(x)


class Config:
	env: str = None
	gamma: float = None
	learning_rate: float = None
	frames: int = None
	episodes: int = None
	max_buff: int = None
	batch_size: int = None

	epsilon: float = None
	eps_decay: float = None
	epsilon_min: float = None

	state_dim: int = None
	state_shape = None
	state_high = None
	state_low = None
	seed = None
	output = 'out'

	action_dim: int = None
	action_high = None
	action_low = None
	action_lim = None

	use_cuda: bool = None

	checkpoint: bool = False
	checkpoint_interval: int = None

	record: bool = False
	record_ep_interval: int = None

	log_interval: int = None
	print_interval: int = None

	update_tar_interval: int = None

	win_reward: float = None
	win_break: bool = None

class Trainer:
	def __init__(self, agent, env, Config, experment, env_name):
		self.agent = agent
		self.env = env
		self.config = config

		# non-Linear epsilon decay
		epsilon_final = self.config.epsilon_min
		epsilon_start = self.config.epsilon
		epsilon_decay = self.config.eps_decay
		self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
			-1. * frame_idx / epsilon_decay)

		self.outputdir = get_output_folder(self.config.output, self.config.env)
		self.experment = experment
		self.env_name = env_name
		self.agent.save_config(self.outputdir)

	def train(self, pre_fr=0):
		losses = []
		all_rewards = []
		ind_reward = []
		running_reward = 0
		episode_reward = 0
		ep_num = 0
		is_win = False
		esp = []
		true_reward = []

		state = self.env.reset()
		lenght = 0
		totallylenght = 0
		finished = 0
		for fr in range(pre_fr + 1, self.config.frames + 1):
			lenght += 1
			epsilon = self.epsilon_by_frame(fr)
			esp.append(epsilon)
			action = self.agent.act(state, epsilon)

			next_state, reward, done, _ = self.env.step(action)
			self.agent.buffer.add(state, action, reward, next_state, done)

			state = next_state
			episode_reward += reward

			loss = 0
			if self.agent.buffer.size() > self.config.batch_size:
				loss = self.agent.learning(fr)
				losses.append(loss)

			if fr % self.config.print_interval == 0:
				print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (fr, np.mean(true_reward[-20:]), loss, ep_num))
				print("esp {}".format((esp[-1])))

			if done:
				finished += 1
				true_reward.append(episode_reward)
				print("epsoide lenght {}".format(lenght))
				lenght = 0
				state = self.env.reset()
				ind_reward.append(np.mean(true_reward[-20:]))
				running_reward += episode_reward
				all_rewards.append(running_reward/finished)
				episode_reward = 0
				ep_num += 1
				avg_reward = float(np.mean(true_reward[-20:]))

				if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
					is_win = True
					self.agent.save_model(self.env_name, self.experment)
					print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
					if self.config.win_break:
						break

		if not is_win:
			print('Did not solve after %d episodes' % ep_num)
			self.agent.save_model(self.env_name, self.experment)
		return all_rewards, ind_reward

class Tester(object):

	def __init__(self, agent, env, env_name, experment, num_episodes=200, max_ep_steps=400, test_ep_steps=500):
		self.num_episodes = num_episodes
		self.max_ep_steps = max_ep_steps
		self.test_ep_steps = test_ep_steps
		self.agent = agent
		self.env = env
		self.agent.is_training = False
		self.agent.load_weights(env_name,experment)
		self.policy = lambda x: agent.act(x)

	def test(self, debug=False, visualize=False):
		avg_reward = 0
		rewards = []
		for episode in range(self.num_episodes):
			s0 = self.env.reset()
			episode_steps = 0
			episode_reward = 0.

			done = False
			while not done:
				if visualize:
					self.env.render()

				action = self.policy(s0)
				s0, reward, done, info = self.env.step(action)
				episode_reward += reward
				episode_steps += 1

				if episode_steps + 1 > self.test_ep_steps:
					done = True

			if debug:
				print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))
			rewards.append(episode_reward)
			#avg_reward += episode_reward
		rewards = np.array(rewards)
		return np.median(rewards), np.std(rewards)

class ReplayBuffer(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []

	def add(self, s0, a, r, s1, done):
		if len(self.buffer) >= self.capacity:
			self.buffer.pop(0)
		self.buffer.append((s0[None, :], a, r, s1[None, :], done))

	def sample(self, batch_size):
		s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
		return np.concatenate(s0), a, r, np.concatenate(s1), done

	def size(self):
		return len(self.buffer)

class DDQNAgent:
	def __init__(self, config: Config):
		self.config = config
		self.is_training = True
		self.buffer = ReplayBuffer(self.config.max_buff)

		self.model = DQN(self.config.state_dim, self.config.action_dim).cuda()
		self.target_model = DQN(self.config.state_dim, self.config.action_dim).cuda()
		self.target_model.load_state_dict(self.model.state_dict())
		self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

		if self.config.use_cuda:
			self.cuda()

	def act(self, state, epsilon=None):
		if epsilon is None: epsilon = self.config.epsilon_min
		if random.random() > epsilon or not self.is_training:
			state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
			if self.config.use_cuda:
				state = state.cuda()
			q_value = torch.argmax(self.model.forward(state)).cpu().item()
			action = q_value
		else:
			action = random.randrange(self.config.action_dim)
		return action

	def learning(self, fr):
		s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

		s0 = torch.tensor(s0, dtype=torch.float)
		s1 = torch.tensor(s1, dtype=torch.float)
		a = torch.tensor(a, dtype=torch.long)
		r = torch.tensor(r, dtype=torch.float)
		done = torch.tensor(done, dtype=torch.float)

		if self.config.use_cuda:
			s0 = s0.cuda()
			s1 = s1.cuda()
			a = a.cuda()
			r = r.cuda()
			done = done.cuda()

		q_values = self.model(s0).cuda()
		next_q_values = self.model(s1).cuda()
		next_q_state_values = self.target_model(s1).cuda()

		q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
		next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
		expected_q_value = r + self.config.gamma * next_q_value * (1 - done)

		loss = (q_value - expected_q_value.detach()).pow(2).mean()

		self.model_optim.zero_grad()
		loss.backward()
		self.model_optim.step()

		if fr % self.config.update_tar_interval == 0:
			self.target_model.load_state_dict(self.model.state_dict())

		return loss.item()

	def cuda(self):
		self.model.cuda()
		self.target_model.cuda()

	def load_weights(self, output,tag):
		self.model.load_state_dict(torch.load( './DQN_{}.pth'.format(output + tag)))

	def save_model(self, output, tag):
		torch.save(self.model.state_dict(), './DQN_{}.pth'.format(output + tag))

	def save_config(self, output):
		with open(output + '/config.txt', 'w') as f:
			attr_val = get_class_attr_val(self.config)
			for k, v in attr_val.items():
				f.write(str(k) + " = " + str(v) + "\n")




config = Config()
config.env = "LunarLander-v2"
config.gamma = 0.99
config.epsilon = 0.8
config.epsilon_min = 0.01
config.eps_decay = 20000
config.frames = 200000
config.use_cuda = True
config.learning_rate = 1e-3
config.max_buff = 10000
config.update_tar_interval = 512
config.batch_size = 256
config.print_interval = 200
config.log_interval = 200
config.win_reward = 200
config.win_break = True
experment = "DQN baseline 2"
fig, axs = plt.subplots(2)
axs[0].set_ylim(-300, 300)
axs[1].set_ylim(-300, 300)

mean_rewards = []
std_rewards = []
for i in range(5):
	experment = "DQN baseline 2 {}".format(i)
	env = gym.make(config.env)
	config.action_dim = env.action_space.n
	config.state_dim = env.observation_space.shape[0]
	agent = DDQNAgent(config)

	trainer = Trainer(agent, env, config, experment, config.env)
	rewards, ind_rewards = trainer.train()

	p1 = axs[0].plot(np.arange(len(rewards)), rewards)

	with open("rewards{}".format(i) + experment, 'wb') as fp:
		pickle.dump(rewards, fp)



	p1 = axs[1].plot(np.arange(len(ind_rewards)), ind_rewards)

	with open("ind_rewards{}".format(i) + experment, 'wb') as fp:
		pickle.dump(ind_rewards, fp)

	tester = Tester(agent, env, config.env, experment)
	mean, std = tester.test()
	mean_rewards.append(mean)
	std_rewards.append(std)

fig4, ax4 = plt.subplots(1)
fig4.tight_layout()

ax4.set_xlabel('Run')
ax4.set_ylabel('Reward')
ax4.set_title("5 run median score {}".format(np.median(mean_rewards)))
ax4.bar(np.arange(len(mean_rewards)), mean_rewards, yerr=std_rewards, align='center', alpha=0.5)
fig4.savefig(config.env + " reward " + experment)
with open("mean_rewards{}" + experment, 'wb') as fp:
	pickle.dump(mean_rewards, fp)
with open("std_rewards{}" + experment, 'wb') as fp:
	pickle.dump(std_rewards, fp)

fig.savefig(config.env + " reward plot " + experment)