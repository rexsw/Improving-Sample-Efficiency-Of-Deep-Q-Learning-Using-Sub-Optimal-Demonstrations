import matplotlib as mpl
#mpl.use('Agg')
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

device = torch.device("cuda:0")
print(torch.cuda.is_available())

class Pclass(torch.nn.Module):
	def __init__(self,layer_size,time,state):
		super(Pclass, self).__init__()

		#try dropout
		#cross entory
		#different acvtors
		#smaller net
		#subtrajators
		# torch.set_default_dtype("long")
		self.size = time*state
		self.actor = torch.nn.Sequential(
		)
		# self.actor.add_module('c1',torch.nn.Conv1d(1, 4, 5))
		# #self.actor.add_module('cb1',	torch.nn.BatchNorm1d(num_features=layer_size))
		# self.actor.add_module('cd1',	torch.nn.Dropout())
		# self.actor.add_module('ca1',	torch.nn.LeakyReLU())
		# self.actor.add_module('c2',torch.nn.Conv1d(4, 1, 5))
		# #self.actor.add_module('cb2',	torch.nn.BatchNorm1d(num_features=layer_size))
		# self.actor.add_module('cd2',	torch.nn.Dropout())
		# self.actor.add_module('ca2',	torch.nn.LeakyReLU())


		self.actor.add_module('l1',torch.nn.Linear(time*state, layer_size))
		#self.actor.add_module('b1',	torch.nn.BatchNorm1d(num_features=layer_size))
		self.actor.add_module('d1',	torch.nn.Dropout(p=0.3))
		self.actor.add_module('a1',	torch.nn.LeakyReLU())
		self.actor.add_module('l4',torch.nn.Linear(layer_size, layer_size))
		#self.actor.add_module('b2',	torch.nn.BatchNorm1d(num_features=layer_size))
		self.actor.add_module('d2',	torch.nn.Dropout(p=0.3))
		self.actor.add_module('a4',torch.nn.LeakyReLU())
		#self.actor.add_module('b3',	torch.nn.BatchNorm1d(num_features=layer_size))
		self.actor.add_module('d2',	torch.nn.Dropout(p=0.3))
		self.actor.add_module('l5',torch.nn.Linear(layer_size, layer_size))
		self.actor.add_module('a5',torch.nn.LeakyReLU())
		self.actor.add_module('d3',	torch.nn.Dropout(p=0.3))
		self.actor.add_module('l6',torch.nn.Linear(layer_size, layer_size))
		self.actor.add_module('a6',torch.nn.LeakyReLU())
		self.actor.add_module('d4',	torch.nn.Dropout(p=0.3))
		self.actor.add_module('l7',torch.nn.Linear(layer_size, layer_size))
		self.actor.add_module('a7',torch.nn.LeakyReLU())
		self.actor.add_module('d5',	torch.nn.Dropout(p=0.3))
		self.actor.add_module('l8',torch.nn.Linear(layer_size,2))
		#self.actor.add_module('a8',torch.nn.ReLU())
		#self.actor.add_module('out', torch.nn.Softmax(dim=1))
		self.actor.to(device)
		self.epochs = 2000
		self.MseLoss = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001, betas=(0.9,0.999))

	def forward(self):
		raise NotImplementedError

	def update(self, train,test, name):
		samples = torch.tensor(test[:,0,:]).float().to(device)
		p_results = torch.squeeze(self.actor(samples))
		results = test[:,1,0]
		p_results = (torch.argmax(p_results, dim=1).detach().cpu().numpy())
		print("test acc")
		print(np.sum(np.ravel(results == np.ravel(p_results))/len(np.ravel(p_results))))


		samples = torch.tensor(train[:,0,:]).float().to(device)
		pp_results = torch.squeeze(self.actor(samples))
		results = train[:,1,0]
		p_results = (torch.argmax(pp_results, dim=1).detach().cpu().numpy())
		print("train acc")
		print(np.sum(np.ravel(results == np.ravel(p_results))/len(np.ravel(p_results))))

		#samples = torch.tensor(train[:,:,0]).float().to(device)
		#results = np.reshape(results,(len(results),1))
		#results = torch.tensor(train[:,0,1]).float().to(device)
		#print(results.requires_grad)
		losses = []
		for _ in range(self.epochs):
			# np.random.shuffle(train)
			samples = torch.tensor(train[:,0,:]).float().to(device)
			#results = np.reshape(results,(len(results),1))
			results = torch.tensor(train[:,1,0]).long().to(device)
			self.optimizer.zero_grad()
			p_results = torch.squeeze(self.actor(samples))
			loss = self.MseLoss(p_results, results)
			#loss = Variable(loss, requires_grad = True)
			#print(loss)
			loss_result = loss.item()
			#print(loss_result)
			losses.append(loss_result)
			loss.backward()
			self.optimizer.step()
			#print(self.actor.l1.weight.grad)
		# 	graph = make_dot(loss)
		# 	break
		# graph.format  = 'pdf'
		# graph.render('round-table.gpdf', view=True)
		# plt.clf()
		# p1 = plt.plot(losses)
		# plt.ylabel("eps")
		# plt.xlabel("loss")
		# plt.savefig("losscurve" + " plot")
		samples = torch.tensor(test[:,0,:]).float().to(device)
		p_results = torch.squeeze(self.actor(samples))
		results = test[:,1,0]
		p_results = (torch.argmax(p_results, dim = 1).detach().cpu().numpy())
		print("test acc")
		print(np.sum(np.ravel(results == np.ravel(p_results))/len(np.ravel(p_results))))

		samples = torch.tensor(train[:,0,:]).float().to(device)
		p_results = torch.squeeze(self.actor(samples))
		results = train[:,1,0]
		p_results =  (torch.argmax(p_results, dim = 1).detach().cpu().numpy())
		missclass = []
		for i, x in enumerate(results):
			if not x == p_results[i]:
				missclass.append((x,train[i,:,2]))
		print("train acc")
		print(np.sum(np.ravel(results == np.ravel(p_results))/len(np.ravel(p_results))))
		torch.save(self.actor.state_dict(), './classifier_{}.pth'.format(name))
		return missclass


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


			# if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
			# 	self.agent.save_checkpoint(fr, self.outputdir)

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

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return (e_x / e_x.sum(axis=0))

class ReplayBuffer(object):
	def __init__(self, capacity, pclass):
		self.capacity = capacity
		self.buffer = []
		self.current = []
		self.scores = []
		self.pclass = pclass
		self.probs = []
		self.rewardtotal = []
		self.deleted = []
		self.indexs = []

		self.reward_history = []
		self.score_history = []

	def add(self, s0, a, r, s1, done):
		self.current.append((s0[None, :], a, r, s1[None, :], done))
		if done:
			state = []
			reward = 0
			rewards = []
			for x in self.current:
				x = x[0].tolist()[0]
				state.extend(x)
				rewards.append(x[2])
			
			self.reward_history.append(sum(rewards))

			inputarray = np.zeros((1,self.pclass.size))
			if len(state) > self.pclass.size:
				inputarray[0,:] = state[:self.pclass.size]
			else:
				inputarray[0,:len(state)] = state
	
			inputarray = torch.tensor(inputarray).float().cuda()
			score = torch.softmax(self.pclass.actor(inputarray), dim = 1).cpu().detach().numpy()
			#score = np.log(self.pclass.actor(inputarray).cpu().detach().numpy())
			index1 = len(self.buffer)
			self.buffer.extend(self.current)
			index2 = len(self.buffer)
			self.indexs.append((index1, index2))
			self.rewardtotal.extend(rewards)
			self.score_history.append(score[0][1])
			self.scores.extend([score[0][1]]*len(self.current))
			self.probs = softmax(self.scores).tolist()	
			self.current = []
			deleted = 0
			while len(self.buffer) + len(self.current) >= self.capacity:
				index = self.indexs[0] 
				del self.buffer[index[0]:index[1]]
				del self.scores[index[0]:index[1]]
				del self.probs[index[0]:index[1]]
				deleted += sum(self.rewardtotal[index[0]:index[1]])
				del self.rewardtotal[index[0]:index[1]]
				indexdif = index[1] - index[0]
				self.indexs = [(x[0] - indexdif,x[1] - indexdif) if x[0] > index[1] else x  for x in self.indexs]


				# cut =1 
				# idx = np.argpartition(np.array(self.scores), cut)
				# index = idx[cut:].tolist()[0]
				# for x in self.indexs:
				# 	if index <= x[1] and index >= x[0]:
				# 		index = x
				# 		break
				# del self.buffer[index[0]:index[1]]
				# del self.scores[index[0]:index[1]]
				# del self.probs[index[0]:index[1]]
				# deleted += sum(self.rewardtotal[index[0]:index[1]])
				# del self.rewardtotal[index[0]:index[1]]
				# indexdif = index[1] - index[0]
				# self.indexs = [(x[0] - indexdif,x[1] - indexdif) if x[0] > index[1] else x  for x in self.indexs]
				#self.probs = np.delete(self.probs,idx[:cut])
				# idx.sort()
				# start = 0
				# for i,x in enumerate(idx[:cut]):
				# 	del self.buffer[x-i]
				# 	del self.scores[x-i]
				# 	del self.probs[x-i]
				# 	deleted += (self.rewardtotal[x-i])
				# 	del self.rewardtotal[x-i]
			self.deleted.append(deleted)

	def sample(self, batch_size):
		s0, a, r, s1, done  = zip(*random.choices(self.buffer,k = batch_size , weights = self.probs))
		#s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
		return np.concatenate(s0), a, r, np.concatenate(s1), done

	def size(self):
		return len(self.buffer)

class DDQNAgent:
	def __init__(self, Config, pclass):
		self.config = config
		self.is_training = True
		self.buffer = ReplayBuffer(self.config.max_buff, pclass)

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
		# Notice that detach the expected_q_value
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
#config.frames = 1500
config.use_cuda = True
config.learning_rate = 1e-3
config.max_buff = 10000
config.update_tar_interval = 512
config.batch_size = 256
config.print_interval = 200
config.log_interval = 200
config.win_reward = 200
config.win_break = True
#experment = "DQN baseline"
experment = "DQN baseline 50"
env = gym.make(config.env)
config.action_dim = env.action_space.n
config.state_dim = env.observation_space.shape[0]
max_time = 300
state_size = config.state_dim
pclass = Pclass(128,max_time,state_size)
agent = DDQNAgent(config,pclass)

demos = 1000
samples = np.zeros((demos,3,max_time*config.state_dim))
rewards = 0
for i_episode in range(1, demos//2):
	state = env.reset()
	done = False
	stop = False
	for t in range(max_time):
		if done or stop:
			samples[i_episode,0,t] = 0
			samples[i_episode,2,t] = -1
		else:
			action = agent.act(state)
			samples[i_episode,2,t] = action
			state, reward, done, _ = env.step(action)
			rewards += reward
			if np.random.random_sample() < 0:
				samples[i_episode,0,t*state_size:(t+1)*state_size] = 0
			else:
				samples[i_episode,0,t*state_size:(t+1)*state_size] = state
			if t % 100 == 0:
				stop = np.random.random_sample() < 0
print("bad reward {}".format(rewards/(demos//2)))
agent.load_weights(config.env, experment)

rewards = 0
for i_episode in range(demos//2, demos):
	state = env.reset()
	done = False
	stop = False
	for t in range(max_time):
		if done or stop:
			samples[i_episode,0,t] = 0
			samples[i_episode,2,t] = -1
		else:
			action = agent.act(state)
			samples[i_episode,2,t] = action
			state, reward, done, _ = env.step(action)
			rewards += reward
			if np.random.random_sample() < 0:
				samples[i_episode,0,t*state_size:(t+1)*state_size] = 0
			else:
				samples[i_episode,0,t*state_size:(t+1)*state_size] = state
			if t % 100 == 0:
				stop = np.random.random_sample() < 0
print("good reward {}".format(rewards/(demos//2)))
results = np.zeros((1000))
samples[demos//2:,1,0] = 1
np.random.shuffle(samples)
train, test = samples[:int(demos*0.8),:,:], samples[int(demos*0.8):,:,:]

missclass = pclass.update(train, test, config.env + experment)

agent = DDQNAgent(config,pclass)
experment = " DQN classifier oldest oops"
#trainer = Trainer(agent, env, config, experment, config.env)
#rewards, ind_rewards = trainer.train()
plt.figure(0)
fig, axs = plt.subplots(2,2, gridspec_kw={'hspace': 1/3, 'wspace':1/3})
fig.tight_layout()
axs[0,0].set_ylim(-300, 300)
axs[0,0].grid(b=True, which='major', color='#666666', linestyle='-')
axs[0,0].set_xlabel('Episode')
axs[0,0].set_ylabel('Running reward average')

axs[0,1].set_ylim(-300, 300)
axs[0,1].grid(b=True, which='major', color='#666666', linestyle='-')
axs[0,1].set_xlabel('Episode')
axs[0,1].set_ylabel('average reward last 20 epsoides')

#axs[1,0].set_ylim(-300, 300)
axs[1,0].grid(b=True, which='major', color='#666666', linestyle='-')
axs[1,0].set_xlabel("Episode")
axs[1,0].set_ylabel("Reward deleted")

axs[1,1].set_ylim(-300, 300)
axs[1,1].grid(b=True, which='major', color='#666666', linestyle='-')
axs[1,1].set_xlabel("Score")
axs[1,1].set_ylabel("Reward")
from mpl_toolkits.mplot3d import Axes3D
plt.figure(1)
fig2 = plt.subplots(1)
fig2[0].tight_layout()
ax2 = fig2[0].add_subplot(111, projection='3d')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward')
ax2.set_zlabel('Score')

fig3, axs3 = plt.subplots(1,2)
fig3.tight_layout()
axs3[0].set_xlabel("Score")
axs3[1].set_xlabel("Reward")

mean_rewards = []
std_rewards = []
for i in range(5):
	experment = "DQN classifier oldest oops {}".format(i)
	env = gym.make(config.env)
	config.action_dim = env.action_space.n
	config.state_dim = env.observation_space.shape[0]
	agent = DDQNAgent(config,pclass)

	trainer = Trainer(agent, env, config, experment, config.env)
	rewards, ind_rewards = trainer.train()
	p0 = axs[0,0].plot(np.arange(len(rewards)), rewards)
	with open("rewards{}".format(i) + experment, 'wb') as fp:
		pickle.dump(rewards, fp)




	p1 = axs[0,1].plot(np.arange(len(ind_rewards)), ind_rewards)
	with open("ind_rewards{}".format(i) + experment, 'wb') as fp:
		pickle.dump(ind_rewards, fp)

	# deleted vs epsoide
	deleted = agent.buffer.deleted
	p2 = axs[1,0].scatter(np.arange(len(deleted)), deleted, s = 1)

	with open("deleted{}".format(i) + experment, 'wb') as fp:
		pickle.dump(deleted, fp)

	# score vs reward
	reward_history = agent.buffer.reward_history
	score_history = agent.buffer.score_history
	p3 = axs[1,1].scatter(score_history, reward_history, s = 1)
	with open("reward_history{}".format(i) + experment, 'wb') as fp:
		pickle.dump(reward_history, fp)
	with open("score_history{}".format(i) + experment, 'wb') as fp:
		pickle.dump(score_history, fp)
	# epsoide vs reward vs score
	p4 = ax2.scatter(np.arange(len(reward_history)), reward_history, score_history, s = 1)

	p5 = axs3[0].hist(score_history, alpha=0.4)
	p5 = axs3[1].hist(reward_history, alpha=0.4)

	tester = Tester(agent, env, config.env, experment)
	mean, std = tester.test()
	mean_rewards.append(mean)
	std_rewards.append(std)

	#weight plots

# elif args.test:
# 	if args.model_path is None:
# 		print('please add the model path:', '--model_path xxxx')
# 		exit(0)
fig.savefig(config.env + " reward plot " + experment)
fig2[0].savefig(config.env + " 3D reward plot " + experment)
fig3.savefig(config.env + " hist " + experment)

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

# main_martix = np.zeros((128,128*4))
# i = 0
# for m in pclass.actor.modules():
# 	if isinstance(m, nn.Linear):
# 		w_martix = m.weight.data.cpu().detach().numpy()
# 		if w_martix.shape == (128 ,128):
# 			main_martix[:,i*128:(i+1)*128] = w_martix
# 			i += 1
# 		elif w_martix.shape == (128, 2400):
# 			axs3.imshow(w_martix)
# 		else:
# 			axs4.imshow(w_martix)
# axs5.imshow(main_martix)

# fig3.savefig(config.env + " weight martix inputs" + experment)
# fig4.savefig(config.env + " weight martix output" + experment)
# fig5.savefig(config.env + " weight martix network" + experment)