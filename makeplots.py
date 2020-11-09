import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
fig, axs = plt.subplots(2,figsize = (10,8))
axs[0].set_ylim(-300, 300)
axs[1].set_ylim(-300, 300)
axs[0].set_ylabel("Runing average reward")
axs[0].set_xlabel("Epsoide")
axs[1].set_ylabel("Raw reward")
axs[1].set_xlabel("Epsoide")
fig.suptitle("Learning progress", y = 1)
fig.tight_layout()
for i in range(5):
	experment = "DQN baseline 2 {}".format(i)

	with open("rewards{}".format(i) + experment, 'rb') as fp:
		rewards = pickle.load(fp)

	with open("ind_rewards{}".format(i) + experment, 'rb') as fp:
		ind_rewards = pickle.load(fp)

	p1 = axs[0].plot(np.arange(len(rewards)), rewards)
	p1 = axs[1].plot(np.arange(len(ind_rewards)), ind_rewards)
plt.show()

with open("mean_rewards{}" + experment, 'rb') as fp:
	mean_rewards_base = pickle.load(fp)
with open("std_rewards{}" + experment, 'rb') as fp:
	std_rewards_base = pickle.load(fp)
std_rewards_base = [x/math.sqrt(200) for x in std_rewards_base]
fig4, ax4 = plt.subplots(1, figsize = (10,8))

ax4.set_xlabel('Run')
ax4.set_ylabel('Reward')
ax4.set_title("Baseline results from 5 runs with standard deviation")
ax4.bar(np.arange(len(mean_rewards_base)), mean_rewards_base, yerr=std_rewards_base, align='center', alpha=0.5)
ax4.legend(["median score {}".format(np.median(mean_rewards_base))])
#fig4.savefig(config.env + " reward " + experment)
fig4.tight_layout()
plt.show()

fig, axs = plt.subplots(2,2, gridspec_kw={'hspace': 1/3, 'wspace':1/3},figsize = (10,8))
fig.suptitle("DQL + classifier Learning progress")
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
axs[1,0].set_ylabel("Score")

axs[1,1].set_ylim(-300, 300)
axs[1,1].grid(b=True, which='major', color='#666666', linestyle='-')
axs[1,1].set_xlabel("Score")
axs[1,1].set_ylabel("Reward")
fig4.tight_layout()

fig3, axs3 = plt.subplots(1,2,figsize = (10,8))
fig3.suptitle("DQL + classifier Distribution of epsoide scores and rewards over 5 runs", y=1)
axs3[0].set_xlabel("Score")
axs3[1].set_xlabel("Reward")
axs3[0].set_ylabel("Amount")
axs3[1].set_ylabel("Amount")
fig3.tight_layout()
for i in range(5):
	experment = "DQN classifier oldest oops {}".format(i)

	with open("rewards{}".format(i) + experment, 'rb') as fp:
		rewards = pickle.load(fp)

	with open("ind_rewards{}".format(i) + experment, 'rb') as fp:
		ind_rewards = pickle.load(fp)



	with open("deleted{}".format(i) + experment, 'rb') as fp:
		deleted = pickle.load(fp)
	with open("reward_history{}".format(i) + experment, 'rb') as fp:
		reward_history = pickle.load(fp)
	with open("score_history{}".format(i) + experment, 'rb') as fp:
		score_history = pickle.load(fp)

	p0 = axs[0,0].plot(np.arange(len(rewards)), rewards)
	p1 = axs[0,1].plot(np.arange(len(ind_rewards)), ind_rewards)
	p2 = axs[1,0].scatter(np.arange(len(score_history)), score_history, s = 0.5)
	p3 = axs[1,1].scatter(score_history, ind_rewards, s = 0.5)

	p5 = axs3[0].hist(score_history, alpha=0.4)
	p5 = axs3[1].hist(ind_rewards, alpha=0.4)

plt.show()
with open("mean_rewards{}" + experment, 'rb') as fp:
	mean_rewards = pickle.load(fp)
with open("std_rewards{}" + experment, 'rb') as fp:
	std_rewards = pickle.load(fp)
std_rewards = [x/math.sqrt(200) for x in std_rewards]


fig4, ax4 = plt.subplots(1, figsize = (10,8))

ax4.set_xlabel('Run')
ax4.set_ylabel('Reward')
ax4.set_title("DQL + classifier results from 5 runs with Standard deviation")
width = 0.35
ax4.bar(np.arange(len(mean_rewards_base)), mean_rewards_base, width, yerr=std_rewards_base, align='center', alpha=0.5)
ax4.bar(np.arange(len(mean_rewards)) + width, mean_rewards, width, yerr=std_rewards, align='center', alpha=0.5)
ax4.legend(["Baseline (Median score {})".format(np.median(mean_rewards_base)),"Median score {}".format(np.median(mean_rewards))])
fig4.tight_layout()
#fig4.savefig(config.env + " reward " + experment)
plt.show()




fig, axs = plt.subplots(2,2, gridspec_kw={'hspace': 1/3, 'wspace':1/3},figsize = (10,8))
fig.suptitle("DQL + classifier with dual sub-optimal policies Learning progress")
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
axs[1,0].set_ylabel("Score")

axs[1,1].set_ylim(-300, 300)
axs[1,1].grid(b=True, which='major', color='#666666', linestyle='-')
axs[1,1].set_xlabel("Score")
axs[1,1].set_ylabel("Reward")
fig.tight_layout()

fig3, axs3 = plt.subplots(1,2,figsize = (10,8))
fig3.suptitle("DQL + classifier with dual sub-optimal policies Distribution of epsoide scores and rewards over 5 runs", y=1)
axs3[0].set_xlabel("Score")
axs3[1].set_xlabel("Reward")
axs3[0].set_ylabel("Amount")
axs3[1].set_ylabel("Amount")
fig3.tight_layout()
for i in range(5):
	experment = "DQN classifier 5050 oldest {}".format(i)

	with open("rewards{}".format(i) + experment, 'rb') as fp:
		rewards = pickle.load(fp)

	with open("ind_rewards{}".format(i) + experment, 'rb') as fp:
		ind_rewards = pickle.load(fp)



	with open("deleted{}".format(i) + experment, 'rb') as fp:
		deleted = pickle.load(fp)

	with open("reward_history{}".format(i) + experment, 'rb') as fp:
		reward_history = pickle.load(fp)
	with open("score_history{}".format(i) + experment, 'rb') as fp:
		score_history = pickle.load(fp)
	p0 = axs[0,0].plot(np.arange(len(rewards)), rewards)
	p1 = axs[0,1].plot(np.arange(len(ind_rewards)), ind_rewards)
	p2 = axs[1,0].scatter(np.arange(len(score_history)), score_history, s = 0.5)
	p3 = axs[1,1].scatter(score_history, ind_rewards, s = 0.5)

	p5 = axs3[0].hist(score_history, alpha=0.4)
	p5 = axs3[1].hist(ind_rewards, alpha=0.4)

plt.show()

with open("mean_rewards{}" + experment, 'rb') as fp:
	mean_rewards = pickle.load(fp)
with open("std_rewards{}" + experment, 'rb') as fp:
	std_rewards = pickle.load(fp)
std_rewards = [x/math.sqrt(200) for x in std_rewards]

fig4, ax4 = plt.subplots(1, figsize = (10,8))

ax4.set_xlabel('Run')
ax4.set_ylabel('Reward')
ax4.set_title("DQL + classifier with dual sub-optimal policies results from 5 runs with Standard deviation")
width = 0.35
ax4.bar(np.arange(len(mean_rewards_base)), mean_rewards_base, width, yerr=std_rewards_base, align='center', alpha=0.5)
ax4.bar(np.arange(len(mean_rewards)) + width, mean_rewards, width, yerr=std_rewards, align='center', alpha=0.5)
ax4.legend(["Baseline (Median score {})".format(np.median(mean_rewards_base)),"Median score {}".format(np.median(mean_rewards))])
fig4.tight_layout()
#fig4.savefig(config.env + " reward " + experment)
plt.show()




fig, axs = plt.subplots(2,2, gridspec_kw={'hspace': 1/3, 'wspace':1/3}, figsize = (10,8))
fig.suptitle("DQL + raw classifier scores Learning progress")
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
axs[1,0].set_ylabel("Score")

axs[1,1].set_ylim(-300, 300)
axs[1,1].grid(b=True, which='major', color='#666666', linestyle='-')
axs[1,1].set_xlabel("Score")
axs[1,1].set_ylabel("Reward")
fig.tight_layout()

fig3, axs3 = plt.subplots(1,2, figsize = (10,8))
fig3.suptitle("DQL + raw classifier scores Distribution of epsoide scores and rewards over 5 runs", y=1)
axs3[0].set_xlabel("Score")
axs3[1].set_xlabel("Reward")
axs3[0].set_ylabel("Amount")
axs3[1].set_ylabel("Amount")
fig3.tight_layout()
for i in range(5):
	experment = "DQN classifier raw {}".format(i)

	with open("rewards{}".format(i) + experment, 'rb') as fp:
		rewards = pickle.load(fp)

	with open("ind_rewards{}".format(i) + experment, 'rb') as fp:
		ind_rewards = pickle.load(fp)



	with open("deleted{}".format(i) + experment, 'rb') as fp:
		deleted = pickle.load(fp)

	with open("reward_history{}".format(i) + experment, 'rb') as fp:
		reward_history = pickle.load(fp)
	with open("score_history{}".format(i) + experment, 'rb') as fp:
		score_history = pickle.load(fp)
	p0 = axs[0,0].plot(np.arange(len(rewards)), rewards)
	p1 = axs[0,1].plot(np.arange(len(ind_rewards)), ind_rewards)
	p2 = axs[1,0].scatter(np.arange(len(score_history)), score_history, s = 0.5)
	p3 = axs[1,1].scatter(score_history, ind_rewards, s = 0.5)

	p5 = axs3[0].hist(score_history, alpha=0.4)
	p5 = axs3[1].hist(ind_rewards, alpha=0.4)

plt.show()

with open("mean_rewards{}" + experment, 'rb') as fp:
	mean_rewards = pickle.load(fp)
with open("std_rewards{}" + experment, 'rb') as fp:
	std_rewards = pickle.load(fp)
std_rewards = [x/math.sqrt(200) for x in std_rewards]

fig4, ax4 = plt.subplots(1, figsize = (10,8))

ax4.set_xlabel('Run')
ax4.set_ylabel('Reward')
ax4.set_title("DQL + raw classifier scores results from 5 runs with Standard deviation")
width = 0.35
ax4.bar(np.arange(len(mean_rewards_base)), mean_rewards_base, width, yerr=std_rewards_base, align='center', alpha=0.5)
ax4.bar(np.arange(len(mean_rewards)) + width, mean_rewards, width, yerr=std_rewards, align='center', alpha=0.5)
ax4.legend(["Baseline (Median score {})".format(np.median(mean_rewards_base)),"Median score {}".format(np.median(mean_rewards))])
fig4.tight_layout()
#fig4.savefig(config.env + " reward " + experment)
plt.show()

