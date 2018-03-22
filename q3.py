__author__ = 'deepak'

import numpy as np
import matplotlib.pyplot as plt
# Globals
mu = 0
sig = 1
itertions = 1000
problems = 2000
actions = 1000

def plot_graph(x, y, label):
    for item,lbl in zip(y,label):
        plt.plot(range(itertions), item, label=lbl)
    plt.axis([-10, itertions, 0, 2.5])
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.legend(loc=4)
    plt.title('Reward')
    plt.show()

def get_rand_action():
    return np.random.randint(actions)

def run_bandit(prob_value, q_star):
    Q_est = np.zeros(actions)
    a_track = np.zeros(actions)
    reward_his = np.zeros(itertions)
    for i in range(itertions):
        eps = np.random.uniform(0,1)
        if i==0 or eps < prob_value:
            action = get_rand_action()
        else:
            action = np.argmax(Q_est)
        reward = np.random.normal(q_star[action], sig)
        a_track[action] += 1
        Q_est[action] = Q_est[action] + (1/float(a_track[action]))*(reward-Q_est[action])
        reward_his[i] += reward
    return reward_his

def run_ucb(q_star):
    Q_est = np.zeros(actions)
    a_track = np.ones(actions)
    reward_his = np.zeros(itertions)
    for ite in range(actions):
        rew = np.random.normal(q_star[ite], sig)
        Q_est[ite] = rew
        reward_his[ite] = rew
    for i in range(actions,itertions):
        ucb_vec = Q_est.copy()
        for act in range(actions):
            ucb_vec[act] += np.sqrt((2*np.log(i))/float(a_track[act]))
        action = np.argmax(ucb_vec)
        reward = np.random.normal(q_star[action], sig)
        a_track[action] += 1
        Q_est[action] = Q_est[action] + (1/float(a_track[action]))*(reward-Q_est[action])
        reward_his[i] += reward
    return reward_his

reward_g = np.zeros(itertions)
reward_e1 = np.zeros(itertions)
reward_e2 = np.zeros(itertions)
reward_ucb = np.zeros(itertions)
for ite in range(problems):
    q_star = np.random.normal(mu, sig, (actions))
    reward_g += run_bandit(0, q_star)
    reward_e1 += run_bandit(.1, q_star)
    reward_e2 += run_bandit(.01, q_star)
    reward_ucb += run_ucb(q_star)

reward_g = reward_g/float(problems)
reward_e1 = reward_e1/float(problems)
reward_e2 = reward_e2/float(problems)
reward_ucb = reward_ucb/float(problems)

plot_graph(itertions, [reward_g,reward_e1,reward_e2, reward_ucb],['greedy','eps=.10','eps=.01', 'ucb'])
