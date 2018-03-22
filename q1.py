__author__ = 'deepak'

import numpy as np
import matplotlib.pyplot as plt
# Globals
mu = 0
sig = 1
itertions = 1000
problems = 2000

def plot_graph(x, y, label, axis, xlabel, ylabel):
    for item,lbl in zip(y,label):
        plt.plot(range(itertions), item, label=lbl)
    plt.axis(axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=4)
    plt.title('Reward')
    plt.show()

def get_rand_action():
    return np.random.randint(10)

def run_bandit(prob_value, q_star):
    Q_est = np.zeros(10)
    a_track = np.zeros(10)
    reward_his = np.zeros(itertions)
    #q_star = np.random.normal(mu, sig, (10))
    q_optimal = np.argmax(q_star)
    opt_track = np.zeros(itertions)
    for i in range(itertions):
        eps = np.random.uniform(0,1)
        if i==0 or eps < prob_value:
            action = get_rand_action()
        else:
            action = np.argmax(Q_est)
        if action == q_optimal:
            opt_track[i] = 1
        reward = np.random.normal(q_star[action], sig)
        a_track[action] += 1
        Q_est[action] = Q_est[action] + (1/float(a_track[action]))*(reward-Q_est[action])
        reward_his[i] += reward
    return reward_his, opt_track

reward_g = np.zeros(itertions)
reward_e1 = np.zeros(itertions)
reward_e2 = np.zeros(itertions)
optimal_act = np.zeros((3, itertions))
for ite in range(problems):
    q_star = np.random.normal(mu, sig, (10))
    re_g, opt_g = run_bandit(0, q_star)
    re_e1, opt_e1 = run_bandit(.1, q_star)
    re_e2, opt_e2 = run_bandit(.01, q_star)
    reward_g += re_g
    reward_e1 += re_e1
    reward_e2 += re_e2
    optimal_act[0] += opt_g
    optimal_act[1] += opt_e1
    optimal_act[2] += opt_e2

reward_g = reward_g/float(problems)
reward_e1 = reward_e1/float(problems)
reward_e2 = reward_e2/float(problems)

optimal_act = optimal_act/float(20)

plot_graph(itertions, [reward_g,reward_e1,reward_e2],['greedy','eps=.10','eps=.01'], [-10, itertions, 0, 2.5], 'Step', 'Average Reward')
plot_graph(itertions, optimal_act, ['greedy','eps=.10','eps=.01'], [-10, itertions, 0, 100], 'Step', '% Optimal Action')