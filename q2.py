__author__ = 'deepak'

import numpy as np
import matplotlib.pyplot as plt
from cdf import get_cdf
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
    plt.legend()
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

def softmax_selection(taou):
    Q_est = np.zeros(10)
    a_track = np.zeros(10)
    reward_his = np.zeros(itertions)
    q_optimal = np.argmax(q_star)
    opt_track = np.zeros(itertions)
    for i in range(itertions):
        sfmx_vec = Q_est.copy()
        sfmx_vec /= float(taou)
        e_sfmx = sfmx_vec.copy()
        e_sfmx = np.exp(e_sfmx)
        e_sfmx = e_sfmx/(np.sum(e_sfmx))
        if i==0 :
            action = get_rand_action()
        else:
            action = get_cdf(e_sfmx)
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
reward_sm = np.zeros((4,itertions))
optimal_act = np.zeros((5, itertions))
for ite in range(problems):
    q_star = np.random.normal(mu, sig, (10))
    re_g, opt_g = run_bandit(0, q_star)
    # re_e1, opt_e1 = run_bandit(.1, q_star)
    # re_e2, opt_e2 = run_bandit(.01, q_star)
    reward_g += re_g
    # reward_e1 += re_e1
    # reward_e2 += re_e2

    re_sm1,optsm1 = softmax_selection(.01)
    re_sm2,optsm2 = softmax_selection(.1)
    re_sm3,optsm3 = softmax_selection(10)
    re_sm4,optsm4 = softmax_selection(10000)

    reward_sm[0] += re_sm1
    reward_sm[1] += re_sm2
    reward_sm[2] += re_sm3
    reward_sm[3] += re_sm4

    optimal_act[0] += opt_g
    optimal_act[1] += optsm1
    optimal_act[2] += optsm2
    optimal_act[3] += optsm3
    optimal_act[4] += optsm4

reward_g = reward_g/float(problems)
reward_e1 = reward_e1/float(problems)
reward_e2 = reward_e2/float(problems)
reward_sm = reward_sm/float(problems)
optimal_act = optimal_act/float(20)
#plot_graph(itertions, [reward_g],['greedy','eps=.10','eps=.01'], [-10, itertions, 0, 2.5], 'Step', 'Average Reward')
#plot_graph(itertions, [reward_g, reward_sm[0], reward_sm[1], reward_sm[2], reward_sm[3]], ['Greedy','T=.01', 'T=.1', 'T=10', 'T=10000'], [-10, itertions, -4, 1.5], 'Step', 'Average Reward')
plot_graph(itertions, optimal_act, ['Greedy','T=.01', 'T=.1', 'T=10', 'T=10000'], [-10, itertions, 0, 100], 'Step', '% Optimal Action')