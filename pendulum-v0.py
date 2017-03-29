"""
find at gym https://gym.openai.com/evaluations/eval_HVeym7XSgqvUVKNcLSIQ#reproducibility
task - continuous control
method - deep deterministic policy gradient
https://arxiv.org/abs/1509.02971
"""

import numpy as np
import gym
import copy
from collections import deque
import random
import cPickle as pickle

bsize = 32
state_size = 3
hl1_size = 100
hl2_size = 80
action_size = 1
replay_size = 100*1000
episodes_num = 100*1000
iters_num = 1000
gamma = 0.99
upd_r = 0.01
lr_actor = 1e-4
lr_critic = 1e-3
seed = 13
np.random.seed(seed)
random.seed(seed)
running_r = None
arr_values_rewards = []
version = 2
demo = False
resume = demo
render = demo
allow_writing = not demo
decay = 1e-2

print bsize, hl1_size, hl2_size, replay_size, gamma, upd_r, lr_actor, lr_critic, seed, version, demo

if resume:
    Q = pickle.load(open('Q-pendulum-%d' % version, 'rb'))
    Miu = pickle.load(open('Miu-pendulum-%d' % version, 'rb'))
else:
    Q = {}
    Q['W1'] = np.random.uniform(-1., 1., (state_size, hl1_size)) / np.sqrt(state_size)
    Q['W2'] = np.random.uniform(-1., 1., (action_size+hl1_size, hl2_size)) / np.sqrt(action_size+hl1_size)
    Q['W3'] = np.random.uniform(-3*1e-4, 3*1e-4, (hl2_size, 1)) / np.sqrt(hl2_size)

    Miu = {}
    Miu['W1'] = np.random.uniform(-1., 1., (state_size, hl1_size)) / np.sqrt(state_size)
    Miu['W2'] = np.random.uniform(-1., 1., (hl1_size, hl2_size)) / np.sqrt(hl1_size)
    Miu['W3'] = np.random.uniform(-3*1e-3, 3*1e-3, (hl2_size, action_size)) / np.sqrt(hl2_size)

Q_tar = copy.deepcopy(Q)
Miu_tar = copy.deepcopy(Miu)

Qgrad = { k : np.zeros_like(v) for k,v in Q.iteritems() }
Qgrad_sq = { k : np.zeros_like(v) for k,v in Q.iteritems() }
Miugrad = { k : np.zeros_like(v) for k,v in Miu.iteritems() }
Miugrad_sq = { k : np.zeros_like(v) for k,v in Miu.iteritems() }

R = deque([], replay_size)
env = gym.make('Pendulum-v0')

def sample_batch(R, bsize):
    batch = random.sample(list(R), bsize)
    D_array = np.array(batch)

    states1 = np.array([data[0] for data in D_array])
    actions1 = np.array([data[1] for data in D_array])
    rewards = np.array([[data[2]] for data in D_array])
    states2 = np.array([data[3] for data in D_array])
    dones = np.array([data[4] for data in D_array])

    return states1, actions1, rewards, states2, dones

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    e1 = np.exp(x)
    e2 = np.exp(-x)
    return (e1 - e2) / (e1 + e2)

def actions_Miu(states, Miu):
    hl1 = np.matmul(states, Miu['W1'])
    hl1 = relu(hl1)
    hl2 = np.matmul(hl1, Miu['W2'])
    hl2 = relu(hl2)
    outs = np.matmul(hl2, Miu['W3'])
    actions = 2 * tanh(outs)

    return actions

def values_Q(states, actions, Q):
    hl1 = np.matmul(states, Q['W1'])
    hl1 = relu(hl1)
    actions_hl1 = np.concatenate([actions, hl1], axis=1)
    hl2 = np.matmul(actions_hl1, Q['W2'])
    hl2 = relu(hl2)
    values = np.matmul(hl2, Q['W3'])

    return values, hl2, hl1, actions_hl1

def train_Q(douts, hl2, hl1, actions_hl1, states, Q):
    dhl2 = np.matmul(douts, Q['W3'].transpose())
    dhl2[hl2 <= 0] = 0
    dactions_hl1 = np.matmul(dhl2, Q['W2'].transpose())
    dhl1 = dactions_hl1[:, action_size:]
    dhl1[hl1 <= 0] = 0
    d = {}
    d['W3'] = np.matmul(hl2.transpose(), douts) + decay * Q['W3']
    d['W2'] = np.matmul(actions_hl1.transpose(), dhl2) + decay * Q['W2']
    d['W1'] = np.matmul(states.transpose(), dhl1) + decay * Q['W1']

    for k in Qgrad: Qgrad[k] = Qgrad[k] * 0.9 + d[k] * 0.1
    for k in Qgrad_sq: Qgrad_sq[k] = Qgrad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in Q: Q[k] -= lr_critic * Qgrad[k] / (np.sqrt(Qgrad_sq[k]) + 1e-5)

def train_Miu(states, Miu, Q):
    mhl1 = np.matmul(states, Miu['W1'])
    mhl1 = relu(mhl1)
    mhl2 = np.matmul(mhl1, Miu['W2'])
    mhl2 = relu(mhl2)
    outs = np.matmul(mhl2, Miu['W3'])
    actions = 2 * tanh(outs)

    qhl1 = np.matmul(states, Q['W1'])
    qhl1 = relu(qhl1)
    actions_qhl1 = np.concatenate([actions, qhl1], axis=1)
    qhl2 = np.matmul(actions_qhl1, Q['W2'])
    qhl2 = relu(qhl2)

    dvalues = np.ones((bsize, 1))
    dqhl2 = np.matmul(dvalues, Q['W3'].transpose())
    dqhl2[qhl2 <= 0] = 0
    actions_qhl1 = np.matmul(dqhl2, Q['W2'].transpose())
    dactions = actions_qhl1[:, :action_size]
    dactions /= bsize

    douts = dactions * 2 * (1 + actions/2) * (1 - actions/2)
    dmhl2 = np.matmul(douts, Miu['W3'].transpose())
    dmhl2[mhl2 <= 0] = 0
    dmhl1 = np.matmul(dmhl2, Miu['W2'].transpose())
    dmhl1[mhl1 <= 0] = 0

    d = {}
    d['W3'] = np.matmul(mhl2.transpose(), douts)
    d['W2'] = np.matmul(mhl1.transpose(), dmhl2)
    d['W1'] = np.matmul(states.transpose(), dmhl1)

    for k in Miugrad: Miugrad[k] = Miugrad[k] * 0.9 + d[k] * 0.1
    for k in Miugrad_sq: Miugrad_sq[k] = Miugrad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in Miu: Miu[k] += lr_actor * Miugrad[k] / (np.sqrt(Miugrad_sq[k]) + 1e-5)

def noise(episode):
    if demo:
        return 0.
    if np.random.randint(2) == 0:
        return (1. / (1. + episode/10))
    else:
        return -(1. / (1. + episode/10))

def action_noise(action):
    action_n = action + noise(episode)
    action_n = np.maximum(action_n, -2)
    action_n = np.minimum(action_n, 2)

    return action_n

for episode in range(1, episodes_num+1):
    state1 = env.reset()
    ep_reward = 0.
    value, _, _, _ = values_Q([state1], actions_Miu([state1], Miu), Q)
    for iter in range(1, iters_num+1):
        if render: env.render()
        action = actions_Miu(state1, Miu)
        action = action_noise(action)
        state2, reward, done, _ = env.step(action)
        R.append([state1, action, reward, state2, done])
        ep_reward += reward
        state1 = state2

        if(len(R) > bsize) and not demo:
            states1, actions1, rewards, states2, dones = sample_batch(R, bsize)
            actions2 = actions_Miu(states2, Miu_tar)
            values, _, _, _ = values_Q(states2, actions2, Q_tar)
            second_term = gamma * values
            second_term[dones] = 0
            y = rewards + second_term

            outs, hl2, hl1, actions_hl1 = values_Q(states1, actions1, Q)
            douts = (outs - y) / bsize
            train_Q(douts, hl2, hl1, actions_hl1, states1, Q)
            train_Miu(states1, Miu, Q)

            for k, v in Q.iteritems(): Q_tar[k] = upd_r * v + (1-upd_r) * Q_tar[k]
            for k, v in Miu.iteritems(): Miu_tar[k] = upd_r * v + (1-upd_r) * Miu_tar[k]

        if done or iter == iters_num:
            running_r = (running_r * 0.9 + ep_reward * 0.1) if running_r != None else ep_reward
            arr_values_rewards.append([value, ep_reward])
            if episode % 1 == 0:
                print np.mean(Q['W1']), np.mean(Q['W2']), np.mean(Q['W3'])
                print np.mean(Miu['W1']), np.mean(Miu['W2']), np.mean(Miu['W3'])
                print 'ep: %d, iters: %d, reward %f, run aver: %f' % \
                      (episode, iter, ep_reward, running_r)
            if episode % 10 == 0 and allow_writing:
                pickle.dump(Q, open('Q-pendulum-%d' % version, 'wb'))
                pickle.dump(Miu, open('Miu-pendulum-%d' % version, 'wb'))
                pickle.dump(arr_values_rewards, open('VR-pendulum-%d' % version, 'wb'))
            break
