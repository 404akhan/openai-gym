"""
find at gym https://gym.openai.com/evaluations/eval_0QPN5rFmQkquX79w20PeYQ#reproducibility
deep q learning, 4 -> 40 (sigmoid) -> 40 (sigmoid) -> 2 Q(s, a)
experience replay size = 10000, batch size = 128, gamma = 0.99, learning rate 1e-3
adam optimizer makes things much faster
"""

import gym
from gym import wrappers
from collections import deque
import numpy as np
import random as random
import copy
import cPickle as pickle

np.random.seed(13)
random.seed(13)
N = 10000
D = deque([], N)
M = 100000
T = 600
bsize = 128
gamma = 0.99
lr = 1e-3
version = 1
eps = 1.
resume = False
render = False
allow_writing = True

in_size = 4
hl_size = 40
hl2_size = 40
out_size = 2

env = gym.make('CartPole-v1')

print N, M, T, bsize, gamma, lr, version, eps, resume, render, allow_writing, in_size, hl_size, hl2_size, out_size

if resume:
    model = pickle.load(open('dqn-cartpole-%d' % version, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(in_size, hl_size) / np.sqrt(in_size)
    model['W2'] = np.random.randn(hl_size, hl2_size) / np.sqrt(hl_size)
    model['W3'] = np.random.randn(hl2_size, out_size) / np.sqrt(hl2_size)
grad = {}
grad_sq = {}
for k, v in model.iteritems(): grad[k] = np.zeros_like(v)
for k, v in model.iteritems(): grad_sq[k] = np.zeros_like(v)

running_t = None

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def max_a_Q(state, model):
    states = np.array([state])
    hl = np.matmul(states, model['W1'])
    hl = sigmoid(hl)
    hl2 = np.matmul(hl, model['W2'])
    hl2 = sigmoid(hl2)
    out, = np.matmul(hl2, model['W3'])

    return np.argmax(out)

def max_val_Q(states, model):
    hl = np.matmul(states, model['W1'])
    hl = sigmoid(hl)
    hl2 = np.matmul(hl, model['W2'])
    hl2 = sigmoid(hl2)
    out = np.matmul(hl2, model['W3'])

    return np.max(out, axis=1)

def val_Q(states, model):
    hl = np.matmul(states, model['W1'])
    hl = sigmoid(hl)
    hl2 = np.matmul(hl, model['W2'])
    hl2 = sigmoid(hl2)
    out = np.matmul(hl2, model['W3'])

    return out, hl2, hl

def train(dout, hl2, hl, states1, model):
    global grad, grad_sq
    dhl2 = np.matmul(dout, model['W3'].transpose())
    dhl2 *= hl2 * (1 - hl2)
    dhl = np.matmul(dhl2, model['W2'].transpose())
    dhl *= hl * (1 - hl)
    d = {}
    d['W3'] = np.matmul(hl2.transpose(), dout)
    d['W2'] = np.matmul(hl.transpose(), dhl2)
    d['W1'] = np.matmul(states1.transpose(), dhl)

    for k in grad: grad[k] = grad[k] * 0.9 + d[k] * 0.1
    for k in grad_sq: grad_sq[k] = grad_sq[k] * 0.999 + (d[k]**2) * 0.001
    for k in model: model[k] -= lr * grad[k] / (np.sqrt(grad_sq[k]) + 1e-5)

for episode in range(1, M+1):
    state1 = env.reset()
    for t in range(1, T+1):
        if render: env.render()
        if np.random.random() < eps:
            action = np.random.randint(2)
        else:
            action = max_a_Q(state1, model)

        state2, reward, done, info = env.step(action)
        D.append([state1, action, reward, state2, done])
        state1 = state2

        if len(D) > bsize:
            batch = random.sample(list(D), bsize)
            D_array = np.array(batch)

            states1 = np.array([data[0] for data in D_array])
            actions = np.array([data[1] for data in D_array])
            rewards = np.array([data[2] for data in D_array])
            states2 = np.array([data[3] for data in D_array])
            dones = np.array([data[4] for data in D_array])

            second_term = gamma * max_val_Q(states2, model)
            second_term[dones] = 0
            y = rewards + second_term

            out, hl2, hl = val_Q(states1, model)
            correct = copy.deepcopy(out)
            correct[range(bsize), actions] = y

            dout = (out - correct) / bsize
            train(dout, hl2, hl, states1, model)

        if done or t == T:
            eps = min(eps, 1. / (1 + episode/10))
            running_t = (running_t * 0.9 + t * 0.1) if running_t != None else t
            if episode % 100 == 0 or render:
                print np.mean(model['W1'])
                print np.mean(model['W2'])
                print("Episode %d finished after %d timesteps, running aver %.2f" % (episode, t, running_t))
                if allow_writing: pickle.dump(model, open('dqn-cartpole-%d' % version, 'wb'))
            break

