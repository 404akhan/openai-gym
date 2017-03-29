# find at gym https://gym.openai.com/evaluations/eval_Gq3PuBNkSLSIUx7GKYQKoA#reproducibility
# implementation similar to https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# policy gradient method

import numpy as np
import gym
import cPickle as pickle

hl_size = 60
batch_size = 20
learning_rate = 1e-2
gamma = 0.99
resume = False
render = False
version = 1

in_size = 8
out_size = 4
if resume:
    model = pickle.load(open('save%d.p' % version, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(in_size, hl_size) / np.sqrt(in_size)
    model['W2'] = np.random.randn(hl_size, out_size) / np.sqrt(hl_size)

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
g1_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}
g2_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.matmul(x, model['W1'])
    h[h <= 0] = 0
    logp = np.matmul(h, model['W2'])
    p = np.exp(logp)
    p /= np.sum(p)
    return p, h

def choose_action(p):
    a0, a1, a2, a3 = p[0], p[0] + p[1], p[0] + p[1] + p[2], p[0] + p[1] + p[2] + p[3]
    rem1 = np.random.uniform()
    if rem1 < a0:
        return 0
    elif rem1 < a1:
        return 1
    elif rem1 < a2:
        return 2
    else:
        return 3

def policy_backward(eph, epdlogp):
    dW2 = np.matmul(eph.transpose(), epdlogp)
    dh = np.matmul(epdlogp, model['W2'].transpose())
    dh[eph <= 0] = 0
    dW1 = np.matmul(epx.transpose(), dh)
    return {'W1': dW1, 'W2': dW2}

env = gym.make('LunarLander-v2')
observation = env.reset()
xs, hs, dlogps, rs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
arr_rewards_plt = []

while episode_number < 100000:
    if render: env.render()

    x = observation
    prob, h = policy_forward(x)
    action = choose_action(prob)

    xs.append(x)
    hs.append(h)
    y = np.array([0, 0, 0, 0])
    y[action] = 1
    dlogps.append(y - prob)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    rs.append(reward)

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(rs)
        xs, hs, dlogps, rs = [], [], [], []

        discounted_epr = discount_rewards(epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]
                g1_cache[k] = 0.9 * g1_cache[k] + 0.1 * g
                g2_cache[k] = 0.999 * g2_cache[k] + 0.001 * g**2
                model[k] += learning_rate * g1_cache[k] / (np.sqrt(g2_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        arr_rewards_plt.append(reward_sum)
        if episode_number % 10 == 0:
            print 'episode: %d, reward total was: %f, running mean: %f' % (episode_number, reward_sum, running_reward)
            pickle.dump(model, open('save%d.p' % version, 'wb'))
            pickle.dump(arr_rewards_plt, open('arr_rewards_plt%d.p' % version, 'wb'))
        reward_sum = 0
        observation = env.reset()