#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-15 下午11:55
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from chapter2MultiArmedBandits.k_armed_bandits import KArmedBandits

matplotlib.use('Agg')

"""
如果出现 No module named '_bz2' 错误，是python3中没有装bz2的库导致的，
需根据各自的系统下载这个库包(如在ubuntu上：apt-get install bzip2-devel)，
然后重新编译你的python:
「你下载的python3的解压后文件路径」/configure --prefix=「你的python3安装路径」
make && make install (小心用户权限问题)
"""


def simulate(runs, time, bandits):
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(time):
                bandit.one_act()
                action, reward = bandit.get_action_reward()
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    best_action_counts = best_action_counts.mean(axis=1)
    rewards = rewards.mean(axis=1)
    return best_action_counts, rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('./images/figure_2_1.png')
    plt.close()


def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [KArmedBandits(epsilon=eps) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % eps)
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % eps)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./images/figure_2_2.png')
    plt.close()


def figure_2_3(runs=2000, time=1000):
    bandits = list()
    bandits.append(KArmedBandits(epsilon=0, initial_value=5))
    bandits.append(KArmedBandits(epsilon=0.1, initial_value=0))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./images/figure_2_3.png')
    plt.close()


def figure_2_4(runs=2000, time=1000):
    bandits = list()
    bandits.append(KArmedBandits(epsilon=0, strategy="ucb", confidence=2))
    bandits.append(KArmedBandits(epsilon=0.1))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB confidence = 2')
    plt.plot(average_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('./images/figure_2_4.png')
    plt.close()


def figure_2_5(runs=2000, time=1000):
    bandits = list()
    bandits.append(KArmedBandits(
        strategy="prefer", alpha=0.1, true_reward=4))
    bandits.append(KArmedBandits(
        strategy="prefer", alpha=0.1, prefer_baseline=False, true_reward=4))
    bandits.append(KArmedBandits(
        strategy="prefer", alpha=0.4, true_reward=4))
    bandits.append(KArmedBandits(
        strategy="prefer", alpha=0.4, prefer_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(0, len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('./images/figure_2_5.png')
    plt.close()


def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon greedy', 'bandit preference',
              'upper confident', 'optimistic initialization']
    generators = [lambda epsilon: KArmedBandits(epsilon=epsilon),
                  lambda alpha: KArmedBandits(strategy="prefer", alpha=alpha),
                  lambda conf: KArmedBandits(strategy="ucb", confidence=conf),
                  lambda initial: KArmedBandits(epsilon=0, initial_value=initial)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i + l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('./images/figure_2_6.png')
    plt.close()


if __name__ == '__main__':
    # figure_2_1()
    # figure_2_2()
    # figure_2_3()
    # figure_2_4()
    # figure_2_5()
    figure_2_6()
