#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-23 上午6:38
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import random
import numpy as np


class KArmedBandits:

    """
    价值函数q_*(a)：给定行为a下获得的平均奖励——
    the true value of an action is the mean reward when that action is selected

    行为：在第t期的选择记为A_t

    实际中无法知道真实的价值函数，用估值函数Q(A_t)进行行为选择
    真实奖励：在第t期对行为A_t的奖励服从均值为q_*(A_t)，方差为1的正态分布

    有四种行为选择方案：
        1. 贪婪选择; 2. 近似贪婪选择; 3. 行为奖励的置信度上确界选择; 4. 期望行为偏好选择
        前三种都用到了行为价值的估计 ———— 均值估计; 后一种用到对行为偏好的期望，无需对行为价值估值
    有一种优化方案： 乐观的初始值
    有两种状态： 平稳状态和非平稳状态
    """

    def __init__(self,
                 k=10,
                 alpha=0.1,
                 epsilon=0,
                 confidence=1,
                 true_reward=0,
                 initial_value=0,
                 stationary=True,
                 strategy="greedy",
                 prefer_baseline=True):
        assert strategy in {"greedy", "ucb", "prefer"}
        assert k > 0 and epsilon >= 0 and confidence >= 0

        self._k = k
        self.__rounds = 0
        self._alpha = alpha
        self._epsilon = epsilon
        self._average_reward = 0
        self.__current_reward = 0
        self.__current_action = None
        self._strategy = strategy
        self._stationary = stationary,
        self._confidence = confidence
        self._true_reward = true_reward
        self._initial_value = initial_value
        self._prefer_baseline = prefer_baseline
        
        self._preference = np.zeros(k)
        self._action_prob = np.zeros(k)
        self._action_counter = np.zeros(k)
        self._value_estimate = np.zeros(k) + initial_value
        self._action_values = np.random.randn(k) + true_reward

        self.__best_action = self._action_values.argmax()
        
    def reset(self):
        """
        参数重新初始化
        :return:
        """
        self.__rounds = 0
        self._average_reward = 0
        self.__current_reward = 0
        self.__current_action = None
        self._preference = np.zeros(self._k)
        self._action_prob = np.zeros(self._k)
        self._action_counter = np.zeros(self._k)
        self._value_estimate = np.zeros(self._k) + self._initial_value
        self._action_values = np.random.randn(self._k) + self._true_reward

        self.__best_action = self._action_values.argmax()

    def __greedy_action(self):
        """
        如果_epsilon==0, 则为贪婪选择;
        否则为近似贪婪选择
        :return:
        """
        to_explore = random.random() < self._epsilon
        if to_explore:
            return random.randint(0, self._k-1)
        else:
            return self._value_estimate.argmax()

    def __ucb_action(self):
        """
        upper confidence bound
        行为奖励的置信度上确界选择
        :return:
        """
        estimate = (self._value_estimate +
                    self._confidence *
                    np.sqrt(np.divide(
                        np.log(self.__rounds+1),
                        self._action_counter+1e-5)))
        return estimate.argmax()

    def __grad_bandit_action(self):
        """
        行为偏好选择
        根据梯度更新期望行为偏好
        这里有个隐藏大BUG：按照行为概率分布取对应的行为，而不是取最大概率值的行为
        :return:
        """
        soft_max_deno = np.sum(np.exp(self._preference))
        self._action_prob = np.divide(np.exp(self._preference), soft_max_deno)
        return np.random.choice(range(self._k), p=self._action_prob)
        # return self._action_prob.argmax()

    def __get_reward(self, action):
        """
        即时真实奖励
        :param action: 选择的行为：{0 ～ k-1}
        :return: 
        """
        return np.random.randn() + self._action_values[action]

    def __update_state(self, action):
        """
        一次选择后的状态更新
        :param action: 选择的行为：{0 ～ k-1}
        :return: 
        """
        self.__rounds += 1
        self._action_counter[action] += 1
        step_size = (np.divide(1.0, self._action_counter[action])
                     if self._stationary else self._alpha)

        reward = self.__get_reward(action)

        self.__current_reward = reward
        self.__current_action = action

        self._average_reward = (
                (self.__rounds-1)/self.__rounds * self._average_reward +
                reward/self.__rounds)

        if self._strategy == "prefer":
            if self._prefer_baseline:
                baseline = self._average_reward
            else:
                baseline = 0
            one_hot = np.zeros(self._k)
            one_hot[action] = 1
            self._preference += self._alpha*(reward-baseline)*(one_hot-self._action_prob)
        else:
            self._value_estimate[action] += (
                    step_size * (reward - self._value_estimate[action]))

    def one_act(self):
        """
        一次行为选择
        :return:
        """
        # if self.__rounds == 0:
        #     action = random.randint(0, self._k-1)
        # else:
        if self._strategy == "greedy":
            action = self.__greedy_action()
        elif self._strategy == "ucb":
            action = self.__ucb_action()
        else:
            action = self.__grad_bandit_action()
        self.__update_state(action)

    def get_action_reward(self):
        return self.__current_action, self.__current_reward

    @property
    def best_action(self):
        return self.__best_action

    def get_action_value(self):
        return self._action_values
