#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-23 上午6:38
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import numpy as np
import random


class SimpleBandit:

    """
    价值函数q_*(a)：给定行为a下获得的平均奖励——
    the true value of an action is the mean reward when that action is selected

    行为：在第t期的选择记为A_t

    实际中无法知道真实的价值函数，用估值函数Q(A_t)进行行为选择
    真实奖励：在第t期对行为A_t的奖励服从均值为q_*(A_t)，方差为1的正态分布
    """

    def __init__(self, k=10, epsilon=0):
        self.__k = k
        self.__epsilon = epsilon

        self.__action_counter = dict(zip(range(1, k+1), [0]*k))
        self.__value_estimate = dict(zip(range(1, k+1), [0]*k))

    def random_choice(self):
        return random.randint(1, self.__k)

    def get_reward(self, action):
        return random.normalvariate(self.__value_estimate[action], 1)

    def one_round(self):
        to_explore = random.random() < self.__epsilon
        if to_explore:
            action = self.random_choice()
        else:
            action = max(self.__value_estimate, key=self.__value_estimate.get)
        reward = self.get_reward(action)
        estimate = self.__value_estimate[action]

        self.__action_counter[action] += 1
        self.__value_estimate[action] += (reward-estimate)/self.__action_counter[action]

