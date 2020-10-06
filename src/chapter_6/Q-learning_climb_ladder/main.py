#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Q-learning 爬楼梯
@author cqcet ailab
"""

import numpy as np
import pandas as pd
import os

# 梯子数组，TOP表示达到了目标
LADDER_ARR = ['-', '-', '-', '-', '-', '-', 'TOP']
# 最多实验的次数
MAX_EPISODE = 10
# 行为：向下爬
BEHAVIOR_DOWN = 'DOWN'
# 行为：向上爬
BEHAVIOR_UP = 'UP'
# 行为列表
BEHAVIORS = [BEHAVIOR_DOWN, BEHAVIOR_UP]
# 随机行为发生的概率
EPSILON = 0.1
# 学习率，这个值越高，表示越会参考之前获得的经验
ALPHA = 0.2
# 价值衰减率，表示对未来预测行为的参考程度
GAMMA = 0.9


def setup_q_table(states, behaviors):
    """
    初始化Q表
    :param states: 状态列表
    :param behaviors: 行为列表
    :return: 初始化的Q表
    """
    q_table = pd.DataFrame(
        np.zeros((states, len(behaviors))),
        columns=behaviors,
    )
    return q_table


def choose_next_behavior(state, q_table):
    state_behavior = q_table.iloc[state, :]
    state_behavior_random = (state_behavior == 0).all()
    epsilon_random = np.random.uniform() < EPSILON
    random_choose = state_behavior_random or epsilon_random < EPSILON
    if random_choose:
        next_behavior = np.random.choice(BEHAVIORS)
    else:
        next_behavior = state_behavior.idxmax()
    print('state is {}, state_actions is \r{}\rfinal choose next behavior {}(state_action_random {} epsilon_random {})'.
          format(state, state_behavior, next_behavior, state_behavior_random, epsilon_random))
    return next_behavior


def do_climb_behavior(state, behavior):
    if behavior == BEHAVIOR_DOWN:
        if LADDER_ARR[state] == LADDER_ARR[0]:
            next_state = state
        else:
            next_state = state - 1
    elif behavior == BEHAVIOR_UP:
        if LADDER_ARR[state] == LADDER_ARR[-1]:
            next_state = state
        else:
            next_state = state + 1
    reward = 1 if LADDER_ARR[next_state] == LADDER_ARR[-1] else 0
    return next_state, reward


def process_climb_ladder(q_table, episode_index):
    cur_state = 0
    step_counter = 0
    is_reach_top = False
    print_ladder_state(cur_state)
    while not is_reach_top:
        print('-----episode:{} step:{}-----------'.format(episode_index, step_counter))
        next_behavior = choose_next_behavior(cur_state, q_table)
        next_state, reward = do_climb_behavior(cur_state, next_behavior)
        # 在当前状态，选取下一个行为的奖励值
        q_predict_reward = q_table.loc[cur_state, next_behavior]
        is_reach_top = LADDER_ARR[next_state] == LADDER_ARR[-1]
        if is_reach_top:
            # 如果已经达到了目标，那么就是全额获得reward
            q_target_reward = reward
        else:
            # 如果还没有达到目标，在这个例子中 reward 为 0，接着取将要进行到的状态中，取出所有行为最大的奖励，乘以一个预测衰减
            q_target_reward = reward + GAMMA * q_table.iloc[next_state, :].max()
        # 更新当前状态的Q表信息，为当前值，再加上 预测状态奖励减去当前的奖励乘以一个学习率
        q_table.loc[cur_state, next_behavior] += ALPHA * (q_target_reward - q_predict_reward)

        cur_state = next_state
        step_counter += 1
        print('cur ladder state:')
        print_ladder_state(cur_state)
        print('cur q_table:')
        print(q_table)
        print('--------------step end-----------')
    return q_table


def print_ladder_state(cur_state):
    for index, element in reversed(list(enumerate(LADDER_ARR))):
        print('{:2} {:2}'.format(element, 'O' if index == cur_state else ' '))


def reinforcement_q_learning_main():
    q_table = setup_q_table(len(LADDER_ARR), BEHAVIORS)
    for episode_index in range(MAX_EPISODE):
        print('-------episode:{} start----------'.format(episode_index))
        q_table = process_climb_ladder(q_table, episode_index)
        print('-------episode:{} end------------'.format(episode_index))
        print(q_table)
        print('--------------------------------')
    return q_table


if __name__ == "__main__":
    q_table = reinforcement_q_learning_main()
    print('-------final----------')
    print(q_table)
    print('----------------------')
