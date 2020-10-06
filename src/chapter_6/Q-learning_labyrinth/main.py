#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Q-learning 走迷宫
@author cqcet ailab
"""

import numpy as np
import pandas as pd
from enum import IntEnum


# 节点类型：普通，宝藏，陷阱
class MapNodeType(IntEnum):
    ground = 0
    treasure = 1
    trap = 2


# 动作类型
class BehaviorsType(IntEnum):
    up = 0
    right = 1
    down = 2
    left = 3


# 最多实验的次数
MAX_EPISODE = 50
# 随机行为发生的概率
EPSILON = 0.1
# 学习率，这个值越高，表示越会参考之前获得的经验
ALPHA = 0.2
# 价值衰减率，表示对未来预测行为的参考程度
GAMMA = 0.9


class Map_Node:
    def __init__(self, index, neighbor_nodes, node_type, node_flag):
        # 节点序号
        self.index = index
        # 邻居节点
        self.neighbor_nodes = neighbor_nodes
        # 节点类型
        self.node_type = node_type
        # 节点标记
        self.node_flag = node_flag

    def is_trap_node(self):
        return self.node_type == MapNodeType.trap

    def is_treasure_node(self):
        return self.node_type == MapNodeType.treasure

    def is_terminal(self):
        return self.is_trap_node() or self.is_treasure_node()


def setup_labyrinth_map():
    """
    ○(0)  -   ○(1)  -   ○(2)  -   ○(3)  -  ○(4)
      |         |          |         |         |
    ○(5)  -   ○(6)  -   ○(7)  -   ○(8)  -  ○(9)
      |         |          |         |         |
    ☼(10)  -  ○(11)  -  ○(12)  -  ☼(13)  - ○(14)
      |         |          |         |         |
    ○(15)  -  ○(16)  -  ☼(17)  -  ※(18)  - ○(19)
      |         |          |         |         |
    ○(20)  -  ○(21)  -  ○(22)  -  ○(23)  - ○(24)
    :return:
    """
    map_array = np.array([(0, 0, 0, 0, 0),
                          (0, 0, 0, 0, 0),
                          (2, 0, 0, 2, 0),
                          (0, 0, 2, 1, 0),
                          (0, 0, 0, 0, 0)
                          ])

    map_dic = dict()
    for row in range(map_array.shape[0]):
        for line in range(map_array.shape[1]):
            node_index = line + map_array.shape[1] * row
            map_dic[node_index] = Map_Node(node_index,
                                           [-1 if row - 1 < 0 else (line + map_array.shape[1] * (row - 1)),  # 上
                                            -1 if line + 1 >= map_array.shape[1] else (
                                                    (line + 1) + map_array.shape[1] * row),  # 右
                                            -1 if row + 1 >= map_array.shape[0] else (
                                                    line + map_array.shape[1] * (row + 1)),  # 下
                                            -1 if line - 1 < 0 else ((line - 1) + map_array.shape[1] * row)],  # 左
                                           map_array[row][line],
                                           '○' if map_array[row][line] == MapNodeType.ground else
                                           '※' if map_array[row][line] == MapNodeType.treasure else
                                           '☼')
    print_labyrinth_state(map_dic, 0)
    return map_dic


def print_labyrinth_state(labyrinth_map, cur_state):
    map_len = int(np.sqrt(len(labyrinth_map)))
    for h_index in range(0, map_len):
        for w_index in range(0, map_len):
            node_index = w_index + map_len * h_index
            print('{:<2}'.format('●' if node_index == cur_state else labyrinth_map[node_index].node_flag), end='')
            print('{:<2}'.format('-' if labyrinth_map[node_index].neighbor_nodes[1] != -1 else ''), end='')
        print('\r')
        for w_index in range(0, map_len):
            node_index = w_index + map_len * h_index
            print('{:<3}'.format(' '), end='') if w_index != 0 else print('', end='')
            print('{:<2}'.format('|' if labyrinth_map[node_index].neighbor_nodes[2] != -1 else ''), end='')
        print('')


def setup_q_table(states, behaviors):
    q_table = pd.DataFrame(
        np.zeros((states, len(behaviors))),
        columns=behaviors,
    )
    return q_table


def choose_next_behavior(labyrinth_map, q_table, node):
    state_behavior = q_table.loc[node, :]
    neighbor_nodes = labyrinth_map[node].neighbor_nodes
    options_dir = [index for index, node in enumerate(neighbor_nodes) if node != -1]
    random_choose = np.random.uniform() < EPSILON
    if random_choose:
        next_behavior = np.random.choice(options_dir)
    else:
        max_target_dirs = [x for x in options_dir if state_behavior[x] == np.max(state_behavior)]
        next_behavior = np.random.choice(max_target_dirs)
    next_node = labyrinth_map[neighbor_nodes[next_behavior]]
    reward = 1 if next_node.is_treasure_node() else -1 if next_node.is_trap_node() else 0
    return next_behavior, next_node, reward


def process_explore_labyrinth(labyrinth_map, q_table):
    node_list = [0]
    is_reach_end = False
    while not is_reach_end:
        cur_node = labyrinth_map[node_list[-1]]
        next_behavior, next_node, reward = choose_next_behavior(labyrinth_map, q_table, cur_node.index)
        is_reach_end = next_node.is_terminal()

        if next_node.index not in q_table.index:
            q_table = q_table.append(pd.Series([0] * len(BehaviorsType), index=q_table.columns, name=next_node.index))
        q_predict_reward = q_table.loc[cur_node.index, next_behavior]

        if is_reach_end:
            q_target_reward = reward
        else:
            q_target_reward = reward + GAMMA * q_table.loc[next_node.index, :].max()
        q_table.loc[cur_node.index, next_behavior] += ALPHA * (q_target_reward - q_predict_reward)
        node_list.append(next_node.index)
        print('step {} node path {}'.format(len(node_list), node_list))
        print(q_table)
        print('-----step end-----')
    return q_table, node_list


def reinforcement_q_learning_main():
    # 初始化地图
    labyrinth_map = setup_labyrinth_map()
    # 一开始只有自己的所在地块的q_table
    q_table = setup_q_table(1, BehaviorsType)
    # 信息记录
    explore_statics_dic = dict()
    for episode_index in range(MAX_EPISODE):
        print('-------episode:{} start----------'.format(episode_index))
        q_table, node_list = process_explore_labyrinth(labyrinth_map, q_table)
        print(node_list)
        print(q_table)
        explore_statics_dic[episode_index] = zip(node_list, q_table)
        print('-------episode:{} end------------'.format(episode_index))
    return q_table


if __name__ == "__main__":
    reinforcement_q_learning_main()
