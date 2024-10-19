import random

import numpy as np
import pandas as pd
import heapq

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_data():
    # 100个节点
    data_temp = pd.read_csv('数据集/data.csv', header=None)
    data = []
    for i in range(0, len(data_temp)):
        data.append((data_temp.iloc[i][0], data_temp.iloc[i][1]))

    # 节点间距离
    distance_temp = pd.read_csv('数据集/distance.csv', header=None)
    distance = distance_temp.values

    # 两节点之间是否为充电路段
    roads_temp = pd.read_csv('数据集/roads.csv', header=None)
    roads = roads_temp.values

    # 两个节点之间的行驶速度
    speed_temp = pd.read_csv('数据集/speed.csv', header=None)
    speed = speed_temp.values

    # 两个节点之间是否有路径 主对角线为0
    """
    [[0. 1. 0. ... 1. 1. 0.]    表示在0号节点可到达的位置为[0. 1. 0. ... 1. 1. 0.]，对应的可执行的动作为[1,...,97,98]
     [1. 0. 1. ... 1. 1. 1.]
     [0. 1. 0. ... 1. 1. 0.]
     ...
     [1. 1. 1. ... 0. 1. 1.]
     [1. 1. 1. ... 1. 0. 1.]
     [0. 1. 0. ... 1. 1. 0.]]
    """
    node_road = speed.copy()
    for i in range(len(node_road)):
        for j in range(len(node_road[i])):
            if node_road[i][j] > 0:
                node_road[i][j] = 1

    print(node_road)

    env_info = {
        'data': data,
        'distance': distance,
        'speed': speed,
        'roads': roads,
        'node_road': node_road
    }
    return env_info

def get_model_name():
    model_list = []
    for end in range(0, 100):
        for max_power in range(40, 101, 5):
            for consumption in range(10, 31,10):
                model_name = 'model_' + str(end) + '_' + str(max_power) + '_' + str(consumption)
                model_list.append(model_name)
    return model_list

def generate_random_car():
    start, end = random.sample(range(0, 100), 2)
    #终点
    init_power = random.randint(10, 50)
    max_power = random.randint(40, 100)
    dead_line = np.random.normal(2.1, 0.1)
    consumption = random.uniform(10, 30) / 100
    Evs = {
        'start': start,
        'end': end,
        'init_power': init_power,
        'max_power': max_power,
        'dead_line': dead_line,
        'consumption': consumption
    }
    return Evs

# 判断当前小车是否可以到达终点
def judge_arrive(EV, mapdata):
    start = EV['start']
    end = EV['end']
    init_power = EV['init_power']
    max_power = EV['max_power']
    deadline = EV['dead_line']
    consumption_rate = EV['consumption']  # 注意单位

    distance = mapdata['distance']
    speed = mapdata['speed']
    roads = mapdata['roads']  # 充电路段信息

    # 初始化变量
    current_power = init_power
    total_time = 0
    current_node = start
    charging_rate = 100

    visited = set()
    visited.add(current_node)

    while current_node != end:
        # 查找下一个可行的节点
        neighbors = np.where(mapdata['node_road'][current_node] == 1)[0]

        if len(neighbors) == 0:
            return False  # 没有可行路径

        # 寻找最短的距离
        min_distance = float('inf')
        next_node = None

        for neighbor in neighbors:
            if neighbor not in visited and distance[current_node][neighbor] < min_distance:
                next_node = neighbor
                min_distance = distance[current_node][neighbor]

        if next_node is None:
            print("没有路可以走")
            return False  # 没有可行路径

        # 计算到下一个节点的消耗
        travel_time = min_distance / speed[current_node][next_node]
        power_needed = consumption_rate * travel_time

        if current_power < power_needed:
            #print("电量不足")
            return False  # 电量不足

        if total_time + travel_time > deadline:
            #print("超过截止时间")
            return False  # 超过截止时间

        # 充电逻辑：如果是充电路段，计算充电量，同时考虑消耗和最大电量约束
        if roads[current_node][next_node] == 1:
            # 充电量为 充电时间（即travel_time） * 充电速率
            charging_power = charging_rate * travel_time
            # 新的电量为当前电量 + 充电量 - 消耗电量
            current_power = min(max_power, current_power + charging_power - power_needed)
        else:
            current_power -= power_needed  # 正常消耗电量

        # 更新状态
        total_time += travel_time
        current_node = next_node
        visited.add(current_node)

    # 如果成功到达终点
    return True

# 用于找到最佳路径
def find_max_power_path(EV, mapdata):
    start = EV['start']
    end = EV['end']
    init_power = EV['init_power']
    max_power = EV['max_power']
    deadline = EV['dead_line']
    consumption_rate = EV['consumption']  # 注意单位

    distance = mapdata['distance']
    speed = mapdata['speed']
    roads = mapdata['roads']  # 充电路段信息

    # 初始化优先队列，存储 (当前剩余电量, 当前节点, 剩余时间, 路径)
    pq = [(-init_power, start, deadline, [start])]

    # 用于记录到达每个节点时的最大剩余电量
    best_remaining_power = {start: init_power}

    while pq:
        # 取出当前剩余电量最大的路径
        current_power, current_node, remaining_time, path = heapq.heappop(pq)
        current_power = -current_power  # 转为正值

        # 如果已经到达终点，返回路径和最终剩余电量
        if current_node == end:
            return current_power, path,remaining_time

        # 查找下一个可行的节点
        neighbors = np.where(mapdata['node_road'][current_node] == 1)[0]

        for next_node in neighbors:
            travel_distance = distance[current_node][next_node]
            travel_time = travel_distance / speed[current_node][next_node]
            power_needed = consumption_rate * travel_distance

            # 判断是否可以到达下一个节点
            if current_power >= power_needed:
                new_remaining_time = remaining_time - travel_time
                if new_remaining_time < 0:
                    continue  # 剩余时间不足，跳过该路径

                # 充电逻辑：如果是充电路段，计算充电量，同时考虑消耗和最大电量约束
                if roads[current_node][next_node] == 1:
                    charging_power = 100 * travel_time  # 充电速率为100
                    next_power = min(max_power, current_power + charging_power - power_needed)
                else:
                    next_power = current_power - power_needed

                # 如果到达该节点的剩余电量更多，则更新并继续搜索
                if next_node not in best_remaining_power or next_power > best_remaining_power[next_node]:
                    best_remaining_power[next_node] = next_power
                    heapq.heappush(pq, (-next_power, next_node, new_remaining_time, path + [next_node]))

    # 如果没有找到路径，返回 None
    return None, []


if __name__ == "__main__":
    #生成训练集
    #先生成一辆
    ev_info = get_data()
    count = 0
    ev_list = []
    path_list = []
    while count < 10:
        ev = generate_random_car()
        if judge_arrive(ev,ev_info):
            ev_list.append(ev)
            count += 1
    for EV in ev_list:
        if find_optimal_path(EV,ev_info):
            path = find_optimal_path(EV,ev_info)
            path_list.append(path)
    print(len(path_list),path_list)