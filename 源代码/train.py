import copy
import csv
import json
import sys
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import *  # 用于显示进度条

from env import EVs_Env
from DQN import DQN

from data import get_data, get_model_name, generate_random_car

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# def caculate_loss():
#     #生成多少个

def DQN_train(env, model_name):
    # 定义超参数
    max_episodes = 50000  # 训练episode数量
    max_steps = 500  # 每个回合的最大步数
    batch_size = 32  # 采样数量


    parts = model_name.split('_')
    # 提取出 end, max_power, consumption
    end = int(parts[1])  # 将字符串转换为整数
    max_power = int(parts[2])
    consumption = int(parts[3]) / 100

    # 创建DQN对象
    agent = DQN(env)

    total_rewards = []

    # 开始循环，tqdm用于显示进度条并评估任务时间开销
    for episode in tqdm(range(max_episodes), file=sys.stdout):
        # 重置环境并获取初始状态
        state, _ = env.refresh()
        env.end = end
        env.max_power = max_power
        env.consumption = consumption
        # 当前回合的奖励
        episode_reward = 0

        for step in range(max_steps):

            # 根据当前状态选择动作
            action = agent.choose_action(state)
            # 执行动作，获取新的信息
            next_state, reward, terminated, info = env.step(action)
            # 判断是否达到终止状态
            done = terminated

            # 将这个五元组加入到缓冲区中
            agent.replay_buffer.add(state, action, reward, next_state, done)
            # 累计奖励
            episode_reward += reward

            # 如果经验回放缓冲区已经有足够数据，就更新网络参数
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            # 更新当前状态
            state = next_state

            if done:
                break

        #记录当前回合奖励值
        total_rewards.append(episode_reward)

        # 打印中间值
        if episode % 40 == 0:
            tqdm.write("Episode " + str(episode) + ": " + str(episode_reward) + "======>" + str(env.path) + "======>" + "start:" + str(env.start) + ' ' + 'end:' + str(env.end))
    plt.plot(total_rewards)
    plt.show()
    model_path_name = f'./model/{model_name}.pth'
    torch.save(agent.model.state_dict(), model_path_name)



if __name__ == '__main__':
    Evs = generate_random_car()
    env_info = get_data()
    model_name = get_model_name()

    for model in model_name:
        # print(model)
        env = EVs_Env(Evs, env_info)
        env.reset()
        DQN_train(env, model)

    # sum_list = []
    # #训练50辆车
    # for _ in range(1):
    #     best_powerlist = []
    #     loss_list = []
    #     # for i in range(len(EVs)):
    #     test_list = [1,2]
    #     for i in  test_list:
    #         i -= 1
    #         env = EVs_Env(EVs[i], env_info)
    #         env.reset()
    #         best_power, episode_loss =DQN_train(env,i)
    #         best_powerlist.append(best_power)
    #
    #     sum_list.append(sum(best_powerlist))
    # print(sum_list)
