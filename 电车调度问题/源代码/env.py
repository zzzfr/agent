import gym
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
from gym import spaces
import math

class GymHelper:
    def __init__(self, env, figsize=(3, 3)):
        self.env = env  # 初始化Gym环境
        self.figsize = figsize  # 初始化绘图窗口大小

        plt.figure(figsize=figsize)  # 创建绘图窗口
        plt.title(self.env.spec.id if hasattr(env.spec, 'id') else '')  # 标题设为环境名
        self.img = plt.imshow(env.render())  # 在绘图窗口中显示初始图像

    def render(self, title=None):
        image_data = self.env.render()  # 获取当前环境图像渲染数据

        self.img.set_data(image_data)  # 更新绘图窗口中的图像数据
        display.display(plt.gcf())  # 刷新显示
        display.clear_output(wait=True)  # 有新图片时再清除绘图窗口原有图像
        if title:  # 如果有标题就显示标题
            plt.title(title)

class EVs_Env(gym.Env):
    # 构造函数，参数为node
    def __init__(self, EVs, env_info):

        super(EVs_Env, self).__init__()

        # 环境信息
        self.data = env_info['data']
        self.distance = env_info['distance']
        self.speed = env_info['speed']
        self.roads = env_info['roads']
        self.node_road = env_info['node_road']
        self.node_num = len(self.data)
        self.EVs = EVs

        # 定义动作空间,动作空间为[0,1,2,...,49],选择动作n表示到n号节点
        self.action_space = spaces.Discrete(self.node_num)
        # # 定义观测空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 1), dtype=np.uint8)

        self.start = self.EVs['start']
        self.end = self.EVs['end']
        self.init_power = self.EVs['init_power']
        self.max_power = self.EVs['max_power']
        self.dead_line = self.EVs['dead_line']
        self.consumption = self.EVs['consumption'] / 100

        #后续补充新的属性
        self.success = False    #用于评判模型优劣


        # 剩余时间
        self.remain_time = self.dead_line
        # 当前电量
        self.current_power = self.init_power
        # 当前处在几号节点
        self.current = self.start

        # 用于动态更新奖励函数
        self.episode = 0
    # 环境重置函数
    def reset(self):
        self.start = self.EVs['start']
        self.end = self.EVs['end']
        self.init_power = self.EVs['init_power']
        self.max_power = self.EVs['max_power']
        self.dead_line = self.EVs['dead_line']
        self.consumption = self.EVs['consumption'] / 100    # 注意单位

        self.current = self.start

        self.remain_time = self.dead_line
        self.current_power = self.init_power
        self.last_action = self.start
        return self._get_state(), {}

    # 动作执行函数    函数返回[ns, reward, terminated, info]
    def step(self, action):
        is_charge = self.roads[self.current][action]     # 是否充电路段
        distance = self.distance[self.current][action]   # 距离
        speed = self.speed[self.current][action]         # 行驶速度
        time_consuming = distance / speed  # 行驶耗时
        if time_consuming == 0:
            print("jinggao")
        # time_consuming = distance/speed                  # 行驶耗时
        self.remain_time -= time_consuming               # 剩余时间
        consumption = distance * self.consumption        # 行驶消耗

        '''定义一些参数'''
        c_1 = 1.5 #充电系数
        c_2 = 100 #超时未到达终点
        c_3 = 150 #超时但是到了终点
        c_4 = 100 #电量耗尽但是到了终点
        c_5 = 150 #电量耗尽且未到达终
        # if self.episode < 1000:
        #     c_6 = 100  # 到达终点
        # else:
        #     c_6 = 40
        c_6 = 60
        c_7 = 250 #电量奖励基础
        if self.episode < 1000:
            c_8 = 100  # 给予负向奖励，鼓励满电到达
        else:
            c_8 = 200
        beta = 2

        '''奖励设置'''
        rewards = {
            'r_1' : 0 ,
            'r_2': -c_2 ,
            'r_3': -c_3 ,
            'r_4': -c_4 ,
            'r_5': -c_5 ,
            'r_6': c_6 ,
            'r_7': 0,
            'r_8': c_8
        }

        #更新电量，对这种情况的奖励进行维护

        last_power = self.current_power

        if is_charge:
            charge_power = 100 * time_consuming     # 充电功率为100Kw

            charge_change = charge_power - consumption
            #更新电量
            if charge_change < 0 :
                self.current_power += charge_change #如果电流变化为负,加上这个负的
            else:
                self.current_power = min(self.current_power + charge_change, self.max_power) #保证当前电量是车子的最大电量
            rewards['r_1'] = c_1 * (self.current_power - last_power)
            # rewards['r_1'] = c_1 * charge_change #
        else:
            self.current_power -= consumption


        # 无论怎样都要更新当前位置为执行动作后的位置，否则你还没做下一步，你怎么就知道你会失败呢
        self.current = action

        # 执行动作后，如果剩余时间<0或者电量<0则表示在半路就终止了
        if self.remain_time < 0:
            if self.current != self.end:
                r = rewards['r_3']
            else:
                r = rewards['r_2']
            self.success = False    #任务失败
            done = True
            return self._get_state(), r, done, {}

        if self.current_power < 0:
            if self.current != self.end:
                r = rewards['r_5']
            else:
                r = rewards['r_4']
            self.success = False    #任务失败
            done = True
            return self._get_state(), r, done, {}

        if self.current == self.end:
            rewards['r_7'] = c_7 * math.pow((self.current_power/self.max_power),beta) + rewards['r_6']
            # if self.current_power != self.max_power:
            #     rewards['r_7'] -= rewards['r_8']
            r = rewards['r_7']
            self.success = True
            done = True
            return self._get_state(), r, done, {}
        else:
            r = rewards['r_1']
            self.success = False
            done = False
            return self._get_state(), r, done, {}

    def _get_state(self):
        return [self.current, self.current_power, self.remain_time]