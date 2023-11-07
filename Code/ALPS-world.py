# coding=utf-8
import math
import time
import numpy as np
import random
import itertools
import pickle
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import matplotlib
import matplotlib.colors as clr

# 根据某一概率生成某一数字的方法，用以进行概率判断
def prob2num(seq, prob):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(seq, prob):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

# 定义居民agents所在的社区community类
class community(object):
    # 初始化实验社区的方法
    # 参数列表：
    # N：社区内agents的总数量
    # h：社区内居住单元的总数量
    # size：社区的面积大小，以 米（m） 计算
    # is_lockdown：是否采取封锁措施
    # T_0 T_1：开始和结束封锁措施的时间，以 天（d） 计算
    # p_0：人群中不服从限制性措施的agent比例
    def __init__(self, N, h, size, is_lockdown, T_0, T_1, p_0):
        self.N = N
        self.h = h
        self.size = size
        self.p_0 = p_0
        self.is_lockdown = is_lockdown
        if is_lockdown == True:
            self.T_0 = T_0
            self.T_1 = T_1
        else:
            self.T_0 = 9999999
            self.T_1 = 9999999

# 定义居民个体agent类
class agent(object):
    '''
    ID：agent的唯一身份编号
    h_ID：agent所属居住单元的唯一编号
    h_loc：agent所属居住单元的位置

    init_loc：agent的初始化位置
    cur_loc：当前agent位置
    mot_track：agent整体移动轨迹

    v_t：当前时刻的agent的瞬时速度
    v_array: agent各个时刻的瞬时速度集合

    alpha: 计算agent瞬时速度的参数
    mu:    计算agent瞬时速度的参数
    sigma: 计算agent瞬时速度的参数
    delta: 计算agent即时位置的参数

    is_infected: 当前时刻下agent是否已感染
    is_pathogeny: 当前时刻下agent是否已发病
    is_asy: 如果感染了，那agent是否是无症状患者
    is_mild: 当前时刻下agent的感染是否是轻微/普通型
    is_alive: 当前时刻下agent是否存活
    is_healthy: 当前时刻下agent是否健康
    is_immune: 当前时刻下该agent是否获得免疫力
    is_obey: 该agent在封锁的情况下是否服从管理

    t_infected: 该agent感染传染病的时刻
    t_pathogeny: 该agent发病的时刻
    t_rec_dead: 该agent从感染中康复或者死亡的时刻
    T_latent: 该agent感染传染病的潜伏时长
    T_pathogeny: 该agent发病后的累计时长
    '''
    def __init__(self, init_loc, h_ID, h_loc, ID, is_obey):
        # 初始信息
        self.ID = ID
        self.h_ID = h_ID
        self.h_loc = h_loc
        self.is_obey = is_obey
        # 初始位置
        self.init_loc = init_loc
        self.cur_loc = init_loc
        # 初始运动
        self.v_t = np.array([0, 0])
        self.v_array = []
        self.v_array.append(self.v_t)
        self.mot_track = []
        self.mot_track.append(self.init_loc)
        # 初始健康情况
        self.is_infected = False
        self.is_pathogeny = False
        self.is_asy = False
        self.is_mild = False
        self.is_alive = True
        self.is_healthy = True
        self.is_immune = False
        self.T_latent = 0
        self.T_pathogeny = 0
    
    # 改变agent这一时刻瞬时速度及相应位置的方法
    def change_v_loc(self, alpha, mu, sigma, delta, size):
        # 从v_array和mot_track中获取上一时刻的瞬时速度和位置
        v_before = self.v_array[-1]
        loc_before = self.mot_track[-1]
        # 根据运动模型公式计算当前时刻的瞬时速度和相应位置
        # 计算两个高斯增量w_t_x w_t_y,两个高斯增量分别作用在速度的x方向和y方向
        w_t_x = round(np.random.normal(loc=0.0, scale=1), 1)
        w_t_y = round(np.random.normal(loc=0.0, scale=1), 1)
        w_t = np.array([w_t_x, w_t_y])
        self.v_t = mu*v_before + (1-mu)*alpha*(self.h_loc - loc_before) + sigma*w_t
        self.cur_loc = loc_before + delta*self.v_t
        old_v_t = self.v_t
        # --->当新位置超过社区边界时，考虑反射性边界<----
        x_out = False
        y_out = False
        if self.cur_loc[0] >= size[0] or self.cur_loc[0] <= 0:
            # 当横坐标越界时
            x_out = True
        if self.cur_loc[1] >= size[1] or self.cur_loc[1] <= 0:
            # 当纵坐标越界时
            y_out = True
        # 进行不同情况下的越界判断，并进行运动反弹
        if x_out == True and y_out == True:
            # x 和 y 同时越界
            new_vt = np.array([-0.5*self.v_t[0], -0.5*self.v_t[1]])
        elif x_out == True and y_out == False:
            # 仅有 x 越界了
            new_vt = np.array([-0.5*self.v_t[0], self.v_t[1]])
        elif x_out == False and y_out == True:
            # 仅有 y 越界了
            new_vt = np.array([self.v_t[0], -0.5*self.v_t[1]])
        elif x_out == False and y_out == False:
            # 没有发生越界时
            new_vt = self.v_t
        # 重新赋值速度和位置
        self.cur_loc = loc_before + delta*new_vt
        self.v_t = new_vt
        # 避免后续计算位数爆炸，把小数位舍去
        self.cur_loc = np.array([int(self.cur_loc[0]), int(self.cur_loc[1])])
        self.v_t = np.array([int(self.v_t[0]), int(self.v_t[1])])
        # if self.cur_loc[0] >= 140 or self.cur_loc[1] >= 140 or self.cur_loc[0] < 0 or self.cur_loc[1] < 0:
        #     print('ID:'+str(self.ID)+' cur:'+str(self.cur_loc)+' cur_vt:'+str(self.v_t)+' old_vt:'+str(old_v_t)+' before:'+ str(loc_before))
        # 将新的速度和位置数据存入
        self.v_array.append(self.v_t)
        self.mot_track.append(self.cur_loc)
    

    # 当接触感染者之后agent开始以一定概率被感染进入潜伏期
    # t: 当前的时刻
    # P_I: 感染的几率
    def agent_infected(self, t, P_I):
        # 判断当前时刻agent是否被感染
        result_inf = prob2num([0, 1], [1-P_I, P_I])
        # 当概率判断结果为0时，agent在该时刻未被感染
        # 当概率判断结果为1时，该agent在该时刻已经患病，进一步判断是FT还是NFT
        if result_inf == 1:
            self.is_healthy = False
            self.is_infected = True
            # 记录被感染时刻(感染刚刚发生时不增加感染时长)
            self.t_infected = t
            print('ID-', self.ID, ' is infected')
    
    # 结束潜伏期之后进入发病期
    # t：当前的时刻
    # P_a：成为无症状感染者的概率
    # P_m：成为轻症及普通型患者的概率
    # P_c：成为重症及危重症患者的概率
    def agent_pathogeny(self, t, P_a, P_m, P_c):
        # 判断当前时刻agent是否成为无症状感染者
        result_a = prob2num([0,1],[1-P_a, P_a])
        # 如果是无症状感染者
        if result_a == 1:
            self.is_asy = True
            self.is_pathogeny = True
            self.t_pathogeny = t
            print('-->>ID-', self.ID, ' is ASY')
        # 如果是出现症状的感染者
        elif result_a == 0:
            self.is_asy = False
            self.is_pathogeny = True
            self.t_pathogeny = t
            # 判断当前时刻agent是轻症患者还是重症患者
            result_m = prob2num([0,1], [1-P_m, P_m])
            # 如果是轻症患者
            if result_m == 1:
                self.is_mild = True
                print('-->>ID-', self.ID, ' is Mild')
            # 如果是重症患者
            elif result_m ==0:
                self.is_mild = False
                print('-->>ID-', self.ID, ' is Critical')
    
    # 发病期结束后患者痊愈或者死亡
    # t：当前时刻
    # P_m_r,P_m_d：轻症患者死亡或痊愈的概率
    # P_c_r,P_c_d：重症患者死亡或痊愈的概率
    def agent_r_d(self, t, P_m_r, P_m_d, P_c_r, P_c_d):
        # 判断当前时刻无症状患者是否痊愈（发病期结束必定痊愈）
        if self.is_asy == True and self.is_healthy == False and self.is_pathogeny == True:
            self.is_healthy = True
            self.is_infected = False
            self.is_immune = True
            self.t_rec_dead = t
        # 判断当前时刻轻症患者是否痊愈或死亡或者继续发病
        if self.is_asy == False and self.is_mild == True and self.is_healthy == False and self.is_pathogeny == True:
            result_r = prob2num([0,1], [1-P_m_r, P_m_r])
            result_d = prob2num([0,1], [1-P_m_d, P_m_d])
            # 如果轻症患者痊愈
            if result_r == 1 and result_d == 0:
                self.is_healthy = True
                self.is_infected = False
                self.is_immune = True
                self.t_rec_dead = t
                print(' ID-', self.ID, ' Mild RECO')
            # 如果轻症患者死亡
            elif result_d == 1:
                self.is_infected = False
                self.is_alive = False
                self.t_rec_dead = t
                print(' ID-', self.ID, ' Mild DEAD')
            else:
                pass 
        # 判断当前时刻重症患者是否痊愈或死亡或继续发病
        if self.is_asy == False and self.is_mild == False and self.is_healthy == False and self.is_pathogeny == True:
            result_r = prob2num([0,1], [1-P_c_r, P_c_r])
            result_d = prob2num([0,1], [1-P_c_d, P_c_d])
            # 如果重症患者痊愈
            if result_r == 1 and result_d == 0:
                self.is_healthy = True
                self.is_infected = False
                self.is_immune = True
                self.t_rec_dead = t
                print(' ID-', self.ID, ' Critical RECO')
            # 如果重症患者死亡
            elif result_d == 1:
                self.is_infected = False
                self.is_alive = False
                self.t_rec_dead = t
                print(' ID-', self.ID, ' Critical DEAD')
            else:
                pass 

# 定义模拟中涉及的传染病infection类
class infection(object):
    # 初始化COVID-19传染病的方法
    # r_0：新冠传播中判定密切接触的距离，以 米 为单位
    # T_i：潜伏期时长, 以 小时 为单位
    # T_m：轻症患者发病期时长 以 小时 为单位
    # T_c：重症患者发病期时长 以 小时 为单位
    # P_I：密切接触后感染新冠的概率
    # P_a：潜伏期后成为无症状患者的概率
    # P_m_r, P_m_d：轻症患者痊愈及死亡概率
    # P_c_r, P_c_d：重症患者痊愈及死亡概率
    def __init__(self, r_0, T_i, T_m, T_c, P_I, P_a, P_m, P_c, P_m_r, P_m_d, P_c_r, P_c_d):
        self.r_0 = r_0
        self.T_i = T_i
        self.T_m = T_m
        self.T_c = T_c
        self.P_I = P_I
        self.P_a = P_a
        self.P_m = P_m
        self.P_c = P_c
        self.P_m_r = P_m_r
        self.P_m_d = P_m_d
        self.P_c_r = P_c_r
        self.P_c_d = P_m_d

# 根据社区规模和居住单元数量生成相应的居住单元坐标
# h: 居住单元的数量，以9、25、49、81等完全平方数为例，开平方后为奇数
# size: 社区的规模，以米为单位，通常是标准的正方形
# interval: 居住单元彼此间的间隔
def create_housing_units(h, size, interval):
    # 把生成的居住单元保存在一个数组里
    h_array = []
    h_all = []
    x_len = size[0]
    y_len = size[1]
    interval_len = interval
    # 对h开平方得到居住单元的行列
    h_sqrt = int(math.sqrt(h))
    # 中心居住点坐标固定(社区正中间)
    h_centre = np.array([x_len/2, y_len/2])
    h_array.append(h_centre)
    h_all.append(h_centre)
    # 围绕中心居住单元生成其余的居住单元
    # 生成同一竖向的居住单元
    for i in range(1, int((h_sqrt+1)/2)):
        h_up = np.array([x_len/2, (y_len/2) + (interval_len*i)])
        h_down = np.array([x_len/2, (y_len/2) - (interval_len*i)])
        h_array.append(h_up)
        h_array.append(h_down)
        h_all.append(h_up)
        h_all.append(h_down)
    # 以中心竖直方向的居住单元向两侧生成同排的居住单元
    for h_vert in h_array:
        for i in range(1, int((h_sqrt+1)/2)):
            h_left = np.array([h_vert[0]-(interval_len*i), h_vert[1]])
            h_right = np.array([h_vert[0]+(interval_len*i), h_vert[1]])
            h_all.append(h_left)
            h_all.append(h_right)
    # 返回这些居住单元
    return h_all

# 根据社区规模和居住单元数量生成相应的居住单元坐标
# h 居住单元总数量
# h_x 居住单元每一行多少栋
# h_y 居住单元每一列多少栋
# x_interval 水平方向间隔多少米
# y_interval 竖直方向间隔多少米
def create_new_housing_units(h, size, x_interval, y_interval, h_x, h_y):
    # 把生成的居住单元保存在一个数组里
    h_array = []
    h_all = []
    x_len = size[0]
    y_len = size[1]
    # 固定第一个点的坐标
    init_x = (x_len/2)-((h_x-1)/2)*x_interval
    init_y = (y_len/2)-((h_y-1)/2)*y_interval
    # 以第一个点开始生成第一列所有住所坐标
    for i in range(h_y):
        h_y_local = np.array([init_x, init_y+i*y_interval])
        h_array.append(h_y_local)
    # 再以第一列为基准生成每一行的坐标
    for h_y_local in h_array:
        for i in range(h_x):
            h_x_local = np.array([h_y_local[0]+i*x_interval, h_y_local[1]])
            h_all.append(h_x_local)
    # 返回这些居住单元
    return h_all
    
# 根据社区的情况，批量生成实验用的agent
# h_all: 所有的居住单元坐标
# N: 需要生成的agent数量
# size: 社区的规模，方便生成随机的初始位置
# sco: 生成的agent分布在社区周围多少距离的范围内，以 m 为单位计算
# p_0: 服从管理的社区人数比例
# random_type: true-所属住所及初始位置随机设置没有约束；false-每个住所agent数量一致，agent约束分布在所属住所附近
def create_agents(h_all, N, size, sco, p_0, random_type):
    # 遵从限制的人数，取整
    obey_num = int(N * p_0)
    # 以一个数组保存所有的agents对象
    agent_all = []
    # 将社区规模转变为初始位置的随机范围
    x = size[0]
    y = size[1]
    # 初始化一个数组，登记每一个住所内居住了多少agent
    h_agents_num = []
    for num in range(len(h_all)):
        h_agents_num.append(0)
    # random_type = true 随机分配agent的住所，各个住所的分配agent人数不同，agent也不一定能分配在所属住所附近
    # 平均分配agent的住所，每个住所分配的agent人数保持相同，agent分配在所属住所的附近
    # 根据输入的N批量生成agent对象,ID从0开始编号 [方便以ID为0的agent作为感染起始点]
    for ID in range(0, N):
        # 如果是社区范围内随机分配
        if random_type == True:
            # 生成一个h_index从h_all随机挑选一个居住单元分配给agent
            h_index = random.randint(0, len(h_all)-1)
            h_ID = h_index
            h_loc = h_all[h_index]
            h_agents_num[h_index] = h_agents_num[h_index] + 1
            # 在社区范围内随机生成位置(避免生成在边界附近，范围适当缩小)
            rand_x = random.randint(2, x-2)
            rand_y = random.randint(2, y-2)
            rand_loc = np.array([rand_x, rand_y])
        # 如果是分配到所属住房附近
        if random_type == False:
            # 从第一个houseunit开始分配
            limit = int(N/len(h_all))
            for h_index in range(len(h_all)):
                # 当某住所还没有住满时
                if h_agents_num[h_index] < limit:
                    h_ID = h_index
                    h_loc = h_all[h_index]
                    h_agents_num[h_index] = h_agents_num[h_index] + 1
                    # 在住所的周边sco的范围内随机生成位置，且避免生成在边界附近
                    # 进行区域限制
                    # 左边界
                    if (h_loc[0]-sco) <= 0:
                        x_left = 0
                    elif (h_loc[0]-sco) > 0:
                        x_left = h_loc[0]-sco
                    # 右边界
                    if (h_loc[0] + sco) <= x:
                        x_right = h_loc[0] + sco
                    elif (h_loc[0] + sco) > x:
                        x_right = x
                    # 上边界
                    if (h_loc[1] + sco) <= y:
                        y_top = h_loc[1] + sco
                    elif (h_loc[1] + sco) > y:
                        y_top = y
                    # 下边界
                    if (h_loc[1] - sco) <= 0:
                        y_bottom = 0
                    elif (h_loc[1] - sco) > 0:
                        y_bottom = h_loc[1] - sco
                    # print(x_left,x_right,y_bottom,y_top)
                    rand_x = random.randint(x_left + 2, x_right - 2)
                    rand_y = random.randint(y_bottom + 2, y_top - 2)
                    rand_loc = np.array([rand_x, rand_y])
                    break
                if h_agents_num[-1] >= limit:
                    # 当还有未分配位置的人员
                    if N%len(h_all) != 0:
                        print('Special agent ID: '+ str(ID))
                        rand_h_num = random.randint(0, len(h_all)-1)
                        h_ID = rand_h_num
                        h_loc = h_all[rand_h_num]
                        h_agents_num[rand_h_num] = h_agents_num[rand_h_num] + 1
                        if (h_loc[0]-sco) <= 0:
                            x_left = 0
                        elif (h_loc[0]-sco) > 0:
                            x_left = h_loc[0]-sco
                        if (h_loc[0] + sco) <= x:
                            x_right = h_loc[0] + sco
                        elif (h_loc[0] + sco) > x:
                            x_right = x
                        if (h_loc[1] + sco) <= y:
                            y_top = h_loc[1] + sco
                        elif (h_loc[1] + sco) > y:
                            y_top = y
                        if (h_loc[1] - sco) <= 0:
                            y_bottom = 0
                        elif (h_loc[1] - sco) > 0:
                            y_bottom = h_loc[1] - sco
                        rand_x = random.randint(x_left + 2, x_right - 2)
                        rand_y = random.randint(y_bottom + 2, y_top - 2)
                        rand_loc = np.array([rand_x, rand_y])
                        break                
        # 当ID<obey_num 时，认为该agent遵守封锁
        if ID < obey_num:
            # 初始化agent
            # init_loc, h_ID, h_loc, ID, is_obey
            new_agent = agent(rand_loc, h_ID, h_loc, ID, True)
            # if new_agent.init_loc[0] >= 140 or new_agent.init_loc[1] >= 140 or new_agent.init_loc[0] < 0 or new_agent.init_loc[1] < 0:
            #     print('xxxx ID:'+str(new_agent.ID)+' init:'+str(new_agent.init_loc))
        elif ID >= obey_num:
            new_agent = agent(rand_loc, h_ID, h_loc, ID, False)
            # if new_agent.init_loc[0] >= 140 or new_agent.init_loc[1] >= 140 or new_agent.init_loc[0] < 0 or new_agent.init_loc[1] < 0:
            #     print('xxxx ID:'+str(new_agent.ID)+' init:'+str(new_agent.init_loc))
        # 加入数组中保存
        agent_all.append(new_agent)

    # 返回生成的所有agent和住所分配情况
    return agent_all, h_agents_num

# 计算感染率、易感（正常）率、康复率、死亡率
# agent_all: 当前某一时刻的agent队列
# I_array: 感染队列
# S_array: 未感染队列
# R_array: 康复队列
# D_array: 死亡队列
# N: 总人数
def cal_ISRD(agent_all, I_array, S_array, R_array, D_array, N):
    all_num = len(agent_all)
    I_num = 0
    R_num = 0
    D_num = 0
    for agent in agent_all:
        # 感染的人数统计（既没有康复也没有死亡）
        if agent.is_healthy == False and agent.is_infected == True and agent.is_alive == True:
            I_num = I_num + 1    
        # 康复的人数统计(获得免疫力的即是康复者)
        if agent.is_immune == True and agent.is_alive == True:
            R_num = R_num + 1
        # 死亡的人数统计
        if agent.is_alive == False:
            D_num = D_num + 1
    # 全部计算完后计算S_num
    S_num = N-I_num-R_num-D_num
    # 存入队列中
    # I_array.append(round(I_num/all_num,4)*100)
    # S_array.append(round(S_num/all_num,4)*100)
    # R_array.append(round(R_num/all_num,4)*100)
    # D_array.append(round(D_num/all_num,4)*100)
    I_array.append(I_num)
    S_array.append(S_num)
    R_array.append(R_num)
    D_array.append(D_num)
    # 返回结果
    return I_array, S_array, R_array, D_array

# 进行agent之间的接触判断，必要时计算两个agent之间的距离
# 若两个agent均未感染或都已经感染，则无需计算距离
# 若两个agent中有一名是感染者，另一名未感染且不具有免疫力，则计算距离
#   如果二者距离小于等于r_0，则判断改agent是否被感染
#   如果二者距离大于r_0， 则无任何操作
# agent_all: 保存所有agent信息的队列
# r_0: 最大接触距离
def agents_touch(agent_all, r_0, t, P_I):
    # 从agent_all中筛选出所有agent的两两组合，并计算组合内agent间的距离
    agent_comb_all = itertools.combinations(agent_all, 2)
    for agent_comb in agent_comb_all:
        # 所有的判断建立在两个agent都存活的前提下，若至少其中一个已经死亡，则没有判断接触的必要
        # 假定尸体不是感染病的传播源，不会因接触尸体而感染
        if agent_comb[0].is_alive == True and agent_comb[1].is_alive == True:
            # 如果二者均为感染或二者均不感染
            if agent_comb[0].is_infected == False and agent_comb[1].is_infected == False:
                continue
            elif agent_comb[0].is_infected == True and agent_comb[1].is_infected == True:
                continue
            # 如果二者中有一个感染但另一个具备免疫力 
            elif agent_comb[0].is_infected == True and agent_comb[1].is_immune == True:
                continue
            elif agent_comb[1].is_infected == True and agent_comb[0].is_immune == True:
                continue
            # 如果二者中有一个感染且另一个未感染不具有免疫力
            elif agent_comb[0].is_infected == True and agent_comb[1].is_infected == False and agent_comb[1].is_immune == False:
                loc_1 = agent_comb[0].cur_loc
                loc_2 = agent_comb[1].cur_loc
                agent_dis = round(math.sqrt(int(math.pow(int(loc_1[0]-loc_2[0]), 2) + math.pow(int(loc_1[1]-loc_2[1]), 2))),1)
                # print(loc_1, ' ', loc_2, ' ', agent_dis)
                # 当间距小于等于最大感染距离,进行感染判断
                if agent_dis <= r_0:
                    agent_comb[1].agent_infected(t, P_I)
                continue
            # 同上
            elif agent_comb[1].is_infected == True and agent_comb[0].is_infected == False and agent_comb[0].is_immune == False:
                loc_1 = agent_comb[0].cur_loc
                loc_2 = agent_comb[1].cur_loc
                agent_dis = round(math.sqrt(int(math.pow(int(loc_1[0]-loc_2[0]), 2) + math.pow(int(loc_1[1]-loc_2[1]), 2))),1)
                # print(loc_1, ' ', loc_2, ' ', agent_dis)
                if agent_dis <= r_0:
                    agent_comb[0].agent_infected(t, P_I)
                continue

# 计算当前agent是否靠近其所在住所
# 住所是一个范围，在此范围内都能算回到住所，以 m 计算
# 在住所范围内返回true，否则返回false
def at_home(agent, scope):
    # 获得agent的当前位置和所属的居住地位置
    loc_1 = agent.cur_loc
    loc_2 = agent.h_loc
    # 计算二者间的距离
    dis = round(math.sqrt(int(math.pow(int(loc_1[0]-loc_2[0]), 2) + math.pow(int(loc_1[1]-loc_2[1]), 2))),1)
    # if loc_1[0] >= 140 or loc_1[1] >= 140:
    #     print('ID: '+ str(agent.ID) + ' is_obey:' + str(agent.is_obey))
    if dis <= scope:
        return True
    else :
        return False

# 初始感染者/零号病人的感染（感染未发病）
# t: 当前时刻
# zero-type：控制初始感染者的生成方式
#   0：以ID = 0的agent为初始感染者
#   1：选取出整个agent群体中与其余agent距离合计最小的agent作为初始感染者
#   2：选取出整个agent群体中与其余agent距离小于感染范围最多的agent作为初始感染者
#   3: 同时出现三个感染者，分别符合1,2,3
def zero_infected(zero_type, t, r_0, agent_all):
    print('Zero Infected Type Num:', zero_type)
    if zero_type == 0 or zero_type == 3:
        for agent in agent_all:
            if agent.ID == 0:
                # 调整健康标签
                agent.is_healthy = False
                agent.is_infected = True
                # 记录被感染时刻(感染刚刚发生时不增加感染时长)
                agent.t_infected = t
                break
    if zero_type == 1 or zero_type == 3:
        # 设置存储距离的队列
        distance_array = []
        for agent in agent_all:
            distance_array.append(0)
        # 从agent_all中筛选出所有agent的两两组合，并计算组合内agent间的距离
        agent_comb_all = itertools.combinations(agent_all, 2)
        for agent_comb in agent_comb_all:
            loc_1 = agent_comb[0].cur_loc
            loc_2 = agent_comb[1].cur_loc
            agent_dis = math.sqrt(int(math.pow(abs(loc_1[0]-loc_2[0]), 2) + math.pow(abs(loc_1[1]-loc_2[1]), 2)))
            distance_array[agent_comb[0].ID] = distance_array[agent_comb[0].ID] + agent_dis
            distance_array[agent_comb[1].ID] = distance_array[agent_comb[1].ID] + agent_dis
        # 找到距离合计最短的zero
        zero_ID = distance_array.index(min(distance_array))
        print('Zero ID:', zero_ID, ' Total dis:', distance_array[zero_ID])
        for agent in agent_all:
            if agent.ID == zero_ID:
                # 调整健康标签
                agent.is_healthy = False
                agent.is_infected = True
                # 记录被感染时刻(感染刚刚发生时不增加感染时长)
                agent.t_infected = t
                break
    if zero_type == 2 or zero_type == 3:
        # 设置存储距离的队列
        distance_array = []
        for agent in agent_all:
            distance_array.append(0)
        # 从agent_all中筛选出所有agent的两两组合，并计算组合内agent间的距离
        agent_comb_all = itertools.combinations(agent_all, 2)
        for agent_comb in agent_comb_all:
            loc_1 = agent_comb[0].cur_loc
            loc_2 = agent_comb[1].cur_loc
            agent_dis = math.sqrt(int(math.pow(abs(loc_1[0]-loc_2[0]), 2) + math.pow(abs(loc_1[1]-loc_2[1]), 2)))
            if agent_dis <= r_0:
                distance_array[agent_comb[0].ID] = distance_array[agent_comb[0].ID] + 1
                distance_array[agent_comb[1].ID] = distance_array[agent_comb[1].ID] + 1
        # 找到距离小于r_0次数最多的agent
        zero_ID = distance_array.index(max(distance_array))
        print('Zero ID:', zero_ID, ' Total dis:', distance_array[zero_ID])
        for agent in agent_all:
            if agent.ID == zero_ID:
                # 调整健康标签
                agent.is_healthy = False
                agent.is_infected = True
                # 记录被感染时刻(感染刚刚发生时不增加感染时长)
                agent.t_infected = t
                break

# 运行模拟
def run_code(N, h, size, is_lockdown, T_0, T_1, p_0, x_interval, y_interval, h_x, h_y, sco, sigma):
    # 实例化一个用于做实验的社区(以米为单位输入数据)
    # N, h, size, is_lockdown, T_0, T_1, p_0
    demo_community = community(N, h, size, is_lockdown, T_0, T_1, p_0)
    # 实例化一个实验涉及的传染病
    # r_0, T_i, T_m, T_c, P_I, P_a, P_m_r, P_m_d, P_c_r, P_c_d
    demo_infection = infection(1, 336, 336, 1008, 0.05, 0.001, 0.8, 0.199, 0.95, 0.05, 0.86, 0.14)
    # 根据相关信息批量生成实验社区中各个居住单元的坐标
    # h, size, interval 
    # h_all = create_housing_units(demo_community.h, demo_community.size, interval)
    h_all = create_new_housing_units(demo_community.h, demo_community.size, x_interval, y_interval, h_x, h_y)
    print(h_all)
    # 批量生成实验社区中的agent实例
    # h_all, N, size, p_0
    agent_all, h_agents_num = create_agents(h_all, demo_community.N, demo_community.size, sco, demo_community.p_0, False)
    print('Houseunits info:', h_agents_num)
    # 保存最初的对象
    path = './object/'
    file_name = 'N=' + str(demo_community.N) + ' h=' + str(demo_community.h) + ' sigma=' + str(sigma) + ' T0=' + str(demo_community.T_0) + ' T1=' + str(demo_community.T_1)
    
    # 所需环境全部生成后，进行实验模拟
    # t：实验模拟的总时长（总迭代次数） 单位为小时
    T = 2400
    # 提取相关时间
    # 封锁起止时间
    T_0 = demo_community.T_0 
    T_1 = demo_community.T_1
    # 潜伏期及发病期
    T_i = demo_infection.T_i
    T_m = demo_infection.T_m
    T_c = demo_infection.T_c
    # 初始化agent运动模型所需的各种参数
    # delta没有做要求，此处以1计算
    alpha = 0.2
    mu = 1
    delta = 1
    # 初始感染者的生成模式
    zero_type = 0
    # 初始化结果保存序列
    I_array = []
    S_array = []
    R_array = []
    D_array = []
    # 设置控制模拟是否运行的开关变量
    is_start = True
    if is_start:
        # 开始模拟迭代
        for t in range(T):
            if (t+1)%24 == 0:
                print('Day: ', int((t+1)/24))
            # t = 0 时为最初始状态, 无agent移动, 设置初始感染者（方便起见 ID = 0的agent即初始感染者）
            if t == 0:
                # 第一分钟开始初始感染
                zero_infected(zero_type, t, demo_infection.r_0, agent_all)
                # 统计初始ISRD情况
                I_array, S_array, R_array, D_array = cal_ISRD(agent_all, I_array, S_array, R_array, D_array, demo_community.N)
                # 记录初始情况
                agent_all_begin = agent_all
                agents_begin = path + file_name + '_begin.obj'
                with open(agents_begin, 'wb') as agents_begin_file:
                    pickle.dump(agent_all_begin, agents_begin_file)
            # t >= 1 之后的所有情况
            if t >= 1:
                # 进行感染者感染/痊愈/死亡
                # 进行未感染者的感染判断
                agents_touch(agent_all, demo_infection.r_0, t, demo_infection.P_I)
                # 进行感染者的发病/痊愈/康复
                for agent in agent_all:
                    # 仅计算还活着且已经被感染的agent
                    if agent.is_alive == True and agent.is_infected == True:
                        # 如果是感染者还没有足够的潜伏期
                        if agent.is_pathogeny == False and agent.T_latent <= T_i:
                            agent.T_latent = agent.T_latent + 1
                        # 如果是感染者且已够潜伏期，则进行发病类型判定
                        elif agent.is_pathogeny == False and agent.T_latent > T_i:
                            agent.agent_pathogeny(t, demo_infection.P_a, demo_infection.P_m, demo_infection.P_c)
                        # 如果是已经发病的患者
                        # 如果是无症状感染者且还未康复
                        if agent.is_asy == True and agent.is_healthy == False and agent.is_pathogeny == True:
                            # 无症状感染者的发病时间累计不够两周
                            if agent.T_pathogeny <= 336:
                                agent.T_pathogeny = agent.T_pathogeny + 1
                            # 累计足够发病时间，无症状患者必定康复
                            elif agent.T_pathogeny > 336:
                                agent.agent_r_d(t, demo_infection.P_m_r, demo_infection.P_m_d, demo_infection.P_c_r, demo_infection.P_c_d)
                        # 如果是轻症患者且未康复死亡
                        if agent.is_asy == False and agent.is_mild == True and agent.is_healthy == False and agent.is_pathogeny == True:
                            if agent.T_pathogeny <= T_m:
                                agent.T_pathogeny = agent.T_pathogeny + 1
                            elif agent.T_pathogeny > T_m:
                                agent.agent_r_d(t, demo_infection.P_m_r, demo_infection.P_m_d, demo_infection.P_c_r, demo_infection.P_c_d)
                        # 如果是重症患者且未康复死亡
                        if agent.is_asy == False and agent.is_mild == False and agent.is_healthy == False and agent.is_pathogeny == True:
                            if agent.T_pathogeny <= T_c:
                                agent.T_pathogeny = agent.T_pathogeny + 1
                            elif agent.T_pathogeny > T_c:
                                agent.agent_r_d(t, demo_infection.P_m_r, demo_infection.P_m_d, demo_infection.P_c_r, demo_infection.P_c_d)
                        
                    # 判定完感染及发病情况后，遍历每一个agent进行移动
                    # 进行agent移动(死亡agent不再进行移动)
                    # 在封锁之前，所有agent自由移动
                    if agent.is_alive == True:
                        if t < T_0:
                            agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                        # 在封锁期内，绝大部分agent向houseunit移动，极少部分agent继续自由移动
                        elif t >= T_0 and t <= T_1:
                            demo_infection.P_I = 0.01
                            # 如果agent尚未回到住所内
                            if at_home(agent, 5) == False:
                                # 如果是遵守封锁的agent，执行往住所方向的移动
                                if agent.is_obey == True:
                                    agent.change_v_loc(alpha, (1-mu), sigma, delta, demo_community.size)    
                                # 如果是不遵守封锁的agent，继续自由移动
                                elif agent.is_obey == False:
                                    agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                            # 如果agent已经回到住所内(或是凑巧在住所内)
                            elif at_home(agent, 5) == True:
                                # 如果是遵守封锁的agent，会在住所范围内停止不动         
                                # 如果是不遵守封锁的agent，仍会继续自由移动
                                if agent.is_obey == False:
                                    agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                        # 解除封锁后
                        elif t > T_1: 
                            # 无论agent是否在住所内，无论agent是否服从封锁，只要agent存活，就开始自由移动
                            agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                # 计算这一时刻的各类数值
                I_array, S_array, R_array, D_array = cal_ISRD(agent_all, I_array, S_array, R_array, D_array, demo_community.N)

            # 记录中间状态-1
            if t == 240:
                agent_all_middle_1 = agent_all
                agents_middle_1 = path + file_name + '_middle_1.obj'
                with open(agents_middle_1, 'wb') as agents_middle_1_file:
                    pickle.dump(agent_all_middle_1, agents_middle_1_file)
            # 记录中间状态-2
            if t == 480:
                agent_all_middle_2 = agent_all
                agents_middle_2 = path + file_name + '_middle_2.obj'
                with open(agents_middle_2, 'wb') as agents_middle_2_file:
                    pickle.dump(agent_all_middle_2, agents_middle_2_file)
            # 记录中间状态-3
            if t == 960:
                agent_all_middle_3 = agent_all
                agents_middle_3 = path + file_name + '_middle_3.obj'
                with open(agents_middle_3, 'wb') as agents_middle_3_file:
                    pickle.dump(agent_all_middle_3, agents_middle_3_file)
            # 记录中间状态-4
            if t == 1920:
                agent_all_middle_4 = agent_all
                agents_middle_4 = path + file_name + '_middle_4.obj'
                with open(agents_middle_4, 'wb') as agents_middle_4_file:
                    pickle.dump(agent_all_middle_4, agents_middle_4_file)
    
    
    agent_all_end = agent_all
    agents_end = path + file_name + '_end.obj'
    with open(agents_end, 'wb') as agents_end_file:
        pickle.dump(agent_all_end, agents_end_file)

    # 保存实验数据(序列)
    all_array = [I_array, S_array, R_array, D_array]
    all_array = [[row[i] for row in all_array] for i in range(len(all_array[0]))]
    header_name = ['Infected', 'Susceptible', 'Recovered', 'Dead']
    csv_file = pd.DataFrame(columns=header_name, data=all_array)
    file_name = 'N=' + str(demo_community.N) + ' h=' + str(demo_community.h) + ' sigma=' + str(sigma) + ' T0=' + str(demo_community.T_0) + ' T1=' + str(demo_community.T_1)
    csv_file.to_csv('./result/' + file_name + '.csv')
    # 保存实验数据（agent位置）
    path = './object/'
    print('---All data Save---')
    

    return T, demo_community, demo_infection, I_array, S_array, R_array, D_array, agent_all, h_all, sigma


if __name__ == '__main__':
    sigma = 4
    # sigma_list = [15]
    T_0_list = [120]
    T_1_list = [1200]
    # T_0_list = [120]
    # T_1_list = [1200,1440,1680]
    for T_0 in T_0_list:
        for T_1 in T_1_list:
            # N, h, size, is_lockdown, T_0, T_1, p_0, x_interval, y_interval, h_x, h_y, sco, sigma
            T, demo_community, demo_infection, I_array, S_array, R_array, D_array, agent_all, h_all, sigma = run_code(1000, 16, [140, 140], False, T_0, T_1, 0.98, 40, 40, 4, 4, 60, sigma)
        
            # 绘图的类型
            p_type = 1

            # 1. 查看ISRD曲线
            if p_type == 1 or p_type == 0:
                t_array = []
                for t in range(T):
                    t_array.append(t)
                # 默认采用times字体并解决其自动加粗问题
                plt.rc('font',family='Times New Roman')
                del matplotlib.font_manager.weight_dict['roman']
                matplotlib.font_manager._rebuild()
                plt.axis([0, T, 0, 1010])
                plt.plot(t_array, I_array ,"r",  label = 'Infected')
                plt.plot(t_array, S_array ,"b",  label = 'Susceptible')
                plt.plot(t_array, R_array ,"g",  label = 'Recovered')
                plt.plot(t_array, D_array ,color='black',  label = 'Dead')
                if demo_community.is_lockdown == True:
                    plt.axvline(demo_community.T_0, linestyle='--', color='orange', linewidth = 2.5)
                if demo_community.T_1 < 2400:    
                    plt.axvline(demo_community.T_1, linestyle='--', color='orchid', linewidth = 2.5)
                x_major_locator=MultipleLocator(24)
                # plt.ylim(0,100)
                plt.title('COVID19_ALPS Model')
                plt.xlabel('Time [Days]')
                plt.ylabel('The Number of Agents')
                plt.xticks([x*24 for x in range(0,120,20)], range(0,120,20))
                # plt.xticks([])
                # plt.yticks([])
                plt.legend()
                plt.show()
                # plt.grid(True)
                # mpl.rcParams['xtick.labelsize'] = 18
                # mpl.rcParams['ytick.labelsize'] = 18
                # plt.show()
                # file_name = 'ISRD_N=' + str(demo_community.N) + ' h=' + str(demo_community.h) + ' sigma=' + str(sigma) + ' T0=' + str(demo_community.T_0) + ' T1=' + str(demo_community.T_1)
                # plt.savefig('./picture/'+file_name+'.eps',dpi=1200,format='eps')
                # plt.savefig('./picture/'+file_name+'.png')
                # plt.clf()

            # 2. 查看单一agent运动路线图
            if p_type == 2:
                x_value = []
                y_value = []
                h_x = []
                h_y = []
                agent = agent_all[0]
                for loc in agent.mot_track:
                    x_value.append(loc[0])
                    y_value.append(loc[1])
                for h in h_all:
                    h_x.append(h[0])
                    h_y.append(h[1])
                s_x = agent.mot_track[0][0]
                s_y = agent.mot_track[0][1]
                e_x = agent.mot_track[-1][0]
                e_y = agent.mot_track[-1][1]
                plt.axis([0, demo_community.size[0], 0, demo_community.size[1]])
                plt.plot(x_value, y_value,"b--",linewidth=1)
                plt.scatter(x_value, y_value, s=5, c="red", marker='o')
                plt.scatter(h_x, h_y, s=60, c="w", marker='s', edgecolors='black')
                plt.scatter(s_x, s_y, s=40, c='blue', marker='*')
                plt.scatter(e_x, e_y, s=40, c='green', marker='*')
                plt.show()

            # 3. 查看单一agent每个小时的位移速度
            if p_type == 3 or p_type == 0:
                x_value = []
                y_value = []
                agent = agent_all[1]
                for v in agent.v_array:
                    lenth = sigma*(math.sqrt(math.pow(abs(v[0]),2) + math.pow(abs(v[1]),2)))
                    y_value.append(lenth)
                for i in range(len(y_value)):
                    x_value.append(i)
                plt.plot(x_value, y_value,"b--",linewidth=1)
                plt.show()

            # 4. 查看初始分布与后续分布图
            if p_type == 4:
                x_value = []
                y_value = []
                zero_x = []
                zero_y = []
                for agent in agent_all:
                    if agent.ID == 0:
                        zero_x.append(agent.cur_loc[0])
                        zero_y.append(agent.cur_loc[1])
                    else:
                        x_value.append(agent.cur_loc[0])
                        y_value.append(agent.cur_loc[1])
                
                plt.scatter(zero_x, zero_y, s=40, c='black', marker='*')
                plt.scatter(x_value, y_value, s=20, c="red", marker='o')
                plt.show()

