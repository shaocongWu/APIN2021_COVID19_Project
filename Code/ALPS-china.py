# coding=utf-8
import math
import time
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 依据概率生成某一数字的方法
def prob2num(seq, prob):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(seq, prob):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

# 定义居民个体所在的社区community类
class community(object):
    # 初始化某个社区的方法
    # N：社区内agent数量
    # h：社区内居住单元数量
    # size：社区的面积大小，以 米 计算
    # is_lockdown：是否采取限制性措施，true-采取，false-不采取
    # T_0：开始采取限制性措施的时间(天)，若is_lockdown = false, 则 T_0 = -1
    # is_remove：是否解除已经采取的限制性措施，true-解除，false-不解除
    # T_1：解除限制性措施的时间（天），若is_remove = false, 则 T_1 = -1
    # p_0：服从限制性措施的agent比例
    def __init__(self, N, h, size, is_lockdown, T_0, is_remove, T_1, p_0):
        self.N = N
        self.h = h
        self.size = size
        self.p_0 = p_0
        # 如果执行封锁，则设定T_0
        self.is_lockdown = is_lockdown
        if self.is_lockdown == False:
            self.T_0 = -1
        else:
            self.T_0 = T_0
        # 如果撤销封锁，则设定T_1
        self.is_remove = is_remove
        if self.is_remove == False:
            self.T_1 = -1
        else:
            self.T_1 = T_1
        

    # 输出社区的所有信息
    def show_community_info(self):
        print('---------Community Information---------')
        print('Total number of agents: ', self.N)
        print('Total number of housing units: ', self.h)
        print('Community size: ', self.size)
        print('is_lockdown? -- ',self.is_lockdown)
        print('is_remove? -- ', self.is_remove)
        print('lockdown starts: ', self.T_0)
        print('lockdown ends: ', self.T_1)
        print('Fraction of people following restrictions: ', self.p_0)


# 定义居民个体agent类
class agent(object):
    # ID: 用于标识agent唯一身份的编号
    # init_loc: agent初始化时的随机位置，[x, y]
    # cur_loc: agent的当前位置， [x, y]~t
    # mot_track: agent的运动轨迹，{[x, y], [x, y], ……} 
    # h_ID: agent所属居住单元的ID
    # h_loc: agent所属的居住单元的位置
    # v_t: 当前时刻agent的瞬时速度，初始速度为0 (t = 0) v是矢量
    # v_array: agent各个时刻的瞬时速度集合
    # alpha: 计算agent瞬时速度的参数
    # mu: 计算agent瞬时速度的参数
    # sigma: 计算agent瞬时速度的参数
    # delta: 计算agent即时位置的参数
    # t_0: agent的持续累计易感时间
    # is_infected: 当前时刻下agent是否已感染
    # is_asy: 当前时刻下agent是否是无症状患者，True--无症状患者，False--危重型
    # is_mild: 当前时刻下agent的感染是否是轻微/普通型，True--轻微型，False--危重型
    # t_infected: 该agent感染传染病的时刻
    # t_rec_dead: 该agent从感染中康复或者死亡的时刻
    # T_infected: 该agent感染传染病的累计时长
    # is_alive: 该agent当前的状态是否存活，True--存活, False--死亡
    # is_healthy: 该agent在当前时刻是否健康，True--健康, False--患病
    # is_immune: 该agent是否获得免疫力（从感染中恢复，且不具有二次感染风险）
    # is_obey: 该agent在封锁的情况下是否服从管理
    # 简单初始化一个agent对象只需要初始位置，隶属住所，初始速度和累计易感时间
    def __init__(self, init_loc, h_ID, h_loc, ID, is_obey):
        self.init_loc = init_loc
        self.cur_loc = init_loc
        self.h_ID = h_ID
        self.h_loc = h_loc
        self.ID = ID
        self.is_obey = is_obey
        self.v_t = np.array([0, 0])
        # 初始化速度值队列和初始化位置序列
        self.v_array = []
        self.v_array.append(self.v_t)
        self.mot_track = []
        self.mot_track.append(self.init_loc)
        # 初始化当前的健康与感染情况
        self.t_0 = 0
        self.is_infected = False
        self.T_infected = 0
        self.is_alive = True
        self.is_healthy = True
        self.is_FT = False
        self.is_immune = False
        self.is_change = False

    # 改变agent这一时刻瞬时速度及相应位置的方法
    # 输入的是km，需转换为m
    def change_v_loc(self, alpha, mu, sigma, delta, size):
        # 从v_array和mot_track中获取上一时刻的瞬时速度和位置
        v_before = self.v_array[-1]
        loc_before = self.mot_track[-1]
        # 根据运动模型公式计算当前时刻的瞬时速度和相应位置
        # 计算两个高斯增量w_t_x w_t_y,两个高斯增量分别作用在速度的x方向和y方向
        # 因为单位已具体至英尺，故仅需保留一位小数 
        w_t_x = round(np.random.normal(loc=0.0, scale=1), 1)
        w_t_y = round(np.random.normal(loc=0.0, scale=1), 1)
        w_t = np.array([w_t_x, w_t_y])
        self.v_t = mu*v_before + (1-mu)*alpha*(self.h_loc - loc_before) + sigma*w_t
        self.cur_loc = loc_before + delta*self.v_t
        # --->当新位置超过社区边界时，考虑反射性边界<----
        x_out = False
        y_out = False
        if self.cur_loc[0] >= size[0]*1000 or self.cur_loc[0] <= 0:
            # 当横坐标越界时
            x_out = True
        if self.cur_loc[1] >= size[1]*1000 or self.cur_loc[1] <= 0:
            # 当纵坐标越界时
            y_out = True
        # 进行不同情况下的越界判断，并进行运动反弹
        if x_out == True and y_out == True:
            # x 和 y 同时越界
            new_vt = np.array([-self.v_t[0], -self.v_t[1]])
        elif x_out == True and y_out == False:
            # 仅有 x 越界了
            new_vt = np.array([-self.v_t[0], self.v_t[1]])
        elif x_out == False and y_out == True:
            # 仅有 y 越界了
            new_vt = np.array([self.v_t[0], -self.v_t[1]])
        elif x_out == False and y_out == False:
            # 没有发生越界时
            new_vt = self.v_t
        
        # 重新赋值速度和位置
        self.cur_loc = loc_before + delta*new_vt
        self.v_t = new_vt
        # 避免后续计算位数爆炸，把小数位舍去
        self.cur_loc = np.array([int(self.cur_loc[0]), int(self.cur_loc[1])])
        self.v_t = np.array([int(self.v_t[0]), int(self.v_t[1])])
        # 将新的速度和位置数据存入
        self.v_array.append(self.v_t)
        self.mot_track.append(self.cur_loc)
    
    # 当感染暴露时间超过阈值，agent开始在每一个时间t以概率患病
    # t: 当前的时刻
    # P_I: 感染的几率
    # P_F: 感染为FT的几率
    def agent_infected(self, t, P_I, P_F):
        # 判断当前时刻agent是否患病
        result_inf = prob2num([0, 1], [1-P_I, P_I])
        # print('P_I: ', result_inf)
        # 当概率判断结果为0时，agent在该时刻未患病
        if result_inf == 0: 
            self.is_alive = True
            self.is_healthy = True
            self.is_infected = False
        # 当概率判断结果为1时，该agent在该时刻已经患病，进一步判断是FT还是NFT
        elif result_inf == 1:
            self.is_healthy = False
            self.is_infected = True
            # 记录被感染时刻(感染刚刚发生时不增加感染时长)
            self.t_infected = t
            # 判断当前的感染者是NFT还是FT
            result_FT = prob2num([0, 1], [1-P_F, P_F])
            # 当前感染者是NFT
            if result_FT == 0:
                self.is_FT = False
                print('-->>ID-', self.ID, ' is NFT!')
            # 当前感染者是FT
            elif result_FT == 1:
                self.is_FT = True
                print('-->>ID-', self.ID, ' is FT!')
    
    # 当NFT患者患病时间积累至阈值时，判断当前时刻NFT患者是否康复
    # t：当前时刻
    # P_R: 康复的几率
    def NFT_recovery(self, t, P_R):
        # 以一定概率判断是否康复
        result_recovery = prob2num([0, 1], [1-P_R, P_R])
        # 当agent未康复时
        if result_recovery == 0:
            self.is_infected = True
            self.is_healthy = False
            self.is_alive = True
            self.is_immune = False
            self.T_infected = self.T_infected + 1
        # 当agent康复时,agent被感染过康复is_infected会变为False
        elif result_recovery == 1:
            self.is_alive = True
            self.is_infected = False
            self.is_healthy = True
            self.t_rec_dead = t
            self.is_immune = True
            print('ID-', self.ID, ' is Recovery')

    # 当FT患者患病时间累积至阈值时，判断当前时刻FT患者是否死亡
    # t: 当前时刻
    # P_D: 死亡的几率
    def FT_dead(self, t, P_D):
        # 以一定概率判断是否会死亡
        result_dead = prob2num([0, 1], [1-P_D, P_D])
        self.is_infected = True
        self.is_FT = True
        self.is_healthy = False
        self.is_immune = False
        # 当agent未死亡时
        if result_dead == 0:
            self.is_alive = True
            self.T_infected = self.T_infected + 1
        # 当agent死亡时，不计算在被感染的人数
        elif result_dead == 1:
            self.is_alive = False
            self.is_infected = False
            self.t_rec_dead = t
            print('ID-', self.ID, ' is Dead')


# 定义模拟中涉及的传染病infection类
class infection(object):
    # 初始化某种传染病的方法
    # is_fatal: 该类疾病是否是致命性感染疾病
    # r_0： 该传染病得以传播的最大agent间距，以 m 为单位计算。
    # Tau_0: 该传染病得以传播的最小暴露时间，以 小时 为单位计算。
    # P_I： 某个agent感染该类传染病的概率
    # P_F： 若是致命性传染病，则感染患者中成为FT的概率
    # T_R： NFT患者从疾病中康复所需的最短时间（康复期），以 日 为单位计算。
    # P_R:  NFT患者在康复期后康复的概率
    # T_D： FT患者因疾病死亡所需的最短时间（死亡期）， 以 日 为单位计算。
    # P_D： FT患者在死亡期后死亡的概率
    def __init__(self, is_fatal, r_0, Tau_0, P_I, P_F, T_R, P_R, T_D, P_D):
        self.is_fatal = is_fatal
        self.r_0 = r_0
        self.Tau_0 = Tau_0
        self.P_I = P_I
        self.P_F = P_F
        self.T_R = T_R
        self.P_R = P_R
        self.T_D = T_D
        self.P_D = P_D

# 根据社区规模和居住单元数量生成相应的居住单元坐标
# 方便计算起见，所有单位以m计算，1 km = 1000 m
# h: 居住单元的数量，以9、25、49、81等完全平方数为例，开平方后为奇数
# size: 社区的规模，以英里为单位，通常是标准的正方形
# interval: 居住单元彼此间的间隔
def create_housing_units(h, size, interval):
    # 把生成的居住单元保存在一个数组里
    h_array = []
    h_all = []
    # 将尺寸转换为英尺计量
    x_len = size[0]*1000
    y_len = size[1]*1000
    interval_len = interval*1000
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

# 根据社区的情况，批量生成实验用的agent
# 此处同样以英里为输入单位，换算为英尺
# h_all: 所有的居住单元坐标
# N: 需要生成的agent数量
# size: 社区的规模，方便生成随机的初始位置
# sco: 生成的agent分布在社区周围多少距离的范围内，以 公里km 为单位计算
# p_0: 服从管理的社区人数比例
# random_type: true-所属住所及初始位置随机设置没有约束；false-每个住所agent数量一致，agent约束分布在所属住所附近
def create_agents(h_all, N, size, sco, p_0, random_type):
    # 遵从限制的人数，取整
    obey_num = int(N * p_0)
    # 以一个数组保存所有的agents对象
    agent_all = []
    # 将社区规模转变为初始位置的随机范围
    x = size[0]*1000
    y = size[1]*1000
    # 将范围换算为一个以m度量的距离
    sco_feet = sco*1000
    # 初始化一个数组，登记每一个住所内居住了多少agent
    h_agents_num = []
    for num in range(len(h_all)):
        h_agents_num.append(0)
    # random_type = true 随机分配agent的住所，各个住所的分配agent人数不同，agent也不一定能分配在所属住所附近
    # random_type = flase 平均分配agent的住所，每个住所分配的agent人数保持相同，agent分配在所属住所的附近
    # 根据输入的N批量生成agent对象,ID从0开始编号 [方便以ID为0的agent作为感染起始点]
    for ID in range(0, N):
        if random_type == True:
            # 生成一个h_index从h_all随机挑选一个居住单元分配给agent
            h_index = random.randint(0, len(h_all)-1)
            h_ID = h_index
            h_loc = h_all[h_index]
            h_agents_num[h_index] = h_agents_num[h_index] + 1
            # 在社区范围内随机生成位置(避免生成在边界附近，范围适当缩小)
            rand_x = random.randint(100, x-100)
            rand_y = random.randint(100, y-100)
            rand_loc = np.array([rand_x, rand_y])
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
                    rand_x = random.randint(h_loc[0]-sco_feet, h_loc[0]+sco_feet)
                    rand_y = random.randint(h_loc[1]-sco_feet, h_loc[1]+sco_feet)
                    rand_loc = np.array([rand_x, rand_y])
                    break
        # 当ID<obey_num 时，认为该agent遵守封锁
        if ID < obey_num:
            # 初始化agent
            new_agent = agent(rand_loc, h_ID, h_loc, ID, True)
        elif ID >= obey_num:
            new_agent = agent(rand_loc, h_ID, h_loc, ID, False)
        # 加入数组中保存
        agent_all.append(new_agent)

    # 返回生成的所有agent和住所分配情况
    return agent_all, h_agents_num

# 计算感染率、易感（正常）率、康复率、死亡率
# agent_all: 当前某一时刻的agent队列
# I_array: 感染率队列
# S_array: 易感率队列
# R_array: 康复路队列
# D_array: 死亡率队列
# N: 总人数
def cal_ISRD(agent_all, I_array, S_array, R_array, D_array, N):
    I_num = 0
    R_num = 0
    D_num = 0
    for agent in agent_all:
        # 感染的人数统计（既没有康复也没有死亡）
        if agent.is_infected == True and agent.is_alive == True:
            I_num = I_num + 1    
        # 康复的人数统计(获得免疫力的即是康复者)
        if agent.is_immune == True:
            R_num = R_num + 1
        # 死亡的人数统计
        if agent.is_alive == False:
            D_num = D_num + 1
    # 全部计算完后计算S_num
    S_num = N-I_num-R_num-D_num
    # 存入队列中
    I_array.append(I_num)
    S_array.append(S_num)
    R_array.append(R_num)
    D_array.append(D_num)
    # 返回结果
    return I_array, S_array, R_array, D_array

# 进行agent之间的接触判断，必要时计算两个agent之间的距离
# 若两个agent均未感染或都已经感染，则无需计算距离
# 若两个agent中有一名是感染者，另一名未感染且不具有免疫力，则计算距离
#   如果二者距离小于等于r_0，则累计未感染者的暴露时间
#   如果二者距离大于r_0， 则无任何操作
# 通过agent的is_change控制暴露时间的累计，目前版本下，跟多民暴露者接触仅累计一次暴露时间
# agent_all: 保存所有agent信息的队列
# r_0: 最大接触距离
def agents_touch(agent_all, r_0):
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
                agent_dis = math.sqrt(int(math.pow(abs(loc_1[0]-loc_2[0]), 2) + math.pow(abs(loc_1[1]-loc_2[1]), 2)))
                # print(loc_1, ' ', loc_2, ' ', agent_dis)
                # 当间距小于等于最大感染距离且该agent之前未被累加暴露时间
                if agent_dis <= r_0 and agent_comb[1].is_change == False:
                    agent_comb[1].t_0 = agent_comb[1].t_0 + 1
                    agent_comb[1].is_change = True
                    # print('----ID:', agent_comb[1].ID, ', Total t: ', agent_comb[1].t_0)
                continue
            # 同上
            elif agent_comb[1].is_infected == True and agent_comb[0].is_infected == False and agent_comb[0].is_immune == False:
                loc_1 = agent_comb[0].cur_loc
                loc_2 = agent_comb[1].cur_loc
                agent_dis = math.sqrt(int(math.pow(abs(loc_1[0]-loc_2[0]), 2) + math.pow(abs(loc_1[1]-loc_2[1]), 2)))
                # print(loc_1, ' ', loc_2, ' ', agent_dis)
                if agent_dis <= r_0 and agent_comb[0].is_change == False:
                    agent_comb[0].t_0 = agent_comb[0].t_0 + 1
                    agent_comb[0].is_change = True
                    # print('----ID:', agent_comb[0].ID, ', Total t: ', agent_comb[0].t_0)
                continue

# 计算当前agent是否靠近其所在住所
# agent实例
# 住所是一个范围，在此范围内都能算回到住所，以英尺计算
# 在住所范围内返回true，否则返回false
def at_home(agent, scope):
    # 获得agent的当前位置和所属的居住地位置
    loc_1 = agent.cur_loc
    loc_2 = agent.h_loc
    # 计算二者间的距离
    # print('cur_loc: ', loc_1, '  h_loc: ', loc_2)
    dis = math.sqrt(math.pow(abs(loc_1[0]-loc_2[0]), 2) + math.pow(abs(loc_1[1]-loc_2[1]), 2))
    if dis <= scope:
        return True
    else :
        return False

# 初始感染者/零号病人的感染
# t: 当前时刻
# P_F: 感染为FT的概率
# 零号病人必定感染，且感染按照概率分为FT与NFT
# FT与NFT都存在一定风险，即：传播还未开始便痊愈或死亡
# zero-type：控制初始感染者的生成方式
#   0：以ID = 0的agent为初始感染者
#   1：选取出整个agent群体中与其余agent距离合计最小的agent作为初始感染者
#   2：选取出整个agent群体中与其余agent距离小于感染范围最多的agent作为初始感染者
def zero_infected(zero_type, t, P_F, r_0, agent_all):
    print('Zero Infected Type Num:', zero_type)
    if zero_type == 0 or zero_type == 3:
        for agent in agent_all:
            if agent.ID == 0:
                # 调整健康标签
                agent.is_alive = True
                agent.is_infected = True
                agent.is_healthy = False
                agent.is_immune = False
                # 记录感染时刻，累积感染时长
                agent.t_infected = t
                agent.T_infected = agent.T_infected + 1
                # 判断感染结果
                result_FT = prob2num([0, 1], [1-P_F, P_F])
                # 零号感染者是NFT
                if result_FT == 0:
                    agent.is_FT = False
                    print('---> Zero is NFT <---')
                # 零号感染者是FT
                elif result_FT == 1:
                    agent.is_FT = True
                    print('---> Zero is FT <---')
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
                agent.is_alive = True
                agent.is_infected = True
                agent.is_healthy = False
                agent.is_immune = False
                # 记录感染时刻，累积感染时长
                agent.t_infected = t
                agent.T_infected = agent.T_infected + 1
                # 判断感染结果
                result_FT = prob2num([0, 1], [1-P_F, P_F])
                # 零号感染者是NFT
                if result_FT == 0:
                    agent.is_FT = False
                    print('---> Zero is NFT <---')
                # 零号感染者是FT
                elif result_FT == 1:
                    agent.is_FT = True
                    print('---> Zero is FT <---')
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
                agent.is_alive = True
                agent.is_infected = True
                agent.is_healthy = False
                agent.is_immune = False
                # 记录感染时刻，累积感染时长
                agent.t_infected = t
                agent.T_infected = agent.T_infected + 1
                # 判断感染结果
                result_FT = prob2num([0, 1], [1-P_F, P_F])
                # 零号感染者是NFT
                if result_FT == 0:
                    agent.is_FT = False
                    print('---> Zero is NFT <---')
                # 零号感染者是FT
                elif result_FT == 1:
                    agent.is_FT = True
                    print('---> Zero is FT <---')
                break

if __name__ == '__main__':
    # 实例化一个用于做实验的社区(以英里为单位输入数据，会换算为英尺)
    # N, h, size, is_lockdown, T_0, is_remove, T_1, p_0
    demo_community = community(4000, 9, [0.3,0.3], True, 100, True, 300, 0.98)
    # 实例化一个实验涉及的传染病
    # is_fatal, r_0, Tau_0, P_I, P_F, T_R, P_R, T_D, P_D
    demo_infection = infection(True, 2, 2, 0.05, 0.1, 7, 0.001, 7, 0.1)
    # 根据相关信息批量生成实验社区中各个居住单元的坐标(以英里为单位输入数据，会换算为英尺)
    # h, size, interval 
    h_all = create_housing_units(demo_community.h, demo_community.size, 0.05)
    # print(h_all)
    # 批量生成实验社区中的agent实例
    # h_all, N, size, p_0
    agent_all, h_agents_num = create_agents(h_all, demo_community.N, demo_community.size, 0.05, demo_community.p_0, False)
    print('Houseunits info:', h_agents_num)
    # 所需环境全部生成后，进行实验模拟
    # t：实验模拟的总时长（总迭代次数） 单位为小时
    T = 2400
    # 改为按分钟移动
    T_min = 1
    # 将封锁开始及结束的时间转变为小时计算
    T_0_h = demo_community.T_0 * 24
    T_1_h = demo_community.T_1 * 24
    # NFT和FT的康复时间和死亡时间也转变为小时计算
    T_D_h = demo_infection.T_D * 24
    T_R_h = demo_infection.T_R * 24
    # 初始化agent运动模型所需的各种参数
    # sigma要换算为英尺每小时
    # delta没有做要求，此处以1计算
    alpha = 0.2
    mu = 1
    sigma = 10
    delta = 10
    # 初始感染者的生成模式
    zero_type = 3
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
                for minu in range(T_min):
                    # 第一分钟开始初始感染
                    if minu == 0:
                        zero_infected(zero_type, t, demo_infection.P_F, demo_infection.r_0, agent_all)
                    else:
                        # 进行agent的集体彼此接触判定
                        agents_touch(agent_all, demo_infection.r_0)
                        # 每分钟进行一次移动
                        for agent in agent_all:
                            agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                    # 统计初始ISRD情况
                    I_array, S_array, R_array, D_array = cal_ISRD(agent_all, I_array, S_array, R_array, D_array, demo_community.N)
            # t >= 1 之后的所有情况
            if t >= 1:
                # 重置所有的暴露时间是否累加的控制标签
                for agent in agent_all:
                    agent.is_change = False

                # 进行感染者感染/恢复/死亡
                for agent in agent_all:
                    if agent.is_alive == True:
                        # 先进行感染判定
                        # 如果是未感染者，但暴露时间超过Tau_0且未被感染、未有免疫力、存活，则开始该agent的感染判定
                        if agent.t_0 >= demo_infection.Tau_0 and agent.is_infected == False and agent.is_immune == False:
                            agent.agent_infected(t, demo_infection.P_I, demo_infection.P_F)
                        # 如果是FT感染者，判断累计病程时间，如果超过T_D，则进行该agent的死亡判定
                        if agent.is_FT == True:
                            if agent.T_infected <= T_D_h:
                                agent.T_infected = agent.T_infected + 1
                            if agent.T_infected > T_D_h:
                                agent.FT_dead(t, demo_infection.P_D)
                        # 如果是NFT感染者，判断累计病程时间，如果超过T_R，则进行该agent的康复判定
                        if agent.is_FT == False and agent.is_infected == True:
                            if agent.T_infected <= T_R_h:
                                agent.T_infected = agent.T_infected + 1
                            if agent.T_infected > T_R_h and agent.is_immune == False:
                                agent.NFT_recovery(t, demo_infection.P_R)

                # 移动及接触判定以分钟为单位
                for minu in range(T_min):
                    # 每分钟都进行agent集体彼此间接触情况判定,增加相应的暴露时间
                    agents_touch(agent_all, demo_infection.r_0)
                    # 判定完接触情况后，遍历每一个agent进行移动
                    for agent in agent_all:
                        # 仅处理尚且存活的单位
                        if agent.is_alive == True:
                            # 进行agent移动(死亡agent不再进行移动)
                            # 在封锁之前，所有agent自由移动
                            if t < T_0_h:
                                agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                            # 在封锁期内，绝大部分agent向houseunit移动，极少部分agent继续自由移动
                            elif t >= T_0_h and t <= T_1_h:
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
                            elif t > T_1_h: 
                                # 无论agent是否在住所内，无论agent是否服从封锁，只要agent存活，就开始自由移动
                                agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
                    # 计算这一时刻的各类数值
                    I_array, S_array, R_array, D_array = cal_ISRD(agent_all, I_array, S_array, R_array, D_array, demo_community.N)
            

    # 查看ISRD曲线
    # t_array = []
    # for t in range(T*T_min):
    #     t_array.append(t)
    # plt.axis([0, len(t_array), 0, demo_community.N])
    # plt.plot(t_array, I_array ,"r-", linewidth=3, label='Infected')
    # plt.plot(t_array, S_array ,"b-", linewidth=3, label='Susceptible')
    # plt.plot(t_array, R_array ,"g-", linewidth=3, label='Recovered')
    # plt.plot(t_array, D_array ,color='black', linestyle="-", linewidth=3, label='Dead')
    # plt.axvline(T_0_h, linestyle='--', color='orange', linewidth = 2, label='Lock Down Start')
    # plt.axvline(T_1_h, linestyle='--', color='orchid', linewidth = 2, label='Lock Down End')
    # plt.xlabel('Time(hours)')
    # plt.ylabel('# of People')
    # plt.legend()
    # plt.show()


    # 查看单一agent运动路线图
    # x_value = []
    # y_value = []
    # h_x = []
    # h_y = []
    # agent = agent_all[0]
    # for loc in agent.mot_track:
    #     x_value.append(loc[0])
    #     y_value.append(loc[1])
    # for h in h_all:
    #     h_x.append(h[0])
    #     h_y.append(h[1])
    # s_x = agent.mot_track[0][0]
    # s_y = agent.mot_track[0][1]
    # e_x = agent.mot_track[-1][0]
    # e_y = agent.mot_track[-1][1]
    # plt.axis([0, 1760*demo_community.size[0], 0, 1760*demo_community.size[1]])
    # plt.plot(x_value, y_value,"b--",linewidth=1)
    # plt.scatter(x_value, y_value, s=5, c="red", marker='o')
    # plt.scatter(h_x, h_y, s=60, c="w", marker='s', edgecolors='black')
    # plt.scatter(s_x, s_y, s=40, c='blue', marker='*')
    # plt.scatter(e_x, e_y, s=40, c='green', marker='*')
    # plt.show()

    # 查看单一agent每个小时的位移速度
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


    # 查看初始分布与后续分布图
    # x_value = []
    # y_value = []
    # zero_x = []
    # zero_y = []
    # for agent in agent_all:
    #     if agent.ID == 0:
    #         zero_x.append(agent.cur_loc[0])
    #         zero_y.append(agent.cur_loc[1])
    #     else:
    #         x_value.append(agent.cur_loc[0])
    #         y_value.append(agent.cur_loc[1])
    
    # plt.scatter(zero_x, zero_y, s=40, c='black', marker='*')
    # plt.scatter(x_value, y_value, s=20, c="red", marker='o')
    # plt.show()


    # for times in range(240):
    #     for agent in agent_all:
    #         agent.change_v_loc(alpha, mu, sigma, delta, demo_community.size)
    
    # x_value = []
    # y_value = []
    # zero_x = []
    # zero_y = []
    # for agent in agent_all:
    #     if agent.ID == 0:
    #         zero_x.append(agent.cur_loc[0])
    #         zero_y.append(agent.cur_loc[1])
    #     else:
    #         x_value.append(agent.cur_loc[0])
    #         y_value.append(agent.cur_loc[1])
    
    # plt.scatter(zero_x, zero_y, s=40, c='black', marker='*')
    # plt.scatter(x_value, y_value, s=20, c="red", marker='o')
    # plt.show()
    
    
    
