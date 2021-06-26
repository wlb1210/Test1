import numpy as np
import gym
from gym import spaces
import scipy
import math
import scipy.io as scio
import copy  # 深拷贝
import matplotlib.pyplot as plt
#from matplotlib.matlab import *
import pylab as pl
import tensorflow as tf
import random
from collections import deque  # 作经验池
import heapq  # 用于查找数组中最大的几个元素的位置
import pandas as pd
import heapq

# 7.28.1,与7.26.1无差，只是把reward吞吐量的系数去掉，reward直接等于吞吐量，保存模型到savednetworks12
# 7.26.1,7.23.1设置有误，7.26.1重新设置重新运行，保存模型到savednetworks11
# 7.23.1，在7.22.1修改，done为false，添加了reward=0，（连续两次选到同样的），流量下泄改（之前有错），保存模型到savednetworks10
# 7.22.1，在7.19.1修改，37选10，data5时隙，输入4时隙，动作空间box（37），reward吞吐量，在step函数设置done为false，不重置状态。效果有提升。
# 7.21,在7.19.1基础上修改，在step函数加了流量下泄后第一时隙全为0的时候设置done，效果不好，弃用
# 版本7.19.1,37选10，data5时隙，输入4时隙，动作空间box（37），没有训练，测试部分有改，加了画reward和画吞吐量。
# 版本7.16.1，37选10，data5时隙，输入4时隙，5000周期，action空间multidiscrete类型，保存模型到saved_networks7/save1_10_5
# 版本7.15.1，37选10，data5时隙数据，输入4时隙，跑8000周期，保存模型到saved_networks6/save1_10_5.
# 版本7.12.2,37选10,100时隙数据，输入4个时隙，减小训练时间，跑了10000周期和8000周期分别存模型到saved_networks1和2
# 版本7.8.1， 在7.5.1基础上修改了计算SINR函数，里面计算干扰的距离，经纬度换成距离（最后除以700）。重新跑程序
# 版本7.8.1,37选10，reward吞吐量，box（37），10000周期，原500行数据，保存模型到save1_10_1
# 版本7.5.1,37选4，reward吞吐量。换37选多的方法，改变action空间为multidiscrete，选出10个，然后去掉重复的，从剩下小区随机选
# 版本7.4.1，开始做37选多，做了37选10：action空间设置为box（37），然后选最大的十个索引；跑了12000周期保存模型到save1_10.m中。
# 版本7.4.1,37选4，简化一些。跑5000周期保存模型到save1_4.m中。action空间是box（37）
# 版本7.3.1，在7.2.2基础上修改，状态是data四个时隙。
# 版本7.3.1, datanew是原来的500行数据,save分文件保存模型
# 版本7.2.2，在7.2.1基础上添加了另一种保存模型加载模型，改动地方在run函数,保存模型加载模型不改变数据测试有效果。状态设置是data两个时隙。
# 版本7.2.2，状态设置把time去掉，只用data（在observation改动shape，observation的定义改动），time参与运算，不作为状态
# 版本7.2.1添加了保存模型加载模型（两种方式），改动地方在pposgd
# 版本5.13.1（此版本作为24小时的基础版本）
# dataNew_24xiaoshi500,2个时隙，reward是时延，业务量数据无衰减版本
# 在5.12.3基础上，整理了程序
# 添加了打印统计量，便于查看24小时的统计量变化
# cur_it_r: -80.59052999999993
# cur_ep_cumultime: 26863510.0
# cur_ep_throughput: 1286511
# cur_ep_everage_cumultime: 52467.79296875
# cur_ep_everage_throughput: 2512.716796875
# 版本5.12.3
# 在4.26.2基础上修改observation，shape=[148]，Data和Time两个时隙，37*2*2
# 全程处理的是Data和Time，Data和Time没有动，还是之前的一行一行拼接，[37,None]
# 改动地方是加了一个预处理函数，使Data和Time的维度由[37，None]变为[37*2]
# 改动的效果与5.12.1和5.12.2效果相同
# 程序稍微整理了一下
# 版本4.26.2
# 4.26,在4.24.3（改了波束位置和业务量分布，多选一，没改observation）的基础上修改，修改处：
# NB = -118.5，Beam_Width = 1


# 卫星参数，原
# beam_radius = 1    # 波束半径1m
# beam_round = 3     # 波束的圈数
# P_tot = 33         # dBW 总卫星发射功率
# Num_Choose_Channel = 1
# Ant_Type = 'ITU-R S672-4 Circular'  #GEO卫星，圆形单馈源天线'ITU-R S672-4 Circular', 'ATaD_p389_n1'
# Beam_Width = 2

# 卫星场景统计初版
# Satellite segment parameter
lonS = 20        # [deg] 卫星经度
latS=0          #  [deg] 卫星纬度，赤道上方
altS = 35786    # [km],卫星轨道高度GEO
f=20            # GHz，ka工作频段
Band_tot=500    # MHZ, 系统总的可用带宽
Ant_Type='ITU-R S672-4 Circular'   # GEO卫星，圆形单馈源天线'ITU-R S672-4 Circular', 'ATaD_p389_n1'
G_T=52                                # dBi，卫星天线发射功率增益
L_F=92.44+20*math.log10(altS*f)       # dB 自由空间传输损耗
R=6400                 # Km 地球半径
G_R=41.7                # dB , 地面终端接收天线增益
Pth=-110               # dBm ,终端的功率阈值
radius = 350           # [km],beam radius
P_tot=33                       # W,卫星星载总功率 33dBW
Beam_Width = 1  # 波束宽度

Num_Choose_Channel = 10

class BeamhoppingEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 2
                }

    def __init__(self):

        # self.action_space = gym.spaces.Box(low=0, high=0xff, shape=[37]) # 生成37维的数据，选其中10个最大的索引
        # self.action_space = gym.spaces.MultiBinary(37)

        self.action_space = gym.spaces.MultiDiscrete([37,37,37,37,37,37,37,37,37,37])  # 生成10个0-36的数
        # self.action_space = gym.spaces.Discrete(37)  # 37个小区选一个小区,0,1,2,...,36分别代表选择第0个，第1个，...，第36个小区
        # self.observation_space = gym.spaces.Box(low=0, high=0xFF, shape=[37])  # Box(37), 代表37个小区的分别的累计数据包到达量
        self.observation_space = gym.spaces.Box(low=0, high=0xFF, shape=[740])  # 37*4，Data:[37*4]，4个时隙

        self.beam_location = BeamhoppingEnv.init(self)

        # 统计量初始化
        self.throughput_all = 0
        self.every_cell_cumul_time = {}
        self.every_cell_cumul_time_average = {}
        for i in range(37):
            self.every_cell_cumul_time[i] = 0

        self._ = 0  # 计数
        # self.action_last = np.zeros(10)


############### 此函数输入action, 返回执行action之后的observation， reward，done。用来表示agent与环境的互动过程。
    def step(self, action, Data, Time):
        done = False    # done初始化为False


        # print('ac:',action)
        # 如果action是box类型需要对action处理
        # 处理1
        # action = np.abs(int(np.round((np.clip(action, -2, 2)) * 18)))
        # 处理2
        # action = int(np.round((np.clip(action,-2,2)+2)*9))
        # 处理3
        # mean_ob = sum(self.observation) / 37
        # a = (((np.clip(ac, -2, 2)) + 2) / 4) * mean_ob
        # delta = abs(a[0] - self.observation[0])  # a:(1,)中括号里有一个数。 s:(37,)
        # beam_index = 0
        # for i in range(36):
        #     if abs(a[0] - self.observation[i]) <= delta:
        #         delta = abs(a[0] - self.observation[i])
        #         beam_index = i
        # action = beam_index  # 这里的action是选出来的波束
        # 处理4:将0/1版换成索引版
        # occupy = []
        # for i in range(37):
        #     if ac[i]:
        #         index = i
        #         occupy.append(index)
        # action = occupy
        # 处理4：action是box时，选出值最大的10个索引
        # action=heapq.nlargest(10, range(len(ac)), ac.take)
        # 处理5：action是multidiscrete时，去掉选的重复的
        # action = ac
        # n = len(ac)
        # for i in range(0, n):
        #     for j in range(i + 1, n):
        #         if (action[i] == action[j]):
        #             # print(action[j], j)
        #             l = list(set(list(range(37))).difference(set(action)))
        #             action[j] = (random.sample(l, 1))[0]


        # print('action：',action)                                       # 将选择的小区打印

        # 执行动作：计算信道容量，进行流量下泄。
        SNR = Channel_SINR_Cal(action, self.beam_location)                 # 计算执行动作之后的信道SINR
        Spectrum_Efficiency = Capacity_beam_dvbs2(SNR)                     # 计算执行动作之后的频谱效率
        Spectrum_Efficiency = np.array(Spectrum_Efficiency)
        now_CH = Spectrum_Efficiency * 5000  # C = 频谱效率*带宽/数据包大小。数据包大小100K
        for i in range(len(now_CH)):
            now_CH[i] = int(now_CH[i])
        now_CH_disord = []
        for i in range(37):
            now_CH_disord.append(0)
        for i in range(len(now_CH)):
            now_CH_disord[action[i]] = now_CH[i] # 选多
        # for i in range(len(now_CH)):
        #     now_CH_disord[action] = now_CH[i]  # 选1
        now_CH_disord = np.array(now_CH_disord)
        now_CH_disord = now_CH_disord.reshape([-1, 1])
        # print("SNR:", SNR)
        # print("Spectrum_Efficiency:", Spectrum_Efficiency)
        # print("now_CH:", now_CH)
        # print("now_CH_disord:", now_CH_disord)
        # print("self.data", self.Data)

        # 流量下泄
        self.Data, self.Time, self.throughput = execute_action(Data, Time, now_CH_disord, action)
        self.Time = self.Time + 1
        # self.throughput_all = self.throughput_all + self.throughput  # 累计吞吐量
        # print("流量下泄后的Data,Time:", self.Data, self.Time)
        # print("throughput:", self.throughput)
        # print("throughput_all:", self.throughput_all)

        # reward
        # 吞吐量
        self.reward = self.throughput
        # 公平性
        self.cumul_time, self.cumul_time_average, self.every_cell_cumul_time, self.every_cell_cumul_time_average = self.Caculate_Time(
            self.Data, self.Time)
        self.reward2 = -(max(self.every_cell_cumul_time_average.values()) - min(self.every_cell_cumul_time_average.values()))
        # reward2 = -0.0000001*(max(self.delay37_average_all.values()) - min(self.delay37_average_all.values()))
        # self.reward = reward1+reward2
        # print("reward1", reward1)
        # print("reward2", reward2)

        # self.reward2 = 0   # 公平性
        # for i in range(0,37):    #连续4个action
        #     if 1 not in Action[i][-4]:
        #         self.reward2 = -10
        # self.reward = self.reward1 + self.reward2


        # reward,如果两个时隙都选中一样的波束，reward设置为0

        # 公平性
        # n = len(action)
        # if self._>0:  # 从第二次开始，如果这次动作和上次有相同，reward设置为0
        #     # print("action", action, self.action_last)
        #     for i in range(0, n):
        #         for j in range(i + 1, n):
        #             if (action[i] == self.action_last[j]): # 如果这一步的action和上一步action有相同元素，reward置为0
        #                 # print("00000000000000000000000000000000000000000000")
        #                 self.reward = 0
        # self.action_last = action
        # self._ += 1
        # # print("reward", self.reward)


        # reward，如果10个时隙前的数据还有没处理完的，reward设置为0
        # if self._ > 10:
        #     if any((np.transpose(self.Data))[0])==0:
        #         # print("*************************************************************************88888not")
        #         reward = 0

        # 下一步的Data,Time
        self.Data = np.concatenate([self.Data, self.data_arrival['A'][0]], axis=1)   # shape=[37,None]
        self.Time = np.concatenate([self.Time, np.zeros([37, 1])], axis=1)

        # # 每隔n步将以前的数据丢弃
        D = self.Data.shape[1]  # shape=[37,None]
        if D % 21 == 0:
            self.Data = np.delete(self.Data, 0, axis=1)
        T = self.Time.shape[1]  # shape=[37,None]
        if T % 21 == 0:
            self.Time = np.delete(self.Time, 0, axis=1)
        # print("隔n步丢弃后的Data,Time:", self.Data, self.Time)

        # 更新data_arrival['A']表
        self.data_arrival['A'] = np.insert( self.data_arrival['A'], 500,  self.data_arrival['A'][0],axis=0)  # 将新产生的数据即data_arrival['A']的第一行数据插到data_arrival['A']表中的最后一行
        self.data_arrival['A'] = np.delete( self.data_arrival['A'], 0, axis=0)      # 将data_arrival['A']表中的第一行删除

        # 下一步的observation
        # self.Data_T = np.transpose(self.Data) # shape=
        # self.observation = sum(self.Data_T[:]) # shape=
        self.Data_processed, self.Time_processed = self.Data_preproccess(self.Data, self.Time)  # [37*n]
        # self.observation = np.concatenate([self.Data_processed, self.Time_processed], axis=0)  # [37*2n]
        self.observation = self.Data_processed # 把time去掉
        # print("observation:", self.observation)

        # 统计量
        # print("下一步开始时的时延：")
        # self.cumul_time, self.cumul_time_average, self.every_cell_cumul_time = self.Caculate_Time(self.Data, self.Time)


        # reward
        # reward = -(max(self.cumul_time_average) - min(self.cumul_time_average))  # 时延公平性，平均累计等待时间的最大值和最小值的差
        # reward = -0.0000001*(max(self.delay37_average_all.values()) - min(self.delay37_average_all.values()))
        # reward = -0.1*self.cumul_time_average  # 波束分布的业务+量
        # reward = -0.000003 * self.cumul_time
        # self.reward = 0.002 * self.throughput  # 吞吐量
        # print("时延reward:", reward)
        # print("吞吐量reward:", reward1)
        # print("吞吐量：", self.throughput)
        # print("时延：", self.cumul_time)

        # done
        self._ += 1
        if self._ == 512:
            done = True

        return self.observation, self.reward, self.reward2, done, self.cumul_time, self.throughput, self.Data, self.Time


##########此函数用来对每个ep开始的状态进行重置
    def reset(self):
        # dataNew = 'dataNew_24xiaoshi500.mat'
        # dataNew = 'dataNew_5.mat'   # savednetworks1-savednetworks3-save1.m
        dataNew = 'dataNew_yuan500_05.mat'       # savednetworks3-save2.m
        self.data_arrival = scio.loadmat(dataNew)
        self.Data = self.data_arrival['A'][-1] # shape=[37,None]
        self.Time = np.zeros([37, 1])
        # print("dataNew", self.data_arrival)
        # self.Data = (np.transpose(self.data_arrival['A']))[0]  # Data0.shape=(37,4)
        # print("Data:", self.Data, self.Data.shape)
        # self.Time = np.zeros([37, 4])
        # Time00 = np.arange(4)
        # for i in range(37):
        #     self.Time[i] = Time00  # Time0.shape=(37,4)
        # print("Time0:", self.Time, self.Time.shape)

        # 初始observation
        # self.Data_T = np.transpose(self.Data) # shape=[None,37] 原
        # self.observation = sum(self.Data_T[:])  # shape = [37]
        self.Data_processed, self.Time_processed = self.Data_preproccess(self.Data, self.Time) # [37*n]
        # self.observation = np.concatenate([self.Data_processed, self.Time_processed], axis=0) # [37*2n]
        self.observation = self.Data_processed   # 把time去掉
        # print("reset_observation:", self.observation, self.observation.shape)

        # 统计量
        # print("reset的时延：")
        self.cumul_time, self.cumul_time_average, self.every_cell_cumul_time, self.every_cell_cumul_time_average = self.Caculate_Time(self.Data, self.Time)

        return self.observation, self.Data, self.Time

    # env.render()：显示一个窗口提示当前环境的状态，当调用时，可通过窗口看到算法是如何学习和执行的。
    # 在训练过程中，为了节约时间建议注释掉这条指令。
    # def render(self, mode='human'):
    #     return None

    def close(self):
        return None

    def Caculate_Time(self, Data, Time): # Data.shape:[37,None]
        self.cumul_time = calculate_time(Data, Time)  # 等待时间
        self.cumul_time_average = self.cumul_time / sum(sum(Data))  # 平均等待时间
        self.every_cell_cumul_time = calculate_every_cell_time(Data, Time)  # 37个小区的等待时间
        # print("self.every_cell_cumul_time", self.every_cell_cumul_time)
        # print("cumul_time:", self.cumul_time)
        # print("cumul_time_average:", self.cumul_time_average)
        # print("every_cell_cumul_time:", self.every_cell_cumul_time)
        # print("Data", Data)
        for i in range(37):

            self.every_cell_cumul_time_average[i] = self.every_cell_cumul_time[i]/sum(Data[i])
            if sum(Data[i])==0:
                self.every_cell_cumul_time_average[i] = 0
        # print('self.every_cell_cumul_time_average', self.every_cell_cumul_time_average)

        return self.cumul_time, self.cumul_time_average, self.every_cell_cumul_time, self.every_cell_cumul_time_average


    def init(self):
        # satellite evironment initial
        posi = pd.read_excel("beamcenterV1.xlsx")
        lat_beam = np.array(posi["lat"][0:37])  # 获取维度之维度值
        lon_beam = np.array(posi["lon"][0:37])  # 获取经度值
        beam_location = [lat_beam, lon_beam]
        return beam_location

    # 数据预处理。将数据设置成固定n行，不够的地方补0
    def Data_preproccess(self, Data, Time):  # [37,None]，变为[37*n]
        Data_shape = Data.shape
        self.Data_add = Data # [37,n]
        self.Time_add = Time

        if Data_shape[1] < 20:  # 如果None小于n，就在左补0。左数据表示以前，右数据表示最新的数据
            self.Data_add = np.concatenate([np.zeros([Data_shape[0], 20 - Data_shape[1]]), Data], axis=1)
            self.Time_add = np.concatenate([np.ones([Data_shape[0], 20 - Data_shape[1]]), Time], axis=1)
        elif Data_shape[1] > 20:  # 如果None大于n，截取后n，将以前的数据丢弃
            self.Data_add = Data[:, -20:]
            self.Time_add = Time[:, -20:]
        # # print("Data：", self.Data,self.Time)
        # # print("Data_add:", self.Data_add,self.Time_add)
        self.Data_processed = np.zeros(37*20) # [37*n]
        self.Time_processed = np.zeros(37*20)
        for i in range(20):
            for j in range(37):
                self.Data_processed[j+i*37] = self.Data_add[j][i]
                self.Time_processed[j+i*37] = self.Time_add[j][i]
        # print("processed:", self.Data_processed,self.Time_processed)
        return self.Data_processed, self.Time_processed




###########################################波束生成函数############################
# def Beam_position(beam_radius, round):
#     D = math.sqrt(3) * beam_radius  # 相邻波束中心的距离
#     beam_sum = 1 + 3 * round * (round + 1)
#     beam_location = np.zeros((2, beam_sum))
#
#     for i in range(1, round + 1):  # 遍历每圈，每圈生成的波束构成六边形
#         counter = 1 + 3 * i * (i - 1)
#         for j in range(1, 6 * i + 1):
#             if ((j - 1) % i == 0):  # 六边形顶点上的波束
#                 beam_location[0][counter + j - 1] \
#                     = i * D * math.cos((math.ceil(j / i) - 1) * (math.pi) / 3)
#                 beam_location[1][counter + j - 1] \
#                     = i * D * math.sin((math.ceil(j / i) - 1) * (math.pi) / 3)
#             else:
#                 beam_location[0][counter + j - 1] \
#                     = beam_location[0][counter + j - 2] + D * math.cos((math.ceil(j / i) + 1) * (math.pi) / 3)
#                 beam_location[1][counter + j - 1] \
#                     = beam_location[1][counter + j - 2] + D * math.sin((math.ceil(j / i) + 1) * (math.pi) / 3)
#
#     return beam_location

##############################信道信干燥比计算函数#######################################
# 37选1
def Channel_SINR_Cal(slice, beam_location):
#     f = 20  # 20GHz
#     d = 36000
#     # NB = -108.42  # 10*np.log10(K*T*B) dB 噪声功率 -108.42
#     NB = -118.5 # 根据卫星场景统计初版设置
#     P_beam = P_tot  # 每个波束分得的功率，37选1，即总功率
#     free_space_loss = 20 * np.log10(f) + 20 * np.log10(d) + 92.44  # 自由空间损耗 209.5
#     send_antenna_Gain = 41.6
#     receive_antenna_Gain = 41.7  # db
#     C_beam = P_beam - free_space_loss + send_antenna_Gain + receive_antenna_Gain  # -103.2dBW
#     SINR_beam = []
#     N_I = []
#     Capacity_beam = []
#     N_I = 10 * np.log10(10 ** (NB / 10))
#     SINR_beam.append(C_beam - N_I)
#
#     return SINR_beam
# 原版本
# def Channel_SINR_Cal(slice, beam_location):
# 37选1
#   f = 20  # 20GHz
    # # d = 36000
    # # NB = -108.42  # 10*np.log10(K*T*B) dB 噪声功率 -108.42
    # # P_beam = P_tot  # 每个波束分得的功率，37选1，即总功率
    # # free_space_loss = 20 * np.log10(f) + 20 * np.log10(d) + 92.44  # 自由空间损耗 209.5
    # # send_antenna_Gain = 41.6
    # # receive_antenna_Gain = 41.7  # db
    # # C_beam = P_beam - free_space_loss + send_antenna_Gain + receive_antenna_Gain  # -103.2dBW
    # # SINR_beam = []
    # # N_I = []
    # # Capacity_beam = []
    # # N_I = 10 * np.log10(10 ** (NB / 10))
    # # SINR_beam.append(C_beam - N_I)
    # #
    # # return SINR_beam
# 选多
    f = 20  # 20GHz
    d = 36000
    # NB = -108.42  # 10*np.log10(K*T*B) dB 噪声功率 -108.42
    NB = -118.5
    # print('NB=', NB)
    P_beam = P_tot - 10 * np.log10(Num_Choose_Channel)  # 每个波束分得的功率，37个选10个，将功总率除以10  =23dbW
    # print('P_beam', P_beam)
    free_space_loss = 20 * np.log10(f) + 20 * np.log10(d) + 92.44  # 自由空间损耗 209.5
    # print(" free_space_loss",  free_space_loss)
    # send_antenna_Gain = 41.6
    send_antenna_Gain = 52
    receive_antenna_Gain = 41.7  # db
    C_beam = P_beam - free_space_loss + send_antenna_Gain + receive_antenna_Gain  # -103.2dBW
    # print('C_beam', C_beam)
    action_index = slice
    x_beam_location_interfere = []
    y_beam_location_interfere = []
    for i in action_index:  # 选出波束的索引
        x_beam_location_interfere.append(beam_location[0][i])  # 选出的10个波束的x轴坐标
        y_beam_location_interfere.append(beam_location[1][i])  # 选出的10个波束的y轴坐标
    xy_beam_location_interfere = np.vstack((x_beam_location_interfere, y_beam_location_interfere))  # 将x y轴存为二维数组
    # print("xy_beam_location_interfere", xy_beam_location_interfere)
    len_beam_location_interfere = len(xy_beam_location_interfere[0])  # 10
    SINR_beam = []
    N_I = []
    Capacity_beam = []
    for i in range(len_beam_location_interfere):
        Capacity_beam.append(0)
    for i in range(len_beam_location_interfere):
        interfere_beam_location = np.array([np.delete(xy_beam_location_interfere, i, axis=1)])  # 计算干扰位置，删掉要计算的波束 #list转array
        # print("interfere_beam_location:", interfere_beam_location)
        # x_inter = interfere_beam_location[0][0] - np.tile(beam_location[0][i], len_beam_location_interfere - 1)
        # y_inter = interfere_beam_location[0][1] - np.tile(beam_location[1][i], len_beam_location_interfere - 1)
        # dis_inter = np.sqrt(x_inter ** 2 + y_inter ** 2)  # 计算干扰波束之间的距离
        x_inter = interfere_beam_location[0][0]
        y_inter = interfere_beam_location[0][1]
        # print("x_inter", x_inter)
        # print("y_inter", y_inter)
        # dis_inter = np.sqrt(x_inter ** 2 + y_inter ** 2)  # 计算干扰波束之间的距离
        dis_inter = (DistanceCalculate(x_beam_location_interfere[i], y_beam_location_interfere[i], x_inter, y_inter))/700
        # print("dis_inter", dis_inter)
        transmitting_antenna_Gain = TypeAngle(Ant_Type, dis_inter, Beam_Width)  # 卫星发射天线增益
        # print('发射天线增益',transmitting_antenna_Gain)
        Interfere_beam = P_beam - free_space_loss + send_antenna_Gain + receive_antenna_Gain + transmitting_antenna_Gain  # SINR=（卫星发生功率-自由空间损耗+最大发射天线增益+最大接收天线增益）+发送天线增益
        # print('干扰值',Interfere_beam)
        Interfere_sum = 10 * np.log10(sum(10 ** (Interfere_beam / 10)))  # dB
        # print('总干扰值', Interfere_sum)
        N_I = 10 * np.log10(10 ** (NB / 10) + 10 ** (Interfere_sum / 10))
        # print('N_I',N_I)
        SINR_beam.append(C_beam - N_I)
    # print('N_I', N_I)
    # print('SINR_beam', SINR_beam)
    # 香浓信道公式计算信道容量##########
    # Capacity_beam[i]=B*np.log2(1+np.array(10**((SINR_beam[i])/10)))  #香农容量 array
    # print('Capacity_beam',Capacity_beam)
    return SINR_beam

# 计算距离，计算SINR，功率分配里的
# #  Calculate the distance between the two places based on the latitude and longitude of the two places
def DistanceCalculate(lat1,lon1,lat2,lon2):
    distance = R * np.arccos(np.cos(lat1 * math.pi / 180) * np.cos(lat2 * math.pi / 180)
                 * np.cos((lon2 - lon1) * math.pi / 180) + np.sin(lat1 * math.pi / 180)
                 * np.sin(lat2 * math.pi / 180))
    return distance
# #  Calculate the SINR(Signal to Interference plus Noise Ratio) of beam
# def CalculateSINR(lat_beam,lon_beam,beam,frequency_subband,P,WEATHER):
#     BandNum = frequency_subband[beam]
#     InterferenceBeams = np.where(np.array(frequency_subband)==BandNum)
#     InterfereBeamList= list(InterferenceBeams[0])#返回行索引？
#     InterfereBeamList.remove(beam)  ##计算干扰，自己不算，所以要删除当前的波束
#     powVec=np.zeros(len(lat_beam))
#     for i in range(len(lat_beam)):
#         powVec[i]=P[i]
#     linkbuget=sat_link_pow_coefficient(lat_beam, lon_beam, beam, WEATHER)
#     # C=powVec[beam]*linkbuget/(4*math.pi*radius**2)
#     C=powVec[beam]*linkbuget
#     # N = Band_tot / color * No * pow(10, 6) * linkbuget
#     N = Band_tot / color * No * pow(10, 6)
#     interfere_pow_list=[]
#     for i in InterfereBeamList:
#         # dis=DistanceCalculate(lat_beam[beam], lon_beam[beam], lat_beam[i], lon_beam[i])
#         # interfere_pow_list.append(powVec[i]*sat_link_pow_coefficient(lat_beam,lon_beam,i,WEATHER)/(4*math.pi*dis**2))
#         interfere_pow_list.append(powVec[i]*sat_link_pow_coefficient(lat_beam,lon_beam,i,WEATHER)*0.016)
#     I=sum(interfere_pow_list)
#     return C/(I+N)


##########################################频谱效率计算函数#################################
def Capacity_beam_dvbs2(SNR):
    # 第一步，将SNR转为Es/N0
    SNR = np.array(SNR)
    # print(SNR)
    # for i in range(len(SNR)):
    SNR_real = 10 ** (SNR / 10)  # 将dB值转化为实数值
    roll_off = 0.35  # 滚降因子暂取0.35
    Rs2W = 1 / (1 + roll_off)  # 基于公式：W=Rs*(1+alpha)
    EsN0_real = SNR_real / Rs2W
    EsN0 = 10 * np.log10(EsN0_real)  # 转为db值 numpy.ndarry
    # print('EsN0', EsN0)
    # 第二步，根据标准中Es/N0映射为频谱效率
    EsN0_list = [-2.35, -1.24, -0.30, 1.00, 2.23, 3.10, 4.03, 4.68, 5.18, 6.20,
                 6.42, 5.50, 6.62, 7.91, 9.35, 10.69, 10.98, 8.97, 10.21, 11.03,
                 11.61, 12.89, 13.13, 12.73, 13.64, 14.28, 15.69, 16.05]
    Spectrum_efficiency_list = [0.490243, 0.656448, 0.789412, 0.988858, 1.188304, 1.322253, 1.487473, 1.587196,
                                1.654663, 1.766451,
                                1.788612, 1.779991, 1.980636, 2.228124, 2.478562, 2.646012, 2.679207, 2.637201,
                                2.966728, 3.165623,
                                3.300184, 3.523143, 3.567342, 3.703295, 3.951571, 4.119540, 4.397854, 4.453027]
    idx_abs = []
    for i in range(len(EsN0)):
        idx_abs.append(0)
    idx = []
    for i in range(len(EsN0)):
        idx.append(0)
    idx_min = []
    for i in range(len(EsN0)):
        idx_min.append(0)
    spectrum_efficiency = []
    EsN0_list = np.array(EsN0_list)
    for j in range(len(EsN0)):
        idx_abs[j] = (abs(EsN0_list - EsN0[j]))
        idx_abs[j] = idx_abs[j].tolist()
        idx[j] = min(idx_abs[j])  # float
        idx_min[j] = idx_abs[j].index(idx[j])
        spectrum_efficiency.append(Spectrum_efficiency_list[idx_min[j]])
    # print('spectrum_efficiency', spectrum_efficiency)
    return spectrum_efficiency

############################################时延计算函数#############################
def calculate_time(Data, Time):
    delay = sum(sum(Data * Time))
    return delay

##########################################小区分别的时延计算函数#################
# 计算37个小区分别的等待时延
def calculate_every_cell_time(Data, Time):
    every_cell_cumul_time = {}
    temp = Data * Time
    for i in range(37):
        every_cell_cumul_time[i] = sum(temp[i,:])
    return every_cell_cumul_time

#############################################流量下泄函数##############################
def execute_action(Data0, Time0, now_CH, action):  # 进行流量下泻
    Remain_CH = np.zeros([37, 1])
    Data = copy.deepcopy(Data0)  # shape=[37,None]
    Time = copy.deepcopy(Time0)
    sum_Data = np.sum(Data, axis=1, keepdims=True)

    for idx in action: # 多选多
        if sum_Data[idx][0] > now_CH[idx][0]:  # 如果小区的数据包总数大于信道容量
            temp = now_CH[idx][0]
            count = 0
            # for jdx in Data[idx][::-1]: # 倒序
            for jdx in Data[idx][::1]:  # 左边时隙是以前的，右边是新到的，先服务以前的。
                count = count + 1
                flag = jdx - temp
                if flag < 0:  # 说明没减完
                    Data[idx][-count] = 0
                    temp = temp - jdx
                else:
                    Data[idx][-count] = flag
                    break
        else:  # 如果小区数据包总数小于信道容量，下泄全部
            Remain_CH[idx] = now_CH[idx][0] - Data[idx][0]
            Data[idx][:] = 0
    # idx = action # 多选一
    # if sum_Data[idx][0] > now_CH[idx][0]:  # 如果小区的实时数据包总数大于信道容量
    #     temp = now_CH[idx][0]
    #     count = 0
    #     for jdx in Data[idx][::-1]:
    #         count = count + 1
    #         flag = jdx - temp
    #         if flag < 0:  # 说明没减完
    #             Data[idx][-count] = 0
    #             temp = temp - jdx
    #         else:
    #             Data[idx][-count] = flag
    #             break
    # else:  # 如果小区实时数据包总数小于信道容量，剩余的用于下泄非实时数据包
    #     Remain_CH[idx] = now_CH[idx][0] - Data[idx][0]
    #     Data[idx][:] = 0

    throughput = sum(sum(Data0 - Data))
    #######################################################
    # # 数组中最后若全是0，则删除那一列
    while (Data[0][-1] == 0) & (Data[1][-1] == 0) & (Data[2][-1] == 0) & (Data[3][-1] == 0) & \
            (Data[4][-1] == 0) & (Data[5][-1] == 0) & (Data[6][-1] == 0) & (Data[7][-1] == 0) & \
            (Data[8][-1] == 0) & (Data[9][-1] == 0) & (Data[10][-1] == 0) & (Data[11][-1] == 0) & \
            (Data[12][-1] == 0) & (Data[13][-1] == 0) & (Data[14][-1] == 0) & (Data[15][-1] == 0) & \
            (Data[16][-1] == 0) & (Data[17][-1] == 0) & (Data[18][-1] == 0) & (Data[19][-1] == 0) & \
            (Data[20][-1] == 0) & (Data[21][-1] == 0) & (Data[22][-1] == 0) & (Data[23][-1] == 0) & \
            (Data[24][-1] == 0) & (Data[25][-1] == 0) & (Data[26][-1] == 0) & (Data[27][-1] == 0) & \
            (Data[28][-1] == 0) & (Data[29][-1] == 0) & (Data[30][-1] == 0) & (Data[31][-1] == 0) & \
            (Data[32][-1] == 0) & (Data[33][-1] == 0) & (Data[34][-1] == 0) & (Data[35][-1] == 0) & \
            (Data[36][-1] == 0) & (Data.shape[1] > 1):
        Data = np.delete(Data, -1, axis=1)
        Time = np.delete(Time, -1, axis=1)

    return Data, Time, throughput

# 函数说明
# 其中，根据输入卫星波束天线类型，计算某卫星某波束对于某地球站方向的波束天线增益
# 根据，输入地球站天线类型，计算某卫星某波束对于某地球站方向的地球站天线增益
# 其他变量，参考所给的数据结构
# 输出结果，参考所给的数据结构：Gain_Sat_Tx
## 功能实现
def TypeAngle(Antenna_Type, Xmtr_Angle_OffAxis, beamwidth):
    off_axis_T=Xmtr_Angle_OffAxis  #输入偏轴角，单位：deg
    Gain_Sat_Tx=np.zeros(np.shape(Xmtr_Angle_OffAxis))
    if Antenna_Type=='ITU-R S672-4 Circular':    # GEO卫星，圆形单馈源天线；#FSS
        Gm=41.6  #最大天线增益
        fai_b=beamwidth/2 #波宽的一半
        a=2.88
        b=6.32
        Ls=-25
        l1=len(Xmtr_Angle_OffAxis)
        for i in range(l1):
            if (off_axis_T[i] < fai_b):
                Gain_Sat_Tx[i] = Gm
            elif(off_axis_T[i]>= fai_b and off_axis_T[i] <= a*fai_b):
                #if (off_axis_T[i] >= fai_b and off_axis_T[i] <= a * fai_b/fai_b):
                Gain_Sat_Tx[i] = Gm - 3 *(off_axis_T[i]/fai_b)**2
                #Gain_Sat_Tx[i] = Gm-3*(off_axis_T(off_axis_T[i]>=fai_b and off_axis_T[i]<= a*fai_b)/fai_b)**2     #######
            elif off_axis_T[i] > a*fai_b and off_axis_T[i]<=b*fai_b:
                Gain_Sat_Tx[i] = Gm+Ls
            elif off_axis_T [i]> b*fai_b:
                Gain_Sat_Tx[i]=max(Gm+Ls+20-25*math.log10(off_axis_T[i]), 0)
                #Gain_Sat_Tx[i]=max(Gm+Ls+20-25*math.log10(off_axis_T(off_axis_T[i]> b*fai_b)/fai_b), 0)
        Gain_Sat_Tx=Gain_Sat_Tx - Gm  #天线增益归一化
    elif Antenna_Type == 'ATaD_p389_n1':
    #该天线模型下，选择口径30m，载频2GHz，对应的波宽为0.3632度。

        fai = np.pi*30/(3e8/2e9)*np.sin(np.deg2rad(Xmtr_Angle_OffAxis))     #30m口径天线，2GHz载频#################
        Gmax = 10**(40/10)    #Max Gain=40dBi########
        min_resolution = np.sqrt(7e-8) #防止在0度的时候无法得出天线增益
        n=1
        #fai_l=len(fai)
        #Jnp1_fai=[]
        #for i in range(fai_l):  #未修改完成

        Jnp1_fai = scipy.special.jn(n + 1, abs(fai + min_resolution))
        #Jnp1_fai[i] = scipy.signal.bessel(n+1, abs(fai[i]+min_resolution))     ####贝塞尔函数besselj(第一类贝塞尔函数)
        gain = Gmax*(2**(n+1)*math.factorial(n+1)*Jnp1_fai/abs(fai+min_resolution)**(n+1))**2           ########^改为**
        gain_dB=10*np.log10(gain)
        Gain_Sat_Tx=gain_dB-40
    else:
        print('暂不支持此天线类型')
    return Gain_Sat_Tx

def Caculate_Time(Data, Time): # Data.shape:[37,None]
    cumul_time = calculate_time(Data, Time)  # 等待时间
    cumul_time_average = cumul_time / sum(sum(Data))  # 平均等待时间
    every_cell_cumul_time = calculate_every_cell_time(Data, Time)  # 37个小区的等待时间
    print("every_cell_cumul_time", every_cell_cumul_time)
    # print("cumul_time:", self.cumul_time)
    # print("cumul_time_average:", self.cumul_time_average)
    # print("every_cell_cumul_time:", self.every_cell_cumul_time)
    print("Data", Data)
    every_cell_cumul_time_average = []
    for i in range(37):
        every_cell_cumul_time_average[i] = every_cell_cumul_time[i]/sum(Data[i])
    print('every_cell_cumul_time_average', every_cell_cumul_time_average)

    return cumul_time, cumul_time_average, every_cell_cumul_time, every_cell_cumul_time_average







if __name__ == '__main__':
    env = BeamhoppingEnv()
    env.reset()
    env.step(env.action_space.sample())
    print(env.observation)
    env.step(env.action_space.sample())
    print(env.observation)
