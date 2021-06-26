# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np
import random
from collections import deque  # 作经验池
import copy  # 深拷贝
import heapq  # 用于查找数组中最大的几个元素的位置
import scipy.io as scio
import math
import TypeAngle
import GA_37xuan10

import matplotlib.pyplot as plt

# 添加了画图reward，和打印最后一周期的统计量
# 37选10，隔4步丢弃之前的Data和Time数据
# 种群中个体数量是600时，遗传算法迭代一次的时间为0.8＋s
# 种群中个体数量是10000时，遗传算法迭代一次的时间为19-20s

# 5.22：pop_size = 600，iter = 24，TIMES = 20，EPOCHS = 200，预估26-27个小时
# 5.24：去掉TIMES 和EPOCHS，只输入一个时刻的状态，用遗传算法迭代若干次得到一个动作，比较动作执行后的结果
# 5.24:改好程序后，遗传算法pop_size = 600，iter = 500

# 全局变量
EPOCHS = 1000  # 玩多少局,原1000
TIMES = 1000  # 每一局玩多少次,原1000
POOL_MEMORY = 5000  # 经验池容量
OBSERVE = 1  # 初始化运行这么多epoch次后开始随机采样训练网络
BATCH = 4  # minibatch的大小

INITIATE_EPSILON = 0.5  # 贪婪算法的参数,这里利用EXPORE进行参数下降
FINISH_EPSILON = 0.05
EXPORE = 500

GAMMA = 0.90  # bellman方程中的参数
# 选择策略方式
# 随机方式：random，每次随机选择
# 深度增强学习方式：dqn，利用神经网络选择动作
# 固定时长方式：fix_time
# max_K  选取最长的K个队列
STRATEGY = 'GA'
SAVE_STEP = 10  # 每隔多少次epoch保存一次训练参数
# MODEL = 'test'  # 原，模式，有‘train’和‘test’两种，train时每隔SAVE_STEP步保存一次训练参数，test时不保存参数
MODEL = 'train'

beam_radius = 1  # 波束半径1m
round = 3  # 波束的圈数
P_tot = 33  # dBW 总卫星发射功率
Num_Choose_Channel = 10
Ant_Type = 'ITU-R S672-4 Circular'  ##GEO卫星，圆形单馈源天线'ITU-R S672-4 Circular', 'ATaD_p389_n1'
Beam_Width = 2


# 卫星的多波束生成方式为按圈生成，通过给定波束半径和总共波束圈数，生成所有波束
def Beam_position(beam_radius, round):
    D = math.sqrt(3) * beam_radius  # 相邻波束中心的距离
    beam_sum = 1 + 3 * round * (round + 1)
    beam_location = np.zeros((2, beam_sum))

    for i in range(1, round + 1):  # 遍历每圈，每圈生成的波束构成六边形
        counter = 1 + 3 * i * (i - 1)
        for j in range(1, 6 * i + 1):
            if ((j - 1) % i == 0):  # 六边形顶点上的波束
                beam_location[0][counter + j - 1] \
                    = i * D * math.cos((math.ceil(j / i) - 1) * (math.pi) / 3)
                beam_location[1][counter + j - 1] \
                    = i * D * math.sin((math.ceil(j / i) - 1) * (math.pi) / 3)
            else:
                beam_location[0][counter + j - 1] \
                    = beam_location[0][counter + j - 2] + D * math.cos((math.ceil(j / i) + 1) * (math.pi) / 3)
                beam_location[1][counter + j - 1] \
                    = beam_location[1][counter + j - 2] + D * math.sin((math.ceil(j / i) + 1) * (math.pi) / 3)

    return beam_location


#####################信道容量计算##################
def Channel_SINR_Cal(slice, beam_location):
    f = 20  # 20GHz
    d = 36000
    NB = -108.42  # 10*np.log10(K*T*B) dB 噪声功率 -108.42
    #print('NB=', NB)
    P_beam = P_tot - 10 * np.log10(Num_Choose_Channel)  # 每个波束分得的功率，37个选10个，将功总率除以10  =23dbW
    #print('P_beam', P_beam)
    free_space_loss = 20 * np.log10(f) + 20 * np.log10(d) + 92.44  # 自由空间损耗 209.5
    send_antenna_Gain = 41.6
    receive_antenna_Gain = 41.7  # db
    C_beam = P_beam - free_space_loss + send_antenna_Gain + receive_antenna_Gain  # -103.2dBW
    #print('C_beam', C_beam)
    action_index = slice
    x_beam_location_interfere = []
    y_beam_location_interfere = []
    for i in action_index:  # 选出波束的索引
        x_beam_location_interfere.append(beam_location[0][i])  # 选出的10个波束的x轴坐标
        y_beam_location_interfere.append(beam_location[1][i])  # 选出的10个波束的y轴坐标

    xy_beam_location_interfere = np.vstack((x_beam_location_interfere, y_beam_location_interfere))  # 将x y轴存为二维数组
    # print(xy_beam_location_interfere)
    len_beam_location_interfere = len(xy_beam_location_interfere[0])  # 10
    SINR_beam = []
    N_I = []
    Capacity_beam = []
    for i in range(len_beam_location_interfere):
        Capacity_beam.append(0)
    for i in range(len_beam_location_interfere):
        interfere_beam_location = np.array(
            [np.delete(xy_beam_location_interfere, i, axis=1)])  # 计算干扰位置，删掉要计算的波束 #list转array
        x_inter = interfere_beam_location[0][0] - np.tile(beam_location[0][i], len_beam_location_interfere - 1)
        y_inter = interfere_beam_location[0][1] - np.tile(beam_location[1][i], len_beam_location_interfere - 1)
        dis_inter = np.sqrt(x_inter ** 2 + y_inter ** 2)  # 计算干扰波束之间的距离
        transmitting_antenna_Gain = TypeAngle.TypeAngle(Ant_Type, dis_inter, Beam_Width)  # 卫星发射天线增益
        # print('发射天线增益',transmitting_antenna_Gain)
        Interfere_beam = P_beam - free_space_loss + +send_antenna_Gain + receive_antenna_Gain + transmitting_antenna_Gain  # SINR=（卫星发生功率-自由空间损耗+最大发射天线增益+最大接收天线增益）+发送天线增益
        # print('干扰值',Interfere_beam)
        Interfere_sum = 10 * np.log10(sum(10 ** (Interfere_beam / 10)))  # dB
        # print('总干扰值', Interfere_sum)
        N_I = 10 * np.log10(10 ** (NB / 10) + 10 ** (Interfere_sum / 10))
        # print('N_I',N_I)
        SINR_beam.append(C_beam - N_I)
    #print('N_I', N_I)
    #print('SINR_beam', SINR_beam)
    # 香浓信道公式计算信道容量##########
    # Capacity_beam[i]=B*np.log2(1+np.array(10**((SINR_beam[i])/10)))  #香农容量 array
    # print('Capacity_beam',Capacity_beam)
    return SINR_beam


# DVB-S2模型，计算频谱效率####################
def Capacity_beam_dvbs2(SNR):
    # 第一步，将SNR转为Es/N0
    SNR = np.array(SNR)
    #print(SNR)
    # for i in range(len(SNR)):
    SNR_real = 10 ** (SNR / 10)  # 将dB值转化为实数值
    roll_off = 0.35  # 滚降因子暂取0.35
    Rs2W = 1 / (1 + roll_off)  # 基于公式：W=Rs*(1+alpha)
    EsN0_real = SNR_real / Rs2W
    EsN0 = 10 * np.log10(EsN0_real)  # 转为db值 numpy.ndarry
    #print('EsN0', EsN0)
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
    #print('spectrum_efficiency', spectrum_efficiency)
    return spectrum_efficiency  # ,beam_spectrum_efficiency


def weight_variable(shape):  # 定义层与层之间连接的权重矩阵变量
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)


def bias_variable(shape):  # 定义一层的偏置矩阵
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# x为输入图片，有四维，第一维为多少个patch，第二维为图片的高，第三维为图片的长，第四维为图片的通道个数（是1）
# W为卷积核，有四维，第一维为图片的高，第二维为图片的长，第三维为图片的通道个数，第四维为多少个卷积核
# 滑动步长strides中，表示对X的四个维度的滑动，其中只对图片的高和长进行滑动
def conv2d(x, W, stride1, stride2, pad="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, stride1, stride2, 1], padding=pad)


def createNetwork():
    # 数据网络
    W_Data_conv1 = weight_variable([1, 20, 1, 16])
    b_Data_conv1 = bias_variable([16])

    W_Data_conv2 = weight_variable([1, 5, 16, 32])
    b_Data_conv2 = bias_variable([32])

    # 定义数据输入层
    s_Data = tf.placeholder("float", [None, 37, 40, 1])

    # 第一层（卷积层）
    h_Data_conv1 = tf.nn.relu(conv2d(s_Data, W_Data_conv1, 1, 4) + b_Data_conv1)

    # 第二层（卷积层）
    h_Data_conv2 = tf.nn.relu(conv2d(h_Data_conv1, W_Data_conv2, 1, 2) + b_Data_conv2)

    # 第三层（全连接层） 先展开
    h_Data_conv3_flat = tf.reshape(h_Data_conv2, [-1, 5920])

    # 延时网络
    W_Time_conv1 = weight_variable([1, 20, 1, 8])
    b_Time_conv1 = bias_variable([8])

    W_Time_conv2 = weight_variable([1, 5, 8, 16])
    b_Time_conv2 = bias_variable([16])

    # 定义延时输入层
    s_Time = tf.placeholder("float", [None, 37, 40, 1])

    # 第一层（卷积层）
    h_Time_conv1 = tf.nn.relu(conv2d(s_Time, W_Time_conv1, 1, 4) + b_Time_conv1)

    # 第二层（卷积层）
    h_Time_conv2 = tf.nn.relu(conv2d(h_Time_conv1, W_Time_conv2, 1, 2) + b_Time_conv2)

    # 第三层（全连接层） 先展开
    h_Time_conv3_flat = tf.reshape(h_Time_conv2, [-1, 2960])

    # 合为一个
    s_flat = tf.concat([h_Data_conv3_flat, h_Time_conv3_flat], 1)

    W_fc1 = weight_variable([8880, 1024])
    b_fc1 = bias_variable([1024])

    W_fc2 = weight_variable([1024, 256])
    b_fc2 = bias_variable([256])

    W_fc3 = weight_variable([256, 37])
    b_fc3 = bias_variable([37])

    # 第三层（全连接层） （先展开，已完成，）再运算
    h_fc1 = tf.nn.relu(tf.matmul(s_flat, W_fc1) + b_fc1)

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # 输出层
    network_Action = tf.matmul(h_fc2, W_fc3) + b_fc3

    return s_Data, s_Time, network_Action  # s表示输入层的值，readout表示输出层的值，h_fc1为倒数第二层


# 假定小区数据包的到达率服从泊松分布,定义各小区到达率,产生请求的数据包个数
def gen_data(lam1=1861, lam2=49, lam3=2795, lam4=2126, lam5=1213, lam6=570, lam7=2167, lam8=570,
             lam9=2167, lam10=123, lam11=1739, lam12=2170, lam13=2582, lam14=2427, lam15=223, lam16=3145,
             lam17=2292, lam18=3190, lam19=3022, lam20=3104, lam21=365, lam22=1329, lam23=2444,
             lam24=2768, lam25=2915, lam26=2293, lam27=2159, lam28=343, lam29=408, lam30=1438, lam31=2577, lam32=2279,
             lam33=1932, lam34=3138, lam35=2370, lam36=3131, lam37=745):
    arrival1 = np.random.poisson(lam=lam1, size=1)  # 数据包大小100k
    arrival2 = np.random.poisson(lam=lam2, size=1)
    arrival3 = np.random.poisson(lam=lam3, size=1)
    arrival4 = np.random.poisson(lam=lam4, size=1)
    arrival5 = np.random.poisson(lam=lam5, size=1)
    arrival6 = np.random.poisson(lam=lam6, size=1)
    arrival7 = np.random.poisson(lam=lam7, size=1)
    arrival8 = np.random.poisson(lam=lam8, size=1)
    arrival9 = np.random.poisson(lam=lam9, size=1)
    arrival10 = np.random.poisson(lam=lam10, size=1)
    arrival11 = np.random.poisson(lam=lam11, size=1)
    arrival12 = np.random.poisson(lam=lam12, size=1)
    arrival13 = np.random.poisson(lam=lam13, size=1)
    arrival14 = np.random.poisson(lam=lam14, size=1)
    arrival15 = np.random.poisson(lam=lam15, size=1)
    arrival16 = np.random.poisson(lam=lam16, size=1)
    arrival17 = np.random.poisson(lam=lam17, size=1)
    arrival18 = np.random.poisson(lam=lam18, size=1)
    arrival19 = np.random.poisson(lam=lam19, size=1)
    arrival20 = np.random.poisson(lam=lam20, size=1)
    arrival21 = np.random.poisson(lam=lam21, size=1)
    arrival22 = np.random.poisson(lam=lam22, size=1)
    arrival23 = np.random.poisson(lam=lam23, size=1)
    arrival24 = np.random.poisson(lam=lam24, size=1)
    arrival25 = np.random.poisson(lam=lam25, size=1)
    arrival26 = np.random.poisson(lam=lam26, size=1)
    arrival27 = np.random.poisson(lam=lam27, size=1)
    arrival28 = np.random.poisson(lam=lam28, size=1)
    arrival29 = np.random.poisson(lam=lam29, size=1)
    arrival30 = np.random.poisson(lam=lam30, size=1)
    arrival31 = np.random.poisson(lam=lam31, size=1)
    arrival32 = np.random.poisson(lam=lam32, size=1)
    arrival33 = np.random.poisson(lam=lam33, size=1)
    arrival34 = np.random.poisson(lam=lam34, size=1)
    arrival35 = np.random.poisson(lam=lam35, size=1)
    arrival36 = np.random.poisson(lam=lam36, size=1)
    arrival37 = np.random.poisson(lam=lam37, size=1)
    return np.concatenate([[arrival1], [arrival2], [arrival3], [arrival4],
                           [arrival5], [arrival6], [arrival7], [arrival8],
                           [arrival9], [arrival10], [arrival11], [arrival12],
                           [arrival13], [arrival14], [arrival15], [arrival16],
                           [arrival17], [arrival18], [arrival19], [arrival20],
                           [arrival21], [arrival22], [arrival23], [arrival24],
                           [arrival25], [arrival26], [arrival27], [arrival28],
                           [arrival29], [arrival30], [arrival31], [arrival32],
                           [arrival33], [arrival34], [arrival35], [arrival36],
                           [arrival37]], axis=0)


# 信道容量求法






def calculate_time(Data, Time):
    return sum(sum(Data * Time))


# 计算37个小区中部分小区的等待时延
def calculate_every_cell_time(Data, Time):
    temp = Data * Time
    cumul_time11 = sum(temp[0, :])
    cumul_time22 = sum(temp[1, :])
    cumul_time33 = sum(temp[2, :])
    cumul_time44 = sum(temp[3, :])
    # cumul_time55 = sum(temp[4, :])
    # cumul_time66 = sum(temp[5, :])
    # cumul_time77 = sum(temp[6, :])
    # cumul_time88 = sum(temp[7, :])
    return cumul_time11, cumul_time22, cumul_time33, cumul_time44#, cumul_time55, cumul_time66, cumul_time77, cumul_time88


# 预处理一组数据使其可以送入神经网络中, 如果长度不到20,则前面补0,长度超过20,截取，
def preproccess_alone(Data0, Time0):  # 神经网络需要的shape依次为（？,37,20,1）(?,37,20,1)，我们这里预处理，shape变为（4,20,1），去掉问号
    shape_Data = Data0.shape
    Data0_processed = Data0
    Time0_processed = Time0

    if shape_Data[1] < 40:
        Data0_processed = np.concatenate([np.zeros([shape_Data[0], 40 - shape_Data[1]]), Data0], axis=1)
        Time0_processed = np.concatenate([np.zeros([shape_Data[0], 40 - shape_Data[1]]), Time0], axis=1)
    elif shape_Data[1] > 40:
        Data0_processed = Data0[:, -40:]
        Time0_processed = Time0[:, -40:]
    # 到目前位置，维数分别为（4,20）,(4,20)对前两个添加维度
    Data0_processed = Data0_processed[:, :, np.newaxis]
    Time0_processed = Time0_processed[:, :, np.newaxis]
    return Data0_processed, Time0_processed   # 输出维度（4,20,1）（4,20,1）



def choose_action(Data0_processed, Time0_processed, s_Data, s_Time, network_Action, t, EPSILON, strategy=STRATEGY):
    if strategy == 'dqn':
        epsilon = EPSILON
        if random.random() <= epsilon:  # 随机数小于这个值，则随机选动作，否则，以最大的值选
            print("----------Random Action----------")
            index = random.sample(range(0, 37), 10)
        else:
            #  输出维度（4,20,1）（4,20,1）(4,) 外面加个中括号送入网络中，形状变为（1,4,20,1）（1,4,20,1）(1,4)
            out_2dim = network_Action.eval(
                feed_dict={s_Data: [Data0_processed], s_Time: [Time0_processed]})#
            out = out_2dim[0]  # 因为输出是（1,4）的二维数组，转换为一维数组
            index = heapq.nlargest(10, range(len(out)), out.take) # 选出值最大的两个的索引
    elif strategy == 'random':
        index = random.sample(range(0, 37), 10)

    elif strategy == 'max_K':
        sum_Data = np.squeeze(np.sum(Data0_processed, axis=1))    #首先计算数据矩阵的行和
        index = heapq.nlargest(10, range(len(sum_Data)), sum_Data.take)  # 选出值最大的两个的索引


    # 固定时隙
    elif strategy == 'fix_time':   # 时间比例是6 5 7 7,利用伪逆求解每个动作出现的概率
        #根据小区的流量到达率的比例分配波束，首先生成随机数，看这个随机数处于哪个区间，则调度哪个波束，如果已经加入，则继续，直到10个为止
        index = []
        arrivalRatio = scio.loadmat('dataRatio')['result'][0]  # 读取各个小区所占用时隙比例
        while len(index)<10:
            tmp = random.random()
            for ii in range(0, 37):
                cul_ratio = sum(arrivalRatio[0: ii+1])
                if (tmp < cul_ratio) and (ii not in index):
                    index.append(ii)
                    break
        print(index)

    # 轮训
    elif strategy == 'polling':
        lunxun_temp = [x for x in range(0, 73)]
        lunxun_time = (t-1)*10%37
        lunxun_index = lunxun_temp[lunxun_time:lunxun_time+10]
        index = [x%37 for x in lunxun_index]
        print('-------------------------------------------')
        print(index)


    # 最大最小流
    elif strategy == 'max_min':
        #根据小区的流量到达率的比例分配波束，首先生成随机数，看这个随机数处于哪个区间，则调度哪个波束，如果已经加入，则继续，直到10个为止
        index = []
        arrivalRatio = scio.loadmat('maxMinDataRatio')['result'][0]  # 读取各个小区所占用时隙比例
        while len(index)<10:
            tmp = random.random()
            for ii in range(0, 37):
                cul_ratio = sum(arrivalRatio[0: ii+1])
                if (tmp < cul_ratio) and (ii not in index):
                    index.append(ii)
                    break
        print(index)


    return index


def execute_action(Data0, Time0, now_CH, action):  # 进行流量下泻
    Data = copy.deepcopy(Data0)
    Time = copy.deepcopy(Time0)
    sum_Data = np.sum(Data, axis=1, keepdims=True)
    for idx in action:
        if sum_Data[idx][0] > now_CH[idx][0]:
            temp = now_CH[idx][0]
            count = 0
            for jdx in Data[idx][::-1]:
                count = count + 1
                flag = jdx - temp
                if flag < 0:  # 说明没减完
                    Data[idx][-count] = 0
                    temp = temp - jdx
                else:
                    Data[idx][-count] = flag
                    break
        else:
            Data[idx][:] = 0

    # 计算吞吐量
    throughput = sum(sum(Data0 - Data))

    # 数组中最后若全是0，则删除那一列
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


def calculate_reward(cumul_time2):
    reward = -0.000003 * cumul_time2
    return reward


def main():

    beam_location = Beam_position(beam_radius, round)

    sess = tf.InteractiveSession()

    # s_Data, s_Time, network_Action = createNetwork()

    # 定义代价函数
    # y = tf.placeholder("float", [None, 37])  # 目标y值
    # temp_action = tf.placeholder("float", [None, 37])
    # cost = tf.reduce_mean(tf.square((y - network_Action) * temp_action))  # 目标Q值减去选动作的Q值
    # train_signal_Q_net = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 设置保存器
    # saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    # checkpoint = tf.train.get_checkpoint_state("saved_networks")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    # epoch = 0
    # expool = deque()  # 初始化经验池
    #
    # epsilon = INITIATE_EPSILON
    #
    # all_ep_r = []  # 存放每次EPOCHS的reward_all
    #
    # while epoch < EPOCHS:  # 从头开始共计运行EPOCHS次
    #
    #     # 每一次epoch，epsilon都进行衰减
    #     if epsilon > FINISH_EPSILON:
    #         epsilon = epsilon - (INITIATE_EPSILON - FINISH_EPSILON) / EXPORE

    # 追加方式打开相应文件，用于记录指标
    file_throughput = open('metrics_' + STRATEGY + '/throughput.txt', 'a')
    file_delay = open('metrics_' + STRATEGY + '/delay.txt', 'a')
    file_delay_average = open('metrics_' + STRATEGY + '/delay_average.txt', 'a')
    file_delay1 = open('metrics_' + STRATEGY + '/delay1.txt', 'a')
    file_delay2 = open('metrics_' + STRATEGY + '/delay2.txt', 'a')
    file_delay3 = open('metrics_' + STRATEGY + '/delay3.txt', 'a')
    file_delay4 = open('metrics_' + STRATEGY + '/delay4.txt', 'a')
    file_loss = open('metrics_' + STRATEGY + '/loss.txt', 'a')
    file_epoch = open('metrics_' + STRATEGY + '/epoch.txt', 'a')
    file_reward = open('metrics_' + STRATEGY + '/reward.txt', 'a')

    # 一些统计指标，用于累加求平均
    throughput_all = 0
    delay_all = 0
    delay_average_all = 0
    delay_all2 = 0  # 执行动作后的周期总共累计等待时间
    delay_average_all2 = 0
    delay1_all = 0
    delay2_all = 0
    delay3_all = 0
    delay4_all = 0
    loss_all = 0
    reward_all = 0

    # epoch = epoch + 1
    # print('epoch:' + str(epoch))

    # 输入一个时刻的数据，dataNew4.mat（5.24）
    dataNew = 'dataNew4.mat'
    data_arrival = scio.loadmat(dataNew)  # 读取
    # 初始化，此为准备阶段，表示在运行之前，缓存队列已经有这么多的数据
    Data0 = (np.transpose(data_arrival['A']))[0]   # Data0.shape=(37,4)
    # print("Data0:", Data0.shape)
    Time0 = np.zeros([37,4])
    Time00 = np.arange(4)
    for i in range(37):
        Time0[i] = Time00   # Time0.shape=(37,4)
    # print("Time0:", Time0, Time0.shape)
    # t = 0

    # while t < TIMES:  # 每一个epoch运行TIMES次

        # t = t + 1
    # print('times------------:' + str(t))
    # 每循环一次的规则是先产生新数据,放入矩阵中,新数据的等待时间也加入->计算当前累计时间->选择动作->根据动作取出数据->计算现在累计时间->相应等待时间+1

    # 产生新的数据，放入矩阵中
    # Data0 = np.concatenate([data_arrival['A'][0], Data], axis=1)    # 这里产生数据，需要先判断是否增加/减小到达率  用于固定破松分布的流量请求
    # Time0 = np.concatenate([np.zeros([37, 1]), Time], axis=1)
    # print('*******************************************************')
    # print(sum(sum(data_arrival['A'][0])))
    # 先将第一行添加到最后一行(data['A']结构是（1000，37,1），所以删除行)
    # data_arrival['A'] = np.insert(data_arrival['A'], 500, data_arrival['A'][0], axis=0)
    # data_arrival['A'] = np.delete(data_arrival['A'], 0, axis=0)  # 删除第一行

    # 当前累积等待时间
    cumul_time1 = calculate_time(Data0, Time0)
    cumul_time1_average = cumul_time1/sum(sum(Data0))
    print("当前累积等待时间")
    print("cumul_time1:", cumul_time1)
    print("cumul_time1_average:", cumul_time1_average)
    # cumul_time11, cumul_time22, cumul_time33, cumul_time44 = calculate_every_cell_time(Data0, Time0)

    delay_all = delay_all + cumul_time1
    delay_average_all = delay_average_all + cumul_time1_average
    print("delay_all:", delay_all)
    print('delay_average_all ',delay_average_all )
    # delay1_all = delay1_all + cumul_time11
    # delay2_all = delay2_all + cumul_time22
    # delay3_all = delay3_all + cumul_time33
    # delay4_all = delay4_all + cumul_time44

    # 记录当前状态，三元组
    # print("Data0:", Data0)
    # print("Time0:", Time0)
    state0 = (Data0, Time0)
    #print(Data0)   #######################################################################################
    # 选择动作                        #################################这里应该放入神经网络，前向传输，让其预测出动作
    # 数据首先预处理
    # Data0_processed, Time0_processed = preproccess_alone(Data0, Time0)



    #index = choose_action(Data0_processed, Time0_processed, s_Data, s_Time, network_Action, t, epsilon)
    # index=GA.action(Data0,Time0) # 原
    # index = GA.action(Data0_processed, Time0_processed) #考虑到把Data和Time约束行数，第一种
    # # 每隔n步将以前的数据丢弃 # 考虑到Data和Time到一定行之后就丢弃之前的，第二种5.22
    # D = Data0.shape[1]  # shape=[37,None]
    # if D % 5 == 0:
    #     Data0 = np.delete(Data0, -1, axis=1)
    # T = Time0.shape[1]  # shape=[37,None]
    # if T % 5 == 0:
    #     Time0 = np.delete(Time0, -1, axis=1)
    # print("隔n步丢弃后的Data,Time:", Data0, Time0)
    index = GA_37xuan10.action(Data0, Time0)

    # print(type(index))

    #index.sort()
    # 记录动作
    action = index
    # print(state0)
    print('action：',action)
    # print('cumul_time1',cumul_time1)
    # 执行动作

    SNR = Channel_SINR_Cal(action, beam_location)
    Spectrum_Efficiency = Capacity_beam_dvbs2(SNR)
    Spectrum_Efficiency = np.array(Spectrum_Efficiency)
    now_CH = Spectrum_Efficiency * 5000  # Spectrum_Efficiency*B/(100k)，表示一个数据包100k，数据包大小归一化
    for i in range(len(now_CH)):  # 将信道容量取整
        now_CH[i] = int(now_CH[i])
    now_CH_disord = []
    for i in range(37):
        now_CH_disord.append(0)
    for i in range(len(now_CH)):
        now_CH_disord[action[i]] = now_CH[i]
    now_CH_disord = np.array(now_CH_disord)
    now_CH_disord = now_CH_disord.reshape([-1,1])
    # print("now_CH_disord:", now_CH_disord)#############################################################################
    # print('******************************************************')
    # print(sum(sum(now_CH_disord)))

    Data, Time, once_throughput = execute_action(Data0, Time0, now_CH_disord, action)
    print("执行动作后：")
    print("Data:", Data)
    print("Time:", Time)

    throughput_all = throughput_all + once_throughput
    print('once_throughput',once_throughput)
    print('throughput_all ',throughput_all)
    # 执行动作后的累计等待时间
    cumul_time2 = calculate_time(Data, Time)
    print("cumul_time2:", cumul_time2)
    cumul_time1_average2 = cumul_time2 / sum(sum(Data))  # 执行动作后的平均累计等待时间
    print("cumul_time1_average2:", cumul_time1_average2)
    delay_all2 = delay_all2 + cumul_time2
    delay_average_all2 = delay_average_all2 + cumul_time1_average2
    print("delay_all2:", delay_all2)
    print("delay_average_all2:", delay_average_all2)
    # 计算即时收益
    # reward = calculate_reward(cumul_time2) # 原
    reward = once_throughput
    reward_all = reward_all + reward
    print("reward:", reward)
    print("reward_all:", reward_all)
    # 记录执行动作后到达的新状态
    state1 = (Data, Time)
    #print(Data)##############################################################################

    # if Data.shape[1] > 40:
    #     terminal = True
    #     expool.append((state0, action, reward, state1, terminal))
    #     break
    # else:
    #     terminal = False
    #     expool.append((state0, action, reward, state1, terminal))
    # # print(expool)
    #
    # if len(expool) > POOL_MEMORY:  # 如果经验池容量超过了，那么从左边丢弃最早的经验
    #     expool.popleft()
    #     # print(expool)
    # if (epoch > OBSERVE) & (STRATEGY == 'dqn'):  # 在等待这么多步之后，开始抽样
    #     minibatch = random.sample(expool, BATCH)
    #     # print(t)
    #     s_j_batch = [d[0] for d in minibatch]
    #     a_batch = [d[1] for d in minibatch]
    #     r_batch = [d[2] for d in minibatch]
    #     s_j1_batch = [d[3] for d in minibatch]
    #     terminal_batch = [d[4] for d in minibatch]
    #     # print(s_j_batch)
    #     # print(a_batch)
    #     # print(r_batch)
    #     # print(s_j1_batch)
    #
    #     s_j_batch_processed_data = []
    #     s_j_batch_processed_time = []
    #     y_batch = []
    #     tmp_action = []
    #     for i in range(BATCH):
    #         temp = np.zeros([37])
    #
    #         # 预处理神经网络的输入
    #         s_j_batch_Data0_processed, s_j_batch_Time0_processed = preproccess_alone(s_j_batch[i][0], s_j_batch[i][1])
    #         s_j_batch_processed_data.append(s_j_batch_Data0_processed)
    #         s_j_batch_processed_time.append(s_j_batch_Time0_processed)
    #
    #         # 处理神经网络的目标值
    #         s_j1_batch_Data0_processed, s_j1_batch_Time0_processed = preproccess_alone(s_j1_batch[i][0], s_j1_batch[i][1])
    #         if terminal_batch[i]:
    #             temp_temp = []
    #             for i_i in range(37):
    #                 temp_temp.append(r_batch[i])
    #             y_batch.append(temp_temp)
    #         else:
    #             s_j1_out_2dim = network_Action.eval(
    #                 feed_dict={s_Data: [s_j1_batch_Data0_processed], s_Time: [s_j1_batch_Time0_processed]})
    #             s_j1_out_max = max(s_j1_out_2dim[0])  # 因为输出是（1,4）的二维数组，转换为一维数组
    #             temp_temp = []
    #             for i_i in range(37):
    #                 temp_temp.append(s_j1_out_max)
    #             s_j1_out = np.array(temp_temp)
    #             y_batch.append(r_batch[i] + GAMMA * s_j1_out)  # 根据公式，这个时目标y值
    #
    #         # print(r_batch[i] / 50)
    #         # print(a_batch[i])
    #         # print(a_batch[i][0])
    #         # print(a_batch[i][1])
    #         for i_i in range(10):
    #             temp[a_batch[i][i_i]] = 1  # 将选择的两个动作映射回一个四维向量中，如动作2,3，映射为0 0 1 1
    #         tmp_action.append(temp)
    #         # print(temp)
    #
    #     _, train_cost = sess.run([train_signal_Q_net, cost], feed_dict={
    #         y: y_batch,  # 目标值
    #         temp_action: tmp_action,
    #         s_Data: s_j_batch_processed_data,
    #         s_Time: s_j_batch_processed_time})
    #     print('##### cost:' + str(train_cost) + ' ###')
    #     loss_all = loss_all + train_cost

    # Time = Time + 1
    # print('Time ',Time )
    # 画出每个epoch的reward
    # all_ep_r.append(reward_all)
    # if (epoch == EPOCHS):
    #     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    #     plt.xlabel('epochs')
    #     plt.ylabel('Moving averaged epochs reward')
    #     plt.show()
    # if (t == TIMES): # 每周期在最后一步打印统计量
    #     print("每周期的最后一步：")
    #     print("throughput_all:", throughput_all)
    #     # print("cumul_time2:", cumul_time2)
    #     # print('once_throughput', once_throughput)
    #     print("delay_all2:", delay_all2)
    #     print("delay_average_all2:", delay_average_all2)
    #     print("reward_all:", reward_all)






    # if (epoch % SAVE_STEP == 0) & (MODEL == 'train'):  # 每到一次epoch，保存一次参数
    #     saver.save(sess, 'saved_networks/' + 'DQN', global_step=epoch)

    # print(loss_all)
    # print(t)

    # 记录相关数据
    # # file_epoch.write(str(epoch) + '\n')
    # file_loss.write(str(loss_all / t) + '\n')
    # file_delay.write(str(delay_all / t) + '\n')
    # file_delay_average.write(str(delay_average_all / t) + '\n')
    # file_delay1.write(str(delay1_all / t) + '\n')
    # file_delay2.write(str(delay2_all / t) + '\n')
    # file_delay3.write(str(delay3_all / t) + '\n')
    # file_delay4.write(str(delay4_all / t) + '\n')
    # file_throughput.write(str(throughput_all / t) + '\n')
    # file_reward.write(str(reward_all / t) + '\n')

    file_epoch.close()
    file_loss.close()
    file_delay.close()
    file_delay_average.close()
    file_delay1.close()
    file_delay2.close()
    file_delay3.close()
    file_delay4.close()
    file_throughput.close()
    file_reward.close()


if __name__ == "__main__":
    main()
