# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
from collections import deque
import scipy.io as scio

N = 37  # 波束总数
M = 12  # 信道总数
FR = 1  # 固定信道四色复用
lamda = 70  # 1小时泊松到达率70
miu = 0.05  # 持续时间为0.05小时=3分钟
beam_radius = 1  # 波束半径1m
round = 3  # 波束的圈数
Ant_Type = 'ITU-R S672-4 Circular'  ##GEO卫星，圆形单馈源天线'ITU-R S672-4 Circular', 'ATaD_p389_n1'
Pow_Noise_Base = 0  # 接收机噪声功率固定为0dBW
CINR_Threshold = 10  # 相当于最大可允许的干扰功率为3.4668W (2.2217 dBW)
C2N_Max = 15
ACTIONS = M
GAMMA = 0.99  # discount factor
EPSILON_Initial = 0.001  # initial value of epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
EPSILON = EPSILON_Initial
BATCH = 8

EXPLORE = 3000
OBSERVE = EXPLORE / 20  # replay start size
REPLAY_MEMORY = int(EXPLORE / 6)  # replay memory capacity
Num_Channel = 37
Num_Choose_Channel = 10
Beam_Width = 2

P_tot = 33  # dBW 总卫星发射功率
K = 1.38 * (10 ** (-23))  # 玻尔兹曼常数
T = 207  # K
B = 500 * (10 ** (6))  # 500M


# 行
shuaijian = 0.03
# 24小时的数据
# def gen_data(lam1=int(860*shuaijian), lam2=int(1449*shuaijian), lam3=int(895*shuaijian), lam4=int(1529*shuaijian), lam5=int(712*shuaijian), lam6=int(967*shuaijian), lam7=int(1068*shuaijian), lam8=int(1369*shuaijian),
#              lam9=int(1565*shuaijian), lam10=int(1722*shuaijian), lam11=int(1042*shuaijian), lam12=int(1566*shuaijian), lam13=int(1484*shuaijian), lam14=int(624*shuaijian), lam15=int(922*shuaijian), lam16=int(2043*shuaijian),
#              lam17=int(shuaijian*1695), lam18=int(shuaijian*1692), lam19=int(shuaijian*1323), lam20=int(shuaijian*1402), lam21=int(shuaijian*1864), lam22=int(shuaijian*1829), lam23=int(shuaijian*1646),
#              lam24=int(shuaijian*1866), lam25=int(shuaijian*1414), lam26=int(shuaijian*1694), lam27=int(shuaijian*1758), lam28=int(shuaijian*1539), lam29=int(shuaijian*1909), lam30=int(shuaijian*1537), lam31=int(shuaijian*1770), lam32=int(shuaijian*1981),
#              lam33=int(shuaijian*1734), lam34=int(shuaijian*998), lam35=int(shuaijian*1671), lam36=int(shuaijian*1830), lam37=int(shuaijian*1645)):
# 原数据
def gen_data(lam1=int(860*shuaijian), lam2=int(1450*shuaijian), lam3=int(895*shuaijian), lam4=int(1529*shuaijian), lam5=int(712*shuaijian), lam6=int(968*shuaijian), lam7=int(1067*shuaijian), lam8=int(1370*shuaijian),
             lam9=int(1566*shuaijian), lam10=int(1722*shuaijian), lam11=int(1042*shuaijian), lam12=int(1566*shuaijian), lam13=int(1482*shuaijian), lam14=int(627*shuaijian), lam15=int(922*shuaijian), lam16=int(2043*shuaijian),
             lam17=int(shuaijian*1695), lam18=int(shuaijian*1692), lam19=int(shuaijian*1323), lam20=int(shuaijian*1404), lam21=int(shuaijian*1864), lam22=int(shuaijian*1829), lam23=int(shuaijian*1646),
             lam24=int(shuaijian*1866), lam25=int(shuaijian*1413), lam26=int(shuaijian*1694), lam27=int(shuaijian*1758), lam28=int(shuaijian*1539), lam29=int(shuaijian*1910), lam30=int(shuaijian*1538), lam31=int(shuaijian*1770), lam32=int(shuaijian*1981),
             lam33=int(shuaijian*1732), lam34=int(shuaijian*998), lam35=int(shuaijian*1671), lam36=int(shuaijian*1830), lam37=int(shuaijian*1645)):
    arrival1 = np.random.poisson(lam=lam1, size=1)   #数据包大小100k
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
if __name__ == "__main__":
            ###################得到dataNew.mat的过程##########################
    t = 0
    lamda_pool3 = deque()  # 初始化
    while t < 500:
        t = t + 1
        c = gen_data()
        lamda_pool3.append(c)
    dataNew_yuan500_03= 'dataNew_yuan500_03.mat'
    scio.savemat(dataNew_yuan500_03,{'A': lamda_pool3})  # 以字典的形式保存
# 原来dqn里面的lam数据，500行
# def gen_data_yuan500(lam1=860, lam2=1450, lam3=895, lam4=1529, lam5=712, lam6=968, lam7=1067, lam8=1370,
#              lam9=1566, lam10=1722, lam11=1042, lam12=1566, lam13=1482, lam14=627, lam15=922, lam16=2043,
#              lam17=1695, lam18=1692, lam19=1323, lam20=1404, lam21=1864, lam22=1829, lam23=1646,
#              lam24=1866, lam25=1413, lam26=1694, lam27=1758, lam28=1539, lam29=1910, lam30=1538, lam31=1770, lam32=1981,
#              lam33=1732, lam34=998, lam35=1671, lam36=1830, lam37=1645):
#     arrival1 = np.random.poisson(lam=lam1, size=1)   #数据包大小100k
#     arrival2 = np.random.poisson(lam=lam2, size=1)
#     arrival3 = np.random.poisson(lam=lam3, size=1)
#     arrival4 = np.random.poisson(lam=lam4, size=1)
#     arrival5 = np.random.poisson(lam=lam5, size=1)
#     arrival6 = np.random.poisson(lam=lam6, size=1)
#     arrival7 = np.random.poisson(lam=lam7, size=1)
#     arrival8 = np.random.poisson(lam=lam8, size=1)
#     arrival9 = np.random.poisson(lam=lam9, size=1)
#     arrival10 = np.random.poisson(lam=lam10, size=1)
#     arrival11 = np.random.poisson(lam=lam11, size=1)
#     arrival12 = np.random.poisson(lam=lam12, size=1)
#     arrival13 = np.random.poisson(lam=lam13, size=1)
#     arrival14 = np.random.poisson(lam=lam14, size=1)
#     arrival15 = np.random.poisson(lam=lam15, size=1)
#     arrival16 = np.random.poisson(lam=lam16, size=1)
#     arrival17 = np.random.poisson(lam=lam17, size=1)
#     arrival18 = np.random.poisson(lam=lam18, size=1)
#     arrival19 = np.random.poisson(lam=lam19, size=1)
#     arrival20 = np.random.poisson(lam=lam20, size=1)
#     arrival21 = np.random.poisson(lam=lam21, size=1)
#     arrival22 = np.random.poisson(lam=lam22, size=1)
#     arrival23 = np.random.poisson(lam=lam23, size=1)
#     arrival24 = np.random.poisson(lam=lam24, size=1)
#     arrival25 = np.random.poisson(lam=lam25, size=1)
#     arrival26 = np.random.poisson(lam=lam26, size=1)
#     arrival27 = np.random.poisson(lam=lam27, size=1)
#     arrival28 = np.random.poisson(lam=lam28, size=1)
#     arrival29 = np.random.poisson(lam=lam29, size=1)
#     arrival30 = np.random.poisson(lam=lam30, size=1)
#     arrival31 = np.random.poisson(lam=lam31, size=1)
#     arrival32 = np.random.poisson(lam=lam32, size=1)
#     arrival33 = np.random.poisson(lam=lam33, size=1)
#     arrival34 = np.random.poisson(lam=lam34, size=1)
#     arrival35 = np.random.poisson(lam=lam35, size=1)
#     arrival36 = np.random.poisson(lam=lam36, size=1)
#     arrival37 = np.random.poisson(lam=lam37, size=1)
#     return np.concatenate([[arrival1], [arrival2], [arrival3], [arrival4],
#                            [arrival5], [arrival6], [arrival7], [arrival8],
#                            [arrival9], [arrival10], [arrival11], [arrival12],
#                            [arrival13], [arrival14], [arrival15], [arrival16],
#                            [arrival17], [arrival18], [arrival19], [arrival20],
#                            [arrival21], [arrival22], [arrival23], [arrival24],
#                            [arrival25], [arrival26], [arrival27], [arrival28],
#                            [arrival29], [arrival30], [arrival31], [arrival32],
#                            [arrival33], [arrival34], [arrival35], [arrival36],
#                            [arrival37]], axis=0)
#
# if __name__ == "__main__":
#             ###################得到dataNew.mat的过程##########################
#     t = 0
#     lamda_pool1 = deque()  # 初始化
#     while t < 500:
#         t = t + 1
#         a = gen_data_yuan500()
#         lamda_pool1.append(a)
#     dataNew_yuan500= 'dataNew_yuan500.mat'
#     scio.savemat(dataNew_yuan500, {'A': lamda_pool1})  # 以字典的形式保存

# 原来dqn里面的lam数据，改成40行
# def gen_data_yuan40(lam1=860, lam2=1450, lam3=895, lam4=1529, lam5=712, lam6=968, lam7=1067, lam8=1370,
#              lam9=1566, lam10=1722, lam11=1042, lam12=1566, lam13=1482, lam14=627, lam15=922, lam16=2043,
#              lam17=1695, lam18=1692, lam19=1323, lam20=1404, lam21=1864, lam22=1829, lam23=1646,
#              lam24=1866, lam25=1413, lam26=1694, lam27=1758, lam28=1539, lam29=1910, lam30=1538, lam31=1770, lam32=1981,
#              lam33=1732, lam34=998, lam35=1671, lam36=1830, lam37=1645):
#     arrival1 = np.random.poisson(lam=lam1, size=1)   #数据包大小100k
#     arrival2 = np.random.poisson(lam=lam2, size=1)
#     arrival3 = np.random.poisson(lam=lam3, size=1)
#     arrival4 = np.random.poisson(lam=lam4, size=1)
#     arrival5 = np.random.poisson(lam=lam5, size=1)
#     arrival6 = np.random.poisson(lam=lam6, size=1)
#     arrival7 = np.random.poisson(lam=lam7, size=1)
#     arrival8 = np.random.poisson(lam=lam8, size=1)
#     arrival9 = np.random.poisson(lam=lam9, size=1)
#     arrival10 = np.random.poisson(lam=lam10, size=1)
#     arrival11 = np.random.poisson(lam=lam11, size=1)
#     arrival12 = np.random.poisson(lam=lam12, size=1)
#     arrival13 = np.random.poisson(lam=lam13, size=1)
#     arrival14 = np.random.poisson(lam=lam14, size=1)
#     arrival15 = np.random.poisson(lam=lam15, size=1)
#     arrival16 = np.random.poisson(lam=lam16, size=1)
#     arrival17 = np.random.poisson(lam=lam17, size=1)
#     arrival18 = np.random.poisson(lam=lam18, size=1)
#     arrival19 = np.random.poisson(lam=lam19, size=1)
#     arrival20 = np.random.poisson(lam=lam20, size=1)
#     arrival21 = np.random.poisson(lam=lam21, size=1)
#     arrival22 = np.random.poisson(lam=lam22, size=1)
#     arrival23 = np.random.poisson(lam=lam23, size=1)
#     arrival24 = np.random.poisson(lam=lam24, size=1)
#     arrival25 = np.random.poisson(lam=lam25, size=1)
#     arrival26 = np.random.poisson(lam=lam26, size=1)
#     arrival27 = np.random.poisson(lam=lam27, size=1)
#     arrival28 = np.random.poisson(lam=lam28, size=1)
#     arrival29 = np.random.poisson(lam=lam29, size=1)
#     arrival30 = np.random.poisson(lam=lam30, size=1)
#     arrival31 = np.random.poisson(lam=lam31, size=1)
#     arrival32 = np.random.poisson(lam=lam32, size=1)
#     arrival33 = np.random.poisson(lam=lam33, size=1)
#     arrival34 = np.random.poisson(lam=lam34, size=1)
#     arrival35 = np.random.poisson(lam=lam35, size=1)
#     arrival36 = np.random.poisson(lam=lam36, size=1)
#     arrival37 = np.random.poisson(lam=lam37, size=1)
#     return np.concatenate([[arrival1], [arrival2], [arrival3], [arrival4],
#                            [arrival5], [arrival6], [arrival7], [arrival8],
#                            [arrival9], [arrival10], [arrival11], [arrival12],
#                            [arrival13], [arrival14], [arrival15], [arrival16],
#                            [arrival17], [arrival18], [arrival19], [arrival20],
#                            [arrival21], [arrival22], [arrival23], [arrival24],
#                            [arrival25], [arrival26], [arrival27], [arrival28],
#                            [arrival29], [arrival30], [arrival31], [arrival32],
#                            [arrival33], [arrival34], [arrival35], [arrival36],
#                            [arrival37]], axis=0)
#
# if __name__ == "__main__":
#             ###################得到dataNew.mat的过程##########################
#     t = 0
#     lamda_pool2 = deque()  # 初始化
#     while t < 40:
#         t = t + 1
#         b = gen_data_yuan40()
#         lamda_pool2.append(b)
#     dataNew_yuan40= 'dataNew_yuan40.mat'
#     scio.savemat(dataNew_yuan40, {'A': lamda_pool2})  # 以字典的形式保存

# 卫星场景初版统计的业务量，40行
# def gen_data_boshuweizhi(lam1=874,lam2=815,lam3=1760,lam4=1730,lam5=1290,lam6=41,lam7=816,lam8=1530,lam9=1980,lam10=1651,lam11=610,lam12=11,lam13=864,lam14=1550,lam15=1894,
# lam16=2457,lam17=1673,lam18=854,lam19=186,lam20=140,lam21=330,lam22=530,lam23=400,lam24=340,lam25=160,lam26=370,lam27=210,lam28=469,lam29=1972,lam30=964,
# lam31=554,lam32=1627,lam33=732,lam34=93,lam35=267,lam36=84,lam37=31):
#     arrival1 = np.random.poisson(lam=lam1, size=1)   #数据包大小1M
#     arrival2 = np.random.poisson(lam=lam2, size=1)
#     arrival3 = np.random.poisson(lam=lam3, size=1)
#     arrival4 = np.random.poisson(lam=lam4, size=1)
#     arrival5 = np.random.poisson(lam=lam5, size=1)
#     arrival6 = np.random.poisson(lam=lam6, size=1)
#     arrival7 = np.random.poisson(lam=lam7, size=1)
#     arrival8 = np.random.poisson(lam=lam8, size=1)
#     arrival9 = np.random.poisson(lam=lam9, size=1)
#     arrival10 = np.random.poisson(lam=lam10, size=1)
#     arrival11 = np.random.poisson(lam=lam11, size=1)
#     arrival12 = np.random.poisson(lam=lam12, size=1)
#     arrival13 = np.random.poisson(lam=lam13, size=1)
#     arrival14 = np.random.poisson(lam=lam14, size=1)
#     arrival15 = np.random.poisson(lam=lam15, size=1)
#     arrival16 = np.random.poisson(lam=lam16, size=1)
#     arrival17 = np.random.poisson(lam=lam17, size=1)
#     arrival18 = np.random.poisson(lam=lam18, size=1)
#     arrival19 = np.random.poisson(lam=lam19, size=1)
#     arrival20 = np.random.poisson(lam=lam20, size=1)
#     arrival21 = np.random.poisson(lam=lam21, size=1)
#     arrival22 = np.random.poisson(lam=lam22, size=1)
#     arrival23 = np.random.poisson(lam=lam23, size=1)
#     arrival24 = np.random.poisson(lam=lam24, size=1)
#     arrival25 = np.random.poisson(lam=lam25, size=1)
#     arrival26 = np.random.poisson(lam=lam26, size=1)
#     arrival27 = np.random.poisson(lam=lam27, size=1)
#     arrival28 = np.random.poisson(lam=lam28, size=1)
#     arrival29 = np.random.poisson(lam=lam29, size=1)
#     arrival30 = np.random.poisson(lam=lam30, size=1)
#     arrival31 = np.random.poisson(lam=lam31, size=1)
#     arrival32 = np.random.poisson(lam=lam32, size=1)
#     arrival33 = np.random.poisson(lam=lam33, size=1)
#     arrival34 = np.random.poisson(lam=lam34, size=1)
#     arrival35 = np.random.poisson(lam=lam35, size=1)
#     arrival36 = np.random.poisson(lam=lam36, size=1)
#     arrival37 = np.random.poisson(lam=lam37, size=1)
#     return np.concatenate([[arrival1], [arrival2], [arrival3], [arrival4],
#                            [arrival5], [arrival6], [arrival7], [arrival8],
#                            [arrival9], [arrival10], [arrival11], [arrival12],
#                            [arrival13], [arrival14], [arrival15], [arrival16],
#                            [arrival17], [arrival18], [arrival19], [arrival20],
#                            [arrival21], [arrival22], [arrival23], [arrival24],
#                            [arrival25], [arrival26], [arrival27], [arrival28],
#                            [arrival29], [arrival30], [arrival31], [arrival32],
#                            [arrival33], [arrival34], [arrival35], [arrival36],
#                            [arrival37]], axis=0)
#
# if __name__ == "__main__":
#             ###################得到dataNew.mat的过程##########################
#     t = 0
#     lamda_pool5 = deque()
#     while t < 40:
#         t = t + 1
#         e = gen_data_boshuweizhi()
#         lamda_pool5.append(e)
#     dataNew_boshuweizhi40= 'dataNew_boshuweizhi40.mat'
#     scio.savemat(dataNew_boshuweizhi40, {'A': lamda_pool5})  # 以字典的形式保存

