#  修改的之前问题中的错误：在二进制转十进制，位数乘上2的5-i次方

import time

import matplotlib.pyplot as plt
import math
import random
import numpy as np
import scipy.io as scio
import copy  # 深拷贝
from collections import defaultdict

from baselines.ppo1 import env_beamhopping
from baselines.ppo1.env_beamhopping import BeamhoppingEnv
env = BeamhoppingEnv()

from baselines.ppo1.env_beamhopping import BeamhoppingEnv

def action(rl_action, Data0, Time0):
    pop_size = 50  # 种群数量  348330136  500--》5000,原10000
    upper_limit = 36
    chromosome_length = 37  # 染色体长度 36*6=216，好像没用到
    iter = 50  #迭代次数，原500
    pc = 0.5 # 杂交概率
    pm = 0.01  # 变异概率
    results = []  # 存储每一代的最优解，N个二元组
    pop = init_population1(rl_action, pop_size) # （10000,60）
    best_X = []
    best_Y = []
    Best_fit = []

    for i in range(iter):
        #obj_value = calc_obj_value(pop, chromosome_length, upper_limit)  # 个体评价，有负值      # 获取最大吞吐量
        obj_value,Data1,Time =calc_obj_value(pop, chromosome_length, upper_limit, Data0, Time0,pop_size, i) #POP=ACTION,计算个体的适应度（吞吐量）
        # print('obj_value:',obj_value)
        fit_value = calc_fit_value(obj_value)  # 个体适应度，不好的归0，可以理解为去掉上面的负值
        # print("fit_value:", fit_value)

        best_individual, best_fit, ind = find_best(pop, fit_value)  # 第一个是最优基因序列, 第二个是对应的最佳个体适度
        # print('best_individual:',best_individual)
        # print("best_fit:", best_fit)
        best_individual1 = copy.deepcopy(best_individual)
        # calculate fairness


        C = []
        for j in range(1, 11):
            B = best_individual1[6 * (j - 1):6 * j]
            DECB = binary2decimal2(B, upper_limit, i)
            C.append(DECB)  # 二进制转换为十进制，变成10个数
        best_individual1 = C  # action为选中的十个小区
        # print("best_individual1:", best_individual1)
        # print("best_fit:", best_fit)
        Best_fit.append(best_fit)
        # print("i", i)
        results.append(best_individual) # 每一代的最优个体
        # selection(pop, fit_value, pop_size, upper_limit, i)  # 选择
        # while best_individual not in pop:
        #     selection(pop, fit_value, pop_size, upper_limit, i)
        # crossover(pop, pc, pop_size, upper_limit, ind, best_individual, i)  # 染色体交叉（最优个体之间进行0、1互换）
        # while best_individual not in pop:
        #     crossover(pop, pc, pop_size, upper_limit, ind, best_individual, i)
        # mutation(pop, pm, pop_size, upper_limit, ind, best_individual, i)  # 染色体变异（其实就是随机进行0、1取反）

        # selection(pop, fit_value, pop_size, upper_limit, i)  # 选择
        # crossover(pop, pc, pop_size, upper_limit, ind, best_individual, i)  # 染色体交叉（最优个体之间进行0、1互换）
        # mutation(pop, pm, pop_size, upper_limit, ind, best_individual, i)  # 染色体变异（其实就是随机进行0、1取反）

        selection(pop, fit_value)  # 选择
        crossover(pop, pc)  # 染色体交叉（最优个体之间进行0、1互换）
        mutation(pop, pm)  # 染色体变异（其实就是随机进行0、1取反）

    action1=results[-1] # 最后一代的最优个体，这时还是[60个0/1的数]
    c = []
    for h in range(1, 11):
        b = action1[6 * (h - 1):6 * h]
        decb = binary2decimal2(b, upper_limit, i)
        c.append(decb) # 二进制转换为十进制，变成10个数
    action = c # action为选中的十个小区
    return action, best_fit

# # 计算2进制序列代表的数值
# def binary2decimal2(binary, upper_limit, i):
#     if i==0:
#         upper_limit=2 ** 6- 1
#     else:
#         upper_limit =36
#     t = 0
#     for j in range(len(binary)):
#         t += binary[j] * 2 ** (5-j) # 十进制
#     t = t * upper_limit // (2 ** 6- 1) # 表现度x
#     return t
# 计算2进制序列代表的数值
def binary2decimal2(binary, upper_limit, i):
    upper_limit =36
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 2 ** (5-j) # 十进制
    if t>upper_limit:
        t = t * upper_limit // (2 ** 6- 1) # 表现度x
    return t

# 计算10进制序列代表的2进制数值
def decimal2binary(List): #输入是一个list，输入时10个0-36的十进制数，输出10个二进制数，位数6位
    #result=0
    k=1
    temp = copy.deepcopy(List)
    result={}
    result2=[]
    for j in range(len(List)):
        a=[]
        while(temp[j]):
            i = temp[j]%2
            #print(i)
            #result = k * i + result
            k = k*10
            temp[j] = temp[j]//2
            a.append(i) # 短除法得到二进制
        result[j]=a
        result[j].reverse()

        while len(result[j])<6: # 二进制不够6位补0
            result[j].reverse()
            result[j].append(0)
            result[j].reverse()
        for i in range(6):
            result2.append(result[j][i])

    return result2

# def init_population1(pop_size, chromosome_length): #产生10个【0，36】的整数，编码成2进制 2018/12/18
#     resultList = []  # 用于存放结果的List
#     A = 0  # 最小随机数
#     B = 36  # 最大随机数
#     COUNT = 10
#     # 形如[[0,1,..0,1],[0,1,..0,1]...]
#     pop=[]
#     for i in range(pop_size):
#         resultList = random.sample(range(A, B + 1), COUNT) #从[0,...36]获取10个数:[24, 11, 0, 29, 20, 19, 14, 31, 9, 8]
#         binresult=decimal2binary(resultList) # 转换为二进制数:[6*10个0/1]
#         pop.append(binresult) # [[6*10个0/1],...,[6*10个0/1]],pop_size = 10000个，shape=（10000,60）
#
#     return pop

def init_population1(action, pop_size): #产生10个【0，36】的整数，编码成2进制 2018/12/18
    resultList = []  # 用于存放结果的List
    A = 0  # 最小随机数
    B = 36  # 最大随机数
    COUNT = 10
    # 形如[[0,1,..0,1],[0,1,..0,1]...]
    pop = []
    for i in range(pop_size):
        ac = action
        # resultList = random.sample(range(A, B + 1), COUNT)  # 从[0,...36]获取10个数:[24, 11, 0, 29, 20, 19, 14, 31, 9, 8]
        binresult = decimal2binary(ac)  # 转换为二进制数:[6*10个0/1]
        pop.append(binresult)  # [[6*10个0/1],...,[6*10个0/1]],pop_size = 10000个，shape=（10000,60）
    return pop
    # resultList = []  # 用于存放结果的List
    # A = 0  # 最小随机数
    # B = 36  # 最大随机数
    # COUNT = 10
    # # 形如[[0,1,..0,1],[0,1,..0,1]...]
    # pop=[]
    # for i in range(pop_size-1):
    #     resultList = random.sample(range(A, B + 1), COUNT) #从[0,...36]获取10个数:[24, 11, 0, 29, 20, 19, 14, 31, 9, 8]
    #     binresult=decimal2binary(resultList) # 转换为二进制数:[6*10个0/1]
    #     pop.append(binresult) # [[6*10个0/1],...,[6*10个0/1]],pop_size = 10000个，shape=（10000,60）
    # binresult = decimal2binary(action)
    # pop.append(binresult)
    # return pop


def calc_obj_value(pop, chromosome_length, upper_limit,Data0,Time0,pop_size, I): #pop=[[1,0,1,...],[1,0,0,...],...[0,0,1,...]]，计算适应度
    action_500=[] #存储索引，也就是十进制小区数
    action={}
    for i in range(pop_size):  #10000，应改为每6位计算一次值，即为索引  #2018/12/18
        a = pop[i] #len=60
        c=[]
        for h in range(1,11): #10
            #for i in range(6*(h-1),6*h):
            b=a[6*(h-1):6*h]
            decb=binary2decimal2(b, upper_limit, I)#二进制转10进制(可能会超过0-36范围），并转换为表现度x（约束在0-36范围）
            c.append(decb) # 10个十进制数
        action[i]=c # 10个十进制数是一个个体，action[i]=[10个数]
        action_500.append(action[i]) # 共10000个个体，(10000,10)#500个种群，所有1的索引
    # print("action:", action)
    # print("action_500:", action_500)
    beam_location =env.init()
    obj_value = []
    for x in action_500: #此处x为action
        # 把缩放过后的那个数，带入我们要求的公式中
        # 种群中个体有几个，就有几个这种“缩放过后的数”
        #############################这里替换成 最大化吞吐量################################
        # 执行动作
        SNR = env_beamhopping.Channel_SINR_Cal(x, beam_location)
        Spectrum_Efficiency = env_beamhopping.Capacity_beam_dvbs2(SNR)
        Spectrum_Efficiency = np.array(Spectrum_Efficiency)
        now_CH = Spectrum_Efficiency * 5000  # Spectrum_Efficiency*B/(100k)，表示一个数据包100k，数据包大小归一化
        for i in range(len(now_CH)):  # 将信道容量取整
            now_CH[i] = int(now_CH[i])
        now_CH_disord = []
        for i in range(37):
            now_CH_disord.append(0)
        for i in range(len(now_CH)):
            #now_CH_disord[action[i]] = now_CH[i]
            now_CH_disord[x[i]] = now_CH[i]
        now_CH_disord = np.array(now_CH_disord)
        now_CH_disord = now_CH_disord.reshape([-1, 1])

        Data1, Time, once_throughput_all= env_beamhopping.execute_action(Data0,Time0,now_CH_disord, x)

        obj_value.append(once_throughput_all) # 把吞吐量作为适应度，得到10000个个体的适应度
    # 这里先返回带入公式计算后的数值列表，作为种群个体优劣的评价
    return obj_value,Data1, Time


# 淘汰，将适应度值小于10的适应度设为0
def calc_fit_value(obj_value):
    fit_value = []
    # 去掉小于0的值，更改c_min会改变淘汰的下限
    # 比如设成10可以加快收敛
    # 但是如果设置过大，有可能影响了全局最优的搜索
    c_min = 10
    for value in obj_value:
        if value > c_min:
            temp = value
        else:
            temp = 0.
        fit_value.append(temp)
    # fit_value保存的是活下来的值
    return fit_value


# 找出最优解和最优解的基因编码
def find_best(pop, fit_value):
    # 用来存最优基因编码
    # best_individual = []
    best_individual = pop[0]
    # 先假设第一个基因的适应度最好
    best_fit = fit_value[0]
    ind = 0
    for i in range(1, len(pop)): # 循环10000次，循环每个个体
        if (fit_value[i] > best_fit): # 找到最大适应度，和对应的个体
            best_fit = fit_value[i]
            best_individual = pop[i]
            ind = i
    # best_fit是值
    # best_individual是基因序列
    # print("pop", pop)
    return best_individual, best_fit, ind


# 计算累计概率
def cum_sum(fit_value):
    # 输入[1, 2, 3, 4, 5]，返回[1,3,6,10,15]，matlab的一个函数
    # 这个地方遇坑，局部变量如果赋值给引用变量，在函数周期结束后，引用变量也将失去这个值
    temp = fit_value[:]
    for i in range(len(temp)):
        fit_value[i] = (sum(temp[:i + 1]))


# 轮赌法选择
# def selection(pop, fit_value, pop_size, upper_limit, I):
#     # https://blog.csdn.net/pymqq/article/details/51375522
#
#     p_fit_value = []
#     # 适应度总和
#     total_fit = sum(fit_value)
#     # 归一化，使概率总和为1
#     for i in range(len(fit_value)):
#         p_fit_value.append(fit_value[i] / total_fit) # 概率
#     # 概率求和排序
#
#     # https://www.cnblogs.com/LoganChen/p/7509702.html
#     cum_sum(p_fit_value) # 累计概率求和
#     pop_len = len(pop)
#     # 类似搞一个转盘吧下面这个的意思
#     ms = sorted([random.random() for i in range(pop_len)]) #产生pop_len个0-1之间的随机数
#     fitin = 0
#     newin = 0
#     newpop = pop[:]
#     # 转轮盘选择法
#     while newin < pop_len:
#         # 如果这个概率大于随机出来的那个概率，就选这个
#         if (ms[newin] < p_fit_value[fitin]): # 如果随机数小于fitin位的累计概率和，就将fitin位的个体选中
#             newpop[newin] = pop[fitin]
#             newin = newin + 1
#         else:
#             fitin = fitin + 1
#     # 这里注意一下，因为random.random()不会大于1，所以保证这里的newpop规格会和以前的一样
#     # 而且这个pop里面会有不少重复的个体，保证种群数量一样
#     # 之前是看另一个人的程序，感觉他这里有点bug，要适当修改
#     # pop = newpop[:]  # 选中的个体作为新种群
#     pop = copy.deepcopy(newpop[:])
#     popindex_500 = []  # 存储索引，也就是十进制小区数
#     popindex = {}
#     for i in range(pop_size):  # 30，应改为每6位计算一次值，即为索引  #2018/12/18
#         a = pop[i]  # len=6
#         c = []
#         decb = binary2decimal2(a, upper_limit, I)  # 二进制转10进制(可能会超过0-36范围），并转换为表现度x（约束在0-36范围）
#         c.append(decb)  # 1个十进制数
#         popindex[i] = c  # 1个十进制数是一个个体，action[i]=[1个数]
#
#         popindex_500.append(popindex[i])  # 共10000个个体，(10000,10)#500个种群，所有1的索引
#
#
# # 杂交
# def crossover(pop, pc, pop_size, upper_limit, ind, best_individual, I):
#     # 一定概率杂交，主要是杂交种群种相邻的两个个体
#     pop_len = len(pop)
#     for i in range(pop_len - 1):
#         # 随机看看是否达到杂交概率，如果小于概率，就交换
#         if (random.random() < pc):
#             # 随机选取杂交点，然后交换数组，第i个个体的后多少位与第i+1个个体的后多少位互换
#             cpoint = random.randint(0, len(pop[0]))
#             temp1 = []
#             temp2 = []
#             temp1.extend(pop[i][0:cpoint])
#             temp1.extend(pop[i + 1][cpoint:len(pop[i])])
#             temp2.extend(pop[i + 1][0:cpoint])
#             temp2.extend(pop[i][cpoint:len(pop[i])])
#             pop[i] = temp1[:]
#             pop[i + 1] = temp2[:]
#     popindex_500 = []  # 存储索引，也就是十进制小区数
#     popindex = {}
#     for i in range(pop_size):  # 30，应改为每6位计算一次值，即为索引  #2018/12/18
#         a = pop[i]  # len=6
#         c = []
#         decb = binary2decimal2(a, upper_limit, I)  # 二进制转10进制(可能会超过0-36范围），并转换为表现度x（约束在0-36范围）
#         c.append(decb)  # 1个十进制数
#         popindex[i] = c  # 1个十进制数是一个个体，action[i]=[1个数]
#
#         popindex_500.append(popindex[i])  # 共10000个个体，(10000,10)#500个种群，所有1的索引
#
#
# # 基因突变
# def mutation(pop, pm, pop_size, upper_limit, ind, best_individual, I):
#     px = len(pop)
#     py = len(pop[0])
#     # 每条染色体随便选一个杂交
#     for i in range(px):
#         if (random.random() < pm): # 如果随机数小于概率，就变异
#             mpoint = random.randint(0, py - 1) # 随机选择一个位置变异
#             if (pop[i][mpoint] == 1):
#                 pop[i][mpoint] = 0
#             else:
#                 pop[i][mpoint] = 1
#     pop[ind] = best_individual
#     popindex_500 = []  # 存储索引，也就是十进制小区数
#     popindex = {}
#     for i in range(pop_size):  # 30，应改为每6位计算一次值，即为索引  #2018/12/18
#         a = pop[i]  # len=6
#         c = []
#         decb = binary2decimal2(a, upper_limit, I)  # 二进制转10进制(可能会超过0-36范围），并转换为表现度x（约束在0-36范围）
#         c.append(decb)  # 1个十进制数
#         popindex[i] = c  # 1个十进制数是一个个体，action[i]=[1个数]
#         popindex_500.append(popindex[i])  # 共10000个个体，(10000,10)#500个种群，所有1的索引


# 轮赌法选择
def selection(pop, fit_value):
    # https://blog.csdn.net/pymqq/article/details/51375522

    p_fit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    # 归一化，使概率总和为1
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit) # 概率
    # 概率求和排序

    # https://www.cnblogs.com/LoganChen/p/7509702.html
    cum_sum(p_fit_value) # 累计概率求和
    pop_len = len(pop)
    # 类似搞一个转盘吧下面这个的意思
    ms = sorted([random.random() for i in range(pop_len)]) #产生pop_len个0-1之间的随机数
    fitin = 0
    newin = 0
    newpop = pop[:]
    # 转轮盘选择法
    while newin < pop_len:
        # 如果这个概率大于随机出来的那个概率，就选这个
        if (ms[newin] < p_fit_value[fitin]): # 如果随机数小于fitin位的累计概率和，就将fitin位的个体选中
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    # 这里注意一下，因为random.random()不会大于1，所以保证这里的newpop规格会和以前的一样
    # 而且这个pop里面会有不少重复的个体，保证种群数量一样

    # 之前是看另一个人的程序，感觉他这里有点bug，要适当修改
    pop = newpop[:]  # 选中的个体作为新种群


# 杂交
def crossover(pop, pc):
    # 一定概率杂交，主要是杂交种群种相邻的两个个体
    pop_len = len(pop)
    for i in range(pop_len - 1):
        # 随机看看是否达到杂交概率，如果小于概率，就交换
        if (random.random() < pc):
            # 随机选取杂交点，然后交换数组，第i个个体的后多少位与第i+1个个体的后多少位互换
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1[:]
            pop[i + 1] = temp2[:]


# 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    # 每条染色体随便选一个杂交
    for i in range(px):
        if (random.random() < pm): # 如果随机数小于概率，就变异
            mpoint = random.randint(0, py - 1) # 随机选择一个位置变异
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1