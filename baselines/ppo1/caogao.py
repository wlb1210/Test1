#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wlb time:2020/4/2
# 计算执行动作后的统计量。根据reward的定义选择对应的统计量。
#         self.throughput_all = self.throughput_all + self.throughput        # 累计吞吐量
#         self.cumul_time = calculate_time(self.Data, self.Time)                # 等待时间
#         self.cumul_time_average = self.cumul_time / sum(sum(self.Data))      # 平均等待时间
#         self.every_cell_cumul_time = calculate_every_cell_time(self.Data, self.Time)  # 37个小区的等待时间
#         # for i in range(37):
#         #     # self.every_cell_cumul_time[i] = sum((self.Data*self.Time)[:, i])
#         #     self.every_cell_cumul_time[i] = self.every_cell_cumul_time[i]
#         self.delay_all = self.delay_all + self.cumul_time              # 累计时延
#         self.delay_average_all = self.delay_average_all + self.cumul_time_average  # 平均累计时延
#         for i in range(37):
#             self.every_cell_delay_all[i] = self.every_cell_delay_all[i] + self.every_cell_cumul_time[i] # 37个小区的分别的累计时延
#         for i in range(37):
#             self.every_cell_delay_average_all[i] = self.every_cell_delay_average_all[i] / sum(self.Data[i,:]) # 37个小区的分别的平均累计时延，用于观察实时数据包时延公平性
#         print("throughput_all:", self.throughput_all)
#         print("cumul_time:", self.cumul_time)
#         print("cumul_time_average:", self.cumul_time_average)
#         print("every_cell_cumul_time:", self.every_cell_cumul_time)
#         print("delay_all:", self.delay_all)
#         print("delay_average_all:", self.delay_average_all)
#         print("every_cell_delay_all:", self.every_cell_delay_all)
#         print("every_cell_delay_average_all:", self.every_cell_delay_average_all)


# def Caculate_time(self, Data, Time):  # Data.shape:[None,37]
#     self.cumul_time = calculate_time(Data, Time)  # 等待时间
#     self.cumul_time_average = self.cumul_time / sum(sum(Data))  # 平均等待时间
#     self.every_cell_cumul_time = calculate_every_cell_time(Data, Time)  # 37个小区的等待时间
#     # self.delay_all = self.delay_all + self.cumul_time  # 累计时延
#     # self.delay_average_all = self.delay_average_all + self.cumul_time_average  # 平均累计时延
#     # for i in range(37):
#     #     self.every_cell_delay_all[i] = self.every_cell_delay_all[i] + self.every_cell_cumul_time[i]  # 37个小区的分别的累计时延
#     # for i in range(37):
#     #     self.every_cell_delay_average_all[i] = self.every_cell_delay_all[i] / sum(Data[:,i])  # 37个小区的分别的平均累计时延，用于观察实时数据包时延公平性
#     print("cumul_time:", self.cumul_time)
#     print("cumul_time_average:", self.cumul_time_average)
#     print("every_cell_cumul_time:", self.every_cell_cumul_time)
#     # print("delay_all:", self.delay_all)
#     # print("delay_average_all:", self.delay_average_all)
#     # print("every_cell_delay_all:", self.every_cell_delay_all)
#     # print("every_cell_delay_average_all:", self.every_cell_delay_average_all)
#     # return self.cumul_time, self.cumul_time_average, self.every_cell_cumul_time, self.delay_all,self.delay_average_all,\
#     #        self.every_cell_delay_all, self.every_cell_delay_average_all
#
#     return self.cumul_time, self.cumul_time_average, self.every_cell_cumul_time