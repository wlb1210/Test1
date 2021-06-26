#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wlb time:2020/3/8
import heapq
import os
import time

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import numpy as np
import gym
import matplotlib.pyplot as plt

from baselines.ppo1 import GA_37xuan10
from baselines.ppo1.env_beamhopping import BeamhoppingEnv

env = BeamhoppingEnv()



# 训练策略网络
def train(num_timesteps, seed, max_iters, model_path=None):
    # seed:（int）伪随机生成器的种子。如果没有（默认），则使用随机种子。如果要获得完全确定的结果，则必须将n_cpu_tf_sess设置为1。
    # env_id = 'Humanoid-v2'
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__() # 可能是设置多线程？？？

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    # env = RewScale(env, 0.1)
    logger.log("NOTE: reward will be scaled by a factor of 10  in logged stats. Check the monitor for unscaled reward.")

    # 获得一个新策略网络
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=512, # 每次网络更新的训练次数（每次iters的step次数）
            clip_param=0.1, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=1e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            max_iters=max_iters,
            schedule='constant', # 设置学习率
        )
    env.close()
    # if model_path:
    #     print("model_path:", model_path)
    #     print("save model##################################################################################")
    #     U.save_state(model_path)
    # print("save model##################################################################################")
    # U.save_variables('saved_networks7/save1_10_5')

    return pi

# 环境收益*scale
class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale


ON_TRAIN = True

def main():
    logger.configure()  #日志
    parser = mujoco_arg_parser() #多线程环境
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'beamhopping_policy')) #调参
    parser.set_defaults(num_timesteps=int(5e7)) #设置默认参数(训练次数）

    args = parser.parse_args() # 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中

    max_iters = 30000 # 设置总的iters次数

    # if not args.play:
    if ON_TRAIN:
        print("train$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # train the model
        # print("load model2000##################################################################################")
        # U.save_variables('saved_networks12/savednetwork/save1_10_5_12000.m')
        train(num_timesteps=args.num_timesteps, seed=args.seed, max_iters=max_iters, model_path=args.model_path)
    else:
        print("test@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed, max_iters=max_iters, model_path=args.model_path)
        # pi = train(num_timesteps=1, seed=args.seed, max_iters=max_iters)
        # pi = train(num_timesteps=1, seed=args.seed, max_iters=1, model_path=args.model_path)
        # pi = train(num_timesteps=1, seed=args.seed)
        U.load_variables('saved_networks4/save-14000.m')
        # U.load_variables('saved_networks6/save1_10_5')
        # U.load_state(args.model_path)
        # env = make_mujoco_env('Humanoid-v2', seed=0)
        # 画throughput
        env = BeamhoppingEnv()
        ob, Data0, Time0 = env.reset()
        # print("reset", np.transpose(ob))
        # print("Data0", Data0)
        obreset_processed = np.zeros([20, 37])
        for i in range(20):
            obreset_processed[i] = ob[i * 37:(i + 1) * 37]
        # obreset_processed[0] = ob[0:37]
        # obreset_processed[1] = ob[37:74]
        # obreset_processed[2] = ob[74:111]
        # obreset_processed[3] = ob[111:149]
        # print("obreset_processed:", obreset_processed)
        ob_sum = np.sum(obreset_processed, axis=0, keepdims=True)
        # print("ob_sum:", np.transpose(ob_sum))
        Throughput = []
        Throughput1 = []
        Fairness = []
        Fairness1 = []
        ACTION = []
        t = 0
        delay = []
        Time = []
        Time_GA = []
        while True:
            t = t + 1
            t1 = time.time()
            action = pi.act(stochastic=True, ob=ob)[0]
            # print("a:", a)
            # action = heapq.nlargest(10, range(len(a)), a.take)
            t2 = time.time()
            Time.append(t2-t1)
            action.sort()
            print("action:", action)
            # print("action.sort:", action.sort())
            # if action in ACTION:
            #     print("#####################################################")
            ACTION.append(action)
            # action_ob = ob_sum[0][action]
            # print("action_ob:", action_ob)
            ob, rew, reward2, done, cumul_time, throughput, Data1, Time1 = env.step(action, Data0, Time0)

            # print("ob:", np.transpose(ob), ob.shape)
            ob_processed = np.zeros([20, 37])
            for i in range(20):
                ob_processed[i] = ob[i*37:(i+1)*37]
            # print("ob_processed:", ob_processed)

            print("throughput:", throughput)
            print("Fairness", reward2)
            ob_sum = np.sum(ob_processed, axis=0, keepdims=True)
            # print("ob_sum", np.transpose(ob_sum))
            Throughput.append(throughput)
            Fairness.append(reward2)
            delay.append(cumul_time)

            t3 = time.time()
            action_GA, throughput_GA = GA_37xuan10.action(action, Data0, Time0)

            t4 = time.time()
            Time_GA.append(t4-t3)
            ob, rew, reward2_, done, cumul_time, throughput, Data2, Time2 = env.step(action_GA, Data0, Time0)
            _1,_2,_3,every_cell_cumul_time_average = env.Caculate_Time(Data2, Time2)
            fairness_GA = -(max(every_cell_cumul_time_average.values()) - min(every_cell_cumul_time_average.values()))
            import copy
            Data0 = copy.deepcopy(Data2)
            Time0 = copy.deepcopy(Time2)
            Throughput1.append(throughput_GA)
            Fairness1.append(fairness_GA)
            print("action_GA", action_GA)
            print("throughput_GA", throughput_GA)
            print("Fairness_GA", fairness_GA)
            # print("Data2", Data2)


            # env.render()
            # if done:
            if t==100:
                break
                # ob = env.reset()
        plt.plot(np.arange(len(Throughput)), Throughput)
        plt.xlabel('times');plt.ylabel('throughput')
        plt.show()
        plt.plot(np.arange(len(Throughput1)), Throughput1)
        plt.xlabel('times');plt.ylabel('throughput1')
        plt.show()
        plt.plot(np.arange(len(Fairness)), Fairness)
        plt.xlabel('times');plt.ylabel('Fairness')
        plt.show()
        plt.plot(np.arange(len(Fairness1)), Fairness1)
        plt.xlabel('times');plt.ylabel('Fairness1')
        plt.show()
        # plt.plot(np.arange(len(delay)), delay)
        # plt.xlabel('times');plt.ylabel('delay')
        # plt.show()
            # 送入GA
        # t1 = time.time()
        # action_GA, throughput_GA = GA_37xuan10.action(action, Data0, Time0)
        # t2 = time.time()
        # print("t2-t1", t2-t1)
        # Throughput1.append(throughput_GA)
        # print("action_GA", action_GA)
        # print("throughput_GA", throughput_GA)
        # plt.plot(np.arange(len(Throughput1)), Throughput1)
        # plt.xlabel('times');plt.ylabel('throughput1')
        # plt.show()


        # rl-action action: action: [1, 9, 11, 13, 18, 20, 30, 32, 33, 35]
        # throughput: 15000
        # action_GA action_GA [11, 3, 34, 21, 8, 18, 28, 33, 27, 32]
        # throughput_GA 15865
        # action_GA [9, 30, 19, 10, 17, 21, 29, 35, 28, 13]
        # action_GA [34, 11, 20, 27, 1, 25, 30, 32, 21, 9]
        # throughput_GA 16876

        Throu = np.array(Throughput)
        import pandas as pd
        data = pd.DataFrame(Throu)
        writer = pd.ExcelWriter('./Throughput.xlsx')
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()

        Throu_GA = np.array(Throughput1)
        import pandas as pd
        data = pd.DataFrame(Throu_GA)
        writer = pd.ExcelWriter('./Throughput_GA.xlsx')
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()

        Tim = np.array(Time)
        import pandas as pd
        data = pd.DataFrame(Tim)
        writer = pd.ExcelWriter('./Time.xlsx')
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()

        Tim_GA = np.array(Time_GA)
        import pandas as pd
        data = pd.DataFrame(Tim_GA)
        writer = pd.ExcelWriter('./Time_GA.xlsx')
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()

        Fair = np.array(Fairness)
        import pandas as pd
        data = pd.DataFrame(Fair)
        writer = pd.ExcelWriter('./Fairness.xlsx')
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()

        Fair_GA = np.array(Fairness1)
        import pandas as pd
        data = pd.DataFrame(Fair_GA)
        writer = pd.ExcelWriter('./Fairness_GA.xlsx')
        data.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()
        writer.close()
if __name__ == '__main__':
    main()