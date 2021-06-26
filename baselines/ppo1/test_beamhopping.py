#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wlb time:2020/3/8
import heapq
import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import numpy as np
import gym
import matplotlib.pyplot as plt

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


ON_TRAIN = None

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
        train(num_timesteps=args.num_timesteps, seed=args.seed, max_iters=max_iters, model_path=args.model_path)
    else:
        print("test@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # construct the model object, load pre-trained model and render
        # pi = train(num_timesteps=1, seed=args.seed, max_iters=max_iters, model_path=args.model_path)
        pi = train(num_timesteps=1, seed=args.seed, max_iters=1, model_path=args.model_path)
        # pi = train(num_timesteps=1, seed=args.seed)
        U.load_variables('saved_networks12/savednetwork/save1_10_5_12000.m')
        # U.load_state(args.model_path)
        # env = make_mujoco_env('Humanoid-v2', seed=0)


        # 画reward
        # env = BeamhoppingEnv()
        # ob = env.reset()
        # # print("reset", ob)
        # obreset_processed = np.zeros([4, 37])
        # obreset_processed[0] = ob[0:37]
        # obreset_processed[1] = ob[37:74]
        # obreset_processed[2] = ob[74:111]
        # obreset_processed[3] = ob[111:149]
        # # print("obreset_processed:", obreset_processed)
        # ob_sum = np.sum(obreset_processed, axis=0, keepdims=True)
        # print("ob_sum:", np.transpose(ob_sum))
        # REW = []
        # epoch = 0
        # while epoch<100:
        #     # print("################################################################epoch:", epoch)
        #     epoch = epoch + 1
        #     Throughput = []
        #     # REW = []
        #     t = 0
        #     cumul_rew = 0
        #     delay = []
        #     done = False
        #     while t<512:
        #         # print("t", t)
        #         t = t + 1
        #         a = pi.act(stochastic=True, ob=ob)[0]
        #         # print("a:", a)
        #         action = heapq.nlargest(10, range(len(a)), a.take)
        #         print("run_action:", action)
        #         action_ob = ob_sum[0][action]
        #         print("action_ob:", action_ob)
        #         ob, rew, done, cumul_time, throughput = env.step(a)
        #         cumul_rew += rew
        #         # print("ob:", ob)
        #         ob_processed = np.zeros([4,37])
        #         # for i in range(39):
        #         #     ob_processed[i] = ob[i*37:(i+1)*37]
        #         #
        #         ob_processed[0] = ob[0:37]
        #         ob_processed[1] = ob[37:74]
        #         ob_processed[2] = ob[74:111]
        #         ob_processed[3] = ob[111:149]
        #         # ob_processed[39] = ob[39*37:(40*37+1)]
        #         # print("ob_processed:", ob_processed)
        #         ob_sum = np.sum(ob_processed, axis=0, keepdims=True)
        #         print("ob_sum:", np.transpose(ob_sum))
        #     REW.append(cumul_rew)
        # plt.plot(np.arange(len(REW)), REW)
        # plt.xlabel('times');plt.ylabel('REW')
        # plt.show()




        # 画throughput
        env = BeamhoppingEnv()
        ob = env.reset()
        # print("reset", ob)
        obreset_processed = np.zeros([4, 37])
        obreset_processed[0] = ob[0:37]
        obreset_processed[1] = ob[37:74]
        obreset_processed[2] = ob[74:111]
        obreset_processed[3] = ob[111:149]
        # print("obreset_processed:", obreset_processed)
        ob_sum = np.sum(obreset_processed, axis=0, keepdims=True)
        # print("ob_sum:", np.transpose(ob_sum))
        Throughput = []
        ACTION = []
        t = 0
        delay = []
        while True:
            t = t + 1
            a = pi.act(stochastic=True, ob=ob)[0]
            # print("a:", a)
            action = heapq.nlargest(10, range(len(a)), a.take)
            action.sort()
            print("action:", action)
            # print("action.sort:", action.sort())
            if action in ACTION:
                print("#####################################################")
            ACTION.append(action)
            action_ob = ob_sum[0][action]
            # print("action_ob:", action_ob)
            ob, rew, done, cumul_time, throughput = env.step(a)
            # print("ob:", ob)
            ob_processed = np.zeros([4, 37])
            # for i in range(39):
            #     ob_processed[i] = ob[i*37:(i+1)*37]
            #
            ob_processed[0] = ob[0:37]
            ob_processed[1] = ob[37:74]
            ob_processed[2] = ob[74:111]
            ob_processed[3] = ob[111:149]
            # ob_processed[39] = ob[39*37:(40*37+1)]
            # print("ob_processed:", ob_processed)
            print("rew:", rew)
            # print("cumul_time:", cumul_time)
            # print("throughput:", throughput)
            ob_sum = np.sum(ob_processed, axis=0, keepdims=True)
            # print("ob_sum:", np.transpose(ob_sum))
            Throughput.append(throughput)
            delay.append(cumul_time)
            # de = cumul_time - 302000
            # if t > 100:
            #     delay.append(de)

            # env.render()
            # if done:
            if t==1000:
                break
                # ob = env.reset()
        plt.plot(np.arange(len(Throughput)), Throughput)
        plt.xlabel('times');plt.ylabel('throughput')
        plt.show()
        # plt.plot(np.arange(len(delay)), delay)
        # plt.xlabel('times');plt.ylabel('delay')
        # plt.show()


if __name__ == '__main__':
    main()