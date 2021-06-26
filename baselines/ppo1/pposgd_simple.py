from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import matplotlib.pyplot as plt
import copy


def traj_segment_generator(pi, env, horizon, stochastic, max_iters):
    # U.load_variables('saved_networks2/save3-3400.m')
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode 标记是否处于情节的第一步
    ob, Data0, Time0 = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    cur_ep_cumultime = 0
    cur_ep_throughput = 0

    cur_it_r = 0
    all_it_r = []  # 存放每次iters的reward
    last_all_it_r = []


    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    # print("$$$$$$$$$$$$$$$$$$$$$$$")
    # print("ac:", ac)
    acs = np.array([ac for _ in range(horizon)])
    # print("acs:", acs)
    prevacs = acs.copy()

    # checkpoint = tf.train.get_checkpoint_state("networks")
    # # print(checkpoint)
    #
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     U.loadnet(checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")
    # U.load_variables('save.m')

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0: #判断已到达最大Iters步数 弹出
            #yield：generator(生成器）：相当于return
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # import heapq
        # action = heapq.nlargest(10, range(len(ac)), ac.take)

        ob, rew, rew2, new, cumul_time, throughput, Data1, Time1 = env.step(ac, Data0, Time0)
        Data0 = copy.deepcopy(Data1)
        Time0 = copy.deepcopy(Time1)

        rews[i] = rew
        cur_it_r += rew
        cur_ep_ret += rew # 累计回报
        cur_ep_len += 1 # 更新步数
        cur_ep_cumultime += cumul_time
        cur_ep_throughput += throughput
        # print("cur_it_r:", cur_it_r)
        # print("1cur_ep_cumultime:", cur_ep_cumultime)
        # print("1cur_ep_throughput:", cur_ep_throughput)
        # print("1cur_ep_everage_cumultime:", cur_ep_cumultime / 512)
        # print("1cur_ep_everage_throughput:", cur_ep_throughput / 512)

        # 判断是否终止，若终止则更新状态
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_cumultime = 0
            cur_ep_throughput = 0
            ob, Data0, Time0 = env.reset()

        # if ((t>=(horizon-50) * (max_iters-1))&(t % horizon==0)):
        #     last_all_it_r.append(cur_it_r)

        elif (t % horizon==0 and t!=0):
            all_it_r.append(cur_it_r)
            cur_it_r=0
            cur_ep_cumultime = 0
            cur_ep_throughput = 0



        # plot每次iters的reward曲线
        # if t==(horizon * (max_iters)):
        # if (t == horizon*5000):
        #     print("save model5000##################################################################################")
        # print("load model2000##################################################################################")
        # U.save_variables('saved_networks12/savednetwork/save1_10_5_12000.m')
        if (t % (horizon*1000)==0 and t!=0):
            print("save model###################################################################################")
            U.save_variables('save.m')
            plt.plot(np.arange(len(all_it_r)), all_it_r)
            plt.xlabel('Iteration');plt.ylabel('Moving averaged iters reward')
            plt.show()
            # plt.plot(np.arange(len(last_all_it_r)), last_all_it_r)
            # plt.xlabel('Iteration');plt.ylabel('Moving averaged iters reward')
            # plt.show()
            # U.save_variables('save.m')
        if t== (horizon * (max_iters)-1):
            print("cur_it_r:", cur_it_r)
            print("cur_ep_cumultime:", cur_ep_cumultime)
            print("cur_ep_throughput:", cur_ep_throughput)
            print("cur_ep_everage_cumultime:", cur_ep_cumultime/512)
            print("cur_ep_everage_throughput:", cur_ep_throughput / 512)
            print("cumul_time:", cumul_time)
            print("throughput:", throughput)
            plt.plot(np.arange(len(all_it_r)), all_it_r)
            plt.xlabel('Iteration');plt.ylabel('Moving averaged iters reward')
            plt.show()
            # plt.plot(np.arange(len(meanlosses_iters)), meanlosses_iters)
            # plt.xlabel('Iters')
            # plt.ylabel('Moving averaged iters meanlosses')
            # plt.show()

            # 写入excel




        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    使用TD（λ）估算器计算目标值，并使用GAE（lambda）估算优势
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update 每个actor每次更新的时间步长
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff：熵多项式系数
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return 经验回报

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd) # 计算新老策略kl散度
    ent = pi.pd.entropy() # 策略分布的熵
    meankl = tf.reduce_mean(kloldnew) # 平均KL
    meanent = tf.reduce_mean(ent) # 平均熵
    pol_entpen = (-entcoeff) * meanent # 策略分布的熵作为回报，一个参数

    meanlosses_iters = []  # 存放每次iters产生的meanlosses

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold 新老策略的概率比率
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret)) #值函数估计与经验累积回报之间的error，用两者差的平方均值表示
    total_loss = pol_surr + pol_entpen + vf_loss # 总loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables() #网络中的所有可训练参数
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)]) #计算loss和相对策略参数的梯度
    adam = MpiAdam(var_list, epsilon=adam_epsilon) #值估计网络优化器

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv) #新旧策略的迭代
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses) #根据轨迹中的信息计算loss等信息 [状态，动作，目标优势，经验回报，学习率乘数]

    #将当前得网络参数广播给其他进程，初始化优化器
    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, max_iters=max_iters, stochastic=True) # 轨迹生成器

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time() #开始时间
    # print("tstart:", tstart)
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths 情节长度的滚动缓冲区
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards 情节奖励的滚动缓冲
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])!=1, "Only one time constraint permitted"
    # assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    # 主训练循环
    while True:
        if callback: callback(locals(), globals()) # 调用回调函数
        # 若callback存在，则根据步数、episode数和循环次数判断是否要退出循环。默认最多500W步。
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant': # 若步长参数为常量，当前学习率乘数设置为1
            cur_lrmult = 1.0
        elif schedule == 'linear': # 若步长参数为线性，设置目前学习率乘数
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError # 收集没有实施错误

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__() # 使用上面的轨迹生成函数得到轨迹，相关信息以dict形式放在seg中
        add_vtarg_and_adv(seg, gamma, lam) # 估计统一化advantage函数

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"] # 分别为状态、动作、advantage函数估计、值函数估计，获得训练一次iters的值
        vpredbefore = seg["vpred"] # predicted value function before udpate 在更新前预测价值函数
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate 标准化
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), deterministic=pi.recurrent) # 数据集
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy 为状态作滑动平均，作为策略网络的输入

        assign_old_eq_new() # set old parameter values to new parameter values 将新策略参数赋值到旧策略参数，用于网络更新

        # Optimizing...:为下一次iters训练进行网络更新，生成新网络
        # optim_epochs*(timesteps_per_actorbatch/optim_batchsize):每次iters里面网络更新的次数
        # optim_batchsize:minibatch的样本数
        # timesteps_per_actorbatch:总样本数
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names)) # 打印变量名字
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize): # 每抽样optim_batchsize个样本循环一次，更新一次优化器
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult) # 计算新loss和网络参数梯度
                adam.update(g, optim_stepsize * cur_lrmult) # 更新优化器
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        # Evaluating losses...:为了Plot训练指标，计算用到的数据是本次迭代产生的数据（本次训练的网络是上次迭代最终更新的网络）
        # 在缓存里调取的，与本次Iters中前面的Optimizing...更新的网络无关
        logger.log("Evaluating losses...")
        losses = []
        loss_pol_surr = []
        loss_pol_entpen = []
        loss_vf_loss = []
        for batch in d.iterate_once(optim_batchsize): # 每抽样optim_batchsize个样本循环一次
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)

        meanlosses,_,_ = mpi_moments(losses, axis=0) # 每次iters的Loss
        # loss_pol_surr.append(meanlosses[0])
        # loss_pol_entpen.append(meanlosses[1])
        # loss_vf_loss.append(meanlosses[2])

        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        meanlosses_iters.append(meanlosses[2])

        # 画meanlosses图
        # if iters_so_far >= max_iters:
        if (iters_so_far%1000 == 0)&(iters_so_far >= max_iters):
            plt.plot(np.arange(len(meanlosses_iters)), meanlosses_iters)
            plt.xlabel('Iters')
            plt.ylabel('Moving averaged iters meanlosses')
            plt.show()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
