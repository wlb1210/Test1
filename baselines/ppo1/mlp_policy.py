from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space) # 将动作建立为对应的概率分布,根据动作空间类型，生成相应的概率分布类型对像，默认示例中这里为DiagGuass
        sequence_length = None

        # 状态的创建：状态建立的时候 为了避免重复创建同名称的状态， 利用了 get_placeholder 这个函数创建。创建的时候将所有的placeholder
        # 都放入了一个字典里 类型为{“名称”：placeholder，数据类型，数据维度}。 创建好的状态都可以直接用,函数get_placeholder_cached 来进行调用。
        # 网络输入为状态序列
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        # 对ob归一化为obz
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape) # std:标准差

        # value function 网络： 输入为状态 输出为vpred（预测价值）  critic网络
        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0) # clip
            last_out = obz
            # 该网络为双头网络，一头输出值函数（Value function）的估计：另一头输出动作
            # 这里经过若干层（几层通过num_hid_layers指定）FC层，然后输出值函数估计
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        # policy网络： 输入为状态 输出为动作    actor网络
        with tf.variable_scope('pol'):
            last_out = obz
            # 经过几层FC层，输出动作
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):  # 如果是高斯变量 那么网络输出的是均值
                # 网络最后一层FC输出均值，结合标准差，形成动作分布的状态。
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:  # 如果不是高斯变量 那么网络输出的是概率参数
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam) # 从分布参数得到分布

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        # stochastic：决定策略是否随机，即网络输出动作分布后从中采样
        # 还是取值后。后者对一特定分布就是确定的。最后将全部计算封装成一个函数_act（）
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        # print("ob:", ob, ob.shape)
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

