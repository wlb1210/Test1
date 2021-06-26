# import random
#
# import gym
# import numpy as np
# # import time
# #
# #
# # print(time.time())
# # ob = [0,1,2,3]
# # ob_processed = np.zeros([2,2])
# #
# # print(ob[0:2])
# # print(ob_processed[0])
# # ob_processed[0] = ob[0:2]
# # ob_processed[1] = ob[2:4]
# # print(ob_processed)
# a = [0,1,2]
# print(a[::-1])
# print(a[::1])
#
# # n=len(ac)
# # for i in range(0,n):
# #   for j in range(i+1,n):
# #     if(ac[i]==ac[j]):
# #         print(ac[j],j)
# #         ac[j] = (random.sample(range(0, 37), 1))[0]
# # print(ac)
# # bc = list(range(37))
# # print(bc)
# # print (list(set(bc).difference(set(ac))))
# # bc = np.array(bc)
# # ac = np.concatenate([ac, np.zeros(30)], axis=0)
# # ac = np.array(ac)
# # print(bc, ac)
# # print(bc-ac)
# # ac = [1,2,1,3,4,2,4]
# # action = ac
# # n = len(ac)
# # for i in range(0, n):
# #     for j in range(i + 1, n):
# #         # if (action[i] == action[j]):
# #         if (action[i] == action[j]):
# #             print(action[j], j)
# #             l = list(set(list(range(37))).difference(set(action)))
# #             print(l)
# #             action[j] = (random.sample(l, 1))[0]
# #             print(action[j])
# # print(action)
from gym import spaces


def __init__(self, maxUmbralAstral):
    # Print debug
    self.debug = False

    # Outer bound for Astral Fire and Umbral Ice
    BLM.MAXUMBRALASTRAL = maxUmbralAstral

    # Available buffs
    self.BUFFS = []

    # Maximum time available
    self.MAXTIME = 45

    self.HELPER = BLM.Helper()

    # Available abilities
    self.ABILITIES = [
        BLM.Ability("Blizzard 1", 180, 6, 2.5, 2.49, self.HELPER.UmbralIceIncrease, BLM.DamageType.Ice, self.HELPER),
        # 480
        BLM.Ability("Fire 1", 180, 15, 2.5, 2.49, self.HELPER.AstralFireIncrease, BLM.DamageType.Fire, self.HELPER),
        # 1200
        BLM.Ability("Transpose", 0, 0, 0.75, 12.9, self.HELPER.SwapAstralUmbral, BLM.DamageType.Neither, self.HELPER),
        BLM.Ability("Fire 3", 240, 30, 3.5, 2.5, self.HELPER.AstralFireMax, BLM.DamageType.Fire, self.HELPER),  # 2400
        BLM.Ability("Blizzard 3", 240, 18, 3.5, 2.5, self.HELPER.UmbralIceMax, BLM.DamageType.Ice, self.HELPER),  # 2400
        BLM.Ability("Fire 4", 260, 15, 2.8, 2.5, None, BLM.DamageType.Fire, self.HELPER)]  # 2400

    # State including ability cooldowns, buff time remaining, mana, and Astral/Umbral
    self.initialState = np.array([0] * (len(self.ABILITIES) + len(self.BUFFS)) + [BLM.MAXMANA] + [0])

    self.state = self._reset()

    # What the learner can pick between
    self.action_space = spaces.Discrete(len(self.ABILITIES))

    # What the learner can see to make a choice (cooldowns and buffs)
    self.observation_space = spaces.MultiDiscrete(
        [[0, 180]] * (len(self.ABILITIES) + len(self.BUFFS)) + [[0, BLM.MAXMANA]] + [[-3, 3]])
