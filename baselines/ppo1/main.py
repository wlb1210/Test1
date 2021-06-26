#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wlb time:2020/3/17
import gym

env = gym.make('beamhopping-v0')
env.reset()
env.render()
env.close()
