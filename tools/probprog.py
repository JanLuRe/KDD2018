# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:05:18 2015

@author: jr
"""
import numpy as np
import copy as cp


class ProbProg:
    def draw(self, dist, times=1):
        dist_temp = cp.deepcopy(dist)
        dist /= np.nansum(dist)
        if np.abs(np.sum(dist) - 1.0) > 1E-5:
            print(dist_temp)
            print(dist)
        # assert(np.abs(np.sum(dist) - 1.0) < 1E-5)
        draw = np.random.rand(times)
        res = [np.sum(d > np.cumsum(dist)) for d in draw]
        return int(res) if times > 1 else res[0]

    def drawUni(self, num, times=1):
        draw = np.random.randint(0, num, times)
        return draw if times > 1 else draw[0]
