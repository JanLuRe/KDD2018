import numpy as np


class GBase:
    def load(self, type):
        if (type == 0):
            self.genG0()
        if (type == 1):
            self.genG1()
        if (type == 2):
            self.genG2()
        if (type == 3):
            self.genG3()
        if (type == 4):
            self.genGF()
        if (type == 5):
            self.gen_toy()
        return self.transM, self.startS, self.nodes, self.groups

    def init(self):
        self.transM = np.zeros((self.nodes, self.nodes + 1, self.groups))
        self.startS = np.zeros((self.groups, self.nodes))
        self.eta = np.zeros((self.groups, self.nodes))

    def gen_toy(self):
        self.nodes = 3
        self.groups = 3
        self.init()

        self.startS[0, 0] = 1.0
        self.startS[1, 1] = 1.0
        self.startS[2, 2] = 1.0

        self.transM[0, :, 0] = np.array([.01, 0.987, 0.002, 0.001])
        self.transM[1, :, 0] = np.array([.001, 0.001, 0.997, 0.001])
        self.transM[2, :, 0] = np.array([.001, 0.001, 0.001, 0.997])

        self.transM[0, :, 1] = np.array([.001, 0.998, 0.001, 0.0])
        self.transM[1, :, 1] = np.array([.001, 0.001, 0.001, 0.997])
        self.transM[2, :, 1] = np.array([.001, 0.998, 0.001, 0.0])

        self.transM[0, :, 2] = np.array([.001, 0.998, 0.001, 0.0])
        self.transM[1, :, 2] = np.array([.001, 0.498, 0.001, 0.498])
        self.transM[2, :, 2] = np.array([.001, 0.998, 0.001, 0.0])


    def genGF(self):
        self.nodes = 9
        self.groups = 3
        self.init()

        #                              0  1   2  3  4  5  6  7  8  E
        self.startS[0, :] = np.array([.3, 0, .7, 0, 0, 0, 0, 0, 0])
        self.transM[0, np.array([1, 2]), 0] = np.array([0.6, 0.4])
        self.transM[1, np.array([1, 2, 3]), 0] = np.array([0.3, 0.6, 0.1])
        self.transM[2, np.array([0]), 0] = np.array([1.0])
        self.transM[3, np.array([4, 5]), 0] = np.array([0.2, 0.8])
        self.transM[4, np.array([3, 5]), 0] = np.array([0.5, 0.5])
        self.transM[5, np.array([4, 9]), 0] = np.array([0.9, 0.1])

        #                                0    1    2    3    4    5    6    7    8    E
        self.startS[1, :] = np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0])
        self.transM[3, np.array([4, 5]), 1] = np.array([0.2, 0.8])
        self.transM[4, np.array([3, 5]), 1] = np.array([0.5, 0.5])
        self.transM[5, np.array([4, 9]), 1] = np.array([0.9, 0.1])
        self.transM[6, np.array([8])   , 1] = np.array([1.0])
        self.transM[7, np.array([3, 6]), 1] = np.array([0.1, 0.9])
        self.transM[8, np.array([7])   , 1] = np.array([1.0])

        #                                0    1    2    3    4    5    6    7    8    E
        self.startS[2,:]   = np.array([  0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1.0])
        self.transM[0, np.array([3]), 2]    = np.array([1.0])
        self.transM[3, np.array([8, 9]), 2] = np.array([0.9, 0.1])
        self.transM[5, np.array([0]), 2]    = np.array([1.0])
        self.transM[8, np.array([5]), 2]    = np.array([1.0])



    def genG0(self):
        self.nodes  = 10
        self.groups =  3
        self.init()

        #                                0    1    2    3    4    5    6    7    8    9    E
        self.startS[0,:]   = np.array([  0  , .3 , .7 , 0  , 0  , 0  , 0  , 0  , 0  , 0])
        self.transM[0, np.array([0, 10]), 0] = np.array([0.6, 0.4])
        self.transM[1, np.array([0,  2]), 0] = np.array([0.1, 0.9])
        self.transM[2, np.array([0,  1]), 0] = np.array([0.05, 0.95])

        #                                0    1    2    3    4    5    6    7    8    9    E
        self.startS[1,:]   = np.array([  0  , 0  , 0  , 1.0, 0  , 0  , 0  , 0  , 0  , 0])
        self.transM[3, np.array([3, 4, 5]), 1] = np.array([0.3, 0.6, 0.1])
        self.transM[4, np.array([5])      , 1] = np.array([1.0])
        self.transM[5, np.array([4, 10])  , 1] = np.array([0.6, 0.4])

        #                                0    1    2    3    4    5    6    7    8    9    E
        self.startS[2,:]   = np.array([  0  , 0  , 0  , 0  , 0  , 0  , 1.0, 0  , 0  , 0])
        self.transM[6, np.array([7, 8])      , 2] = np.array([0.6, 0.4])
        self.transM[7, np.array([6])         , 2] = np.array([1.0])
        self.transM[8, np.array([6, 7, 8, 9]), 2] = np.array([0.2, 0.55, 0.1, 0.15])
        self.transM[9, np.array([9, 10])     , 2] = np.array([0.7, 0.3])


    def genG1(self):
        self.nodes  = 11
        self.groups =  3
        self.init()

        #                                0    1    2    3    4    5    6    7    8    9    A
        self.startS[0,:]   = np.array([ 1.0 , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0])
        self.transM[ 0, np.array([1, 2])   , 0] = np.array([.45, .55])
        self.transM[ 1, np.array([3])      , 0] = np.array([1.0])
        self.transM[ 2, np.array([0])      , 0] = np.array([1.0])
        self.transM[ 3, np.array([1, 2, 11]), 0] = np.array([.55, .4 , .05])

        #                                0    1    2    3    4    5    6    7    8    9    A
        self.startS[1,:]   = np.array([  0  , 0  , 0  , 0  , 1.0, 0  , 0  , 0  , 0  , 0  , 0])
        self.transM[ 4, np.array([5, 6])   , 1] = np.array([.45, .55])
        self.transM[ 5, np.array([7])      , 1] = np.array([1.0])
        self.transM[ 6, np.array([4])      , 1] = np.array([1.0])
        self.transM[ 7, np.array([5, 6,11]), 1] = np.array([.55, .4 , .05])

        #                                0    1    2    3    4    5    6    7    8    9    A
        self.startS[2,:]   = np.array([  0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 1.0, 0  , 0])
        self.transM[ 8, np.array([9,10]), 2] = np.array([.45, .55])
        self.transM[ 9, np.array([8])   , 2] = np.array([1.0])
        self.transM[10, np.array([9,11]), 2] = np.array([.95, .05])

    def genG2(self):
        self.nodes = 15
        self.groups = 6
        self.init()

        #                                0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    X
        self.startS[0, :] = np.array([.3, .7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.transM[0, np.array([1, 2, 4]), 0] = np.array([0.5, 0.4, .1])
        self.transM[1, np.array([0, 2]), 0] = np.array([0.55, 0.45])
        self.transM[2, np.array([1, 2, 3]), 0] = np.array([0.25, 0.3, 0.45])
        self.transM[3, np.array([2, 4]), 0] = np.array([0.55, 0.45])
        self.transM[4, np.array([3, 4, 15]), 0] = np.array([0.3, 0.45, 0.25])

        #                                0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    X
        self.startS[1, :] = np.array([0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.transM[5, np.array([5, 6, 7]), 1] = np.array([0.3, 0.6, 0.1])
        self.transM[6, np.array([7]), 1] = np.array([1.0])
        self.transM[7, np.array([5, 6, 15]), 1] = np.array([0.1, 0.65, 0.25])

        #                                0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    X
        self.startS[2, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])
        self.transM[8, np.array([9, 10]), 2] = np.array([0.6, 0.4])
        self.transM[9, np.array([8]), 2] = np.array([1.0])
        self.transM[10, np.array([8, 9, 10, 11]), 2] = np.array([0.2, 0.45, 0.1, 0.25])
        self.transM[11, np.array([10, 11, 15]), 2] = np.array([0.3, 0.4, 0.3])

        #                                0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    X
        self.startS[3, :] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.transM[0, np.array([7, 9]), 3] = np.array([0.1, 0.9])
        self.transM[7, np.array([9, 12]), 3] = np.array([0.75, 0.25])
        self.transM[9, np.array([0, 9, 11]), 3] = np.array([0.2, 0.3, 0.5])
        self.transM[11, np.array([7, 9]), 3] = np.array([0.65, 0.35])
        self.transM[12, np.array([11, 12, 15]), 3] = np.array([0.4, 0.3, 0.3])
#
        #                                0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    X
        self.startS[4, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
        self.transM[3, np.array([11]), 4] = np.array([1.0])
        self.transM[5, np.array([13, 14]), 4] = np.array([0.25, 0.75])
        self.transM[11, np.array([3, 5, 11]), 4] = np.array([0.1, 0.7, 0.2])
        self.transM[13, np.array([5, 13, 15]), 4] = np.array([0.55, 0.2, 0.25])
        self.transM[14, np.array([3, 14]), 4] = np.array([0.85, 0.15])

        self.startS[5, :] = np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0])
        self.transM[2, np.array([2, 3, 14]), 5] = np.array([0.4, 0.1, 0.5])
        self.transM[3, np.array([2, 4, 15]), 5] = np.array([0.4, 0.3, 0.3])
        self.transM[4, np.array([3, 4]), 5] = np.array([0.6, 0.4])
        self.transM[7, np.array([2, 4, 14]), 5] = np.array([0.2, 0.2, 0.6])
        self.transM[14, np.array([7]), 5] = np.array([1.0])

    def genG3(self):
        self.nodes = 10
        self.groups = 4
        self.init()

        self.startS[0, :] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.transM[0, np.array([1, 2, 3, 10]), 0] = np.array([.01, .6, .34, .05])
        self.transM[1, np.array([0, 1, 3]), 0] = np.array([.25, .6, .15])
        self.transM[2, np.array([1, 2, 4]), 0] = np.array([.1, .6, .3])
        self.transM[3, np.array([1, 2, 4]), 0] = np.array([.01, .7, .29])
        self.transM[4, np.array([3]), 0] = np.array([1.0])

        self.startS[1, :] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.transM[0, np.array([1, 2, 3, 10]), 1] = np.array([.74, .2, .01, .05])
        self.transM[1, np.array([0, 1, 3]), 1] = np.array([.2, .05, .75])
        self.transM[2, np.array([1, 2, 4]), 1] = np.array([.8, .15, .05])
        self.transM[3, np.array([1, 2, 4]), 1] = np.array([.29, .7, .01])
        self.transM[4, np.array([3]), 1] = np.array([1.0])

        self.startS[2, :] = np.array([0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0])
        self.transM[5, np.array([7, 9]), 2] = np.array([.7, .3])
        self.transM[6, np.array([5, 8]), 2] = np.array([.8, .2])
        self.transM[7, np.array([8, 10]), 2] = np.array([.95, .05])
        self.transM[8, np.array([6, 7]), 2] = np.array([.6, .4])
        self.transM[9, np.array([6]), 2] = np.array([1.0])

        self.startS[3, :] = np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0])
        self.transM[5, np.array([7, 8]), 3] = np.array([.2, .8])
        self.transM[6, np.array([7, 10]), 3] = np.array([.95, .05])
        self.transM[7, np.array([5, 6, 9]), 3] = np.array([.2, .1, .7])
        self.transM[8, np.array([5, 6]), 3] = np.array([.2, .8])
        self.transM[9, np.array([5]), 3] = np.array([1.0])
