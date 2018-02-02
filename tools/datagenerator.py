# Data Generator

import numpy as np
# from collections import Counter
from tools.gbase import GBase
from tools.probprog import ProbProg
# import scipy.io
# import csv
# import pickle
from collections import Counter


class DataGenerator:
    def __init__(self):
        self.gB = GBase()
        self.pp = ProbProg()

    def load_routine(self, rNum, transM=[], startS=[], nodes=[], groups=[]):
        if (len(transM) == 0):
            transM, startS, self.nodes, self.groups = self.gB.load(rNum)
        else:
            self.nodes = nodes
            self.groups = groups

        self.rNum = rNum
        self.gP = np.ones(self.groups) / float(self.groups)
        self.sS = startS
        self.tM = transM
        self.key_reg = dict()
        for i in range(self.nodes + 1):
            self.key_reg[i] = i

    def generate(self, nP):
        gMem = []
        while len(set([itm for l in gMem for itm in l])) < self.groups:
            # init bookkeeping
            gMem = []
            gMem_cl = []
            cMem = []
            data = []
            data_cl = []
            d = 0
            data_seq = []
            gt_seq = []
            data_seq_cl = []
            gt_seq_cl = []

            # draw starting mixture and first point in this mixture
            g = self.pp.draw(self.gP)
            d = self.pp.draw(self.sS[g, :])
            gt_seq.append(g)
            data_seq.append(d)
            count = 1

            # draw n points from the mixture of graphs
            num_data = 1
            while (num_data < nP):
                d = self.pp.draw(self.tM[d, :, g])
                # if sequence ends sample mixture the next sequence is generated from
                if (d == self.nodes):
                    # do bookkeeping
                    if count > 4:
                        data_cl.append(data_seq_cl)
                        gMem_cl.append(gt_seq_cl)
                        data_seq_cl = []
                        gt_seq_cl = []
                        if (np.random.binomial(1, 0.5)):
                            cMem.append(count)
                            num_data += count
                            count = 0
                            data.append(data_seq)
                            gMem.append(gt_seq)
                            data_seq = []
                            gt_seq = []
                    else:
                        num_data -= count
                        data_seq = []
                        gt_seq = []
                        count = 0
                    # sample next mixture and first point of sequence
                    g = self.pp.drawUni(self.groups)
                    d = self.pp.draw(self.sS[g, :])
                    gt_seq.append(g)
                    data_seq.append(d)
                    gt_seq_cl.append(g)
                    data_seq_cl.append(d)
                else:
                    gt_seq.append(g)
                    data_seq.append(d)
                    gt_seq_cl.append(g)
                    data_seq_cl.append(d)
                # num_data += 1
                count += 1
            # store additional statistics
            cMem.append(count)
            cMem.reverse()
            # data = data[:nP, :]
            # z = gMem
            # num_alphabet = self.nodes + 1
            # nump = nP

        print(Counter([itm for l in gMem for itm in l]))

        return data, gMem, data_cl, gMem_cl, self.key_reg
        # pickle.dump([data, gMem, nump, num_alphabet, self.key_reg], open('../../data/files/synth_data_large' + str(self.rNum) + '.pickle', 'wb'))
        # with open("../../data/files/synth_data_large" + str(self.rNum) + ".csv", "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(data)

        # with open("../../data/files/synth_data_large" + str(self.rNum) + "_gt.csv", "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(gMem)

    # def ratio(self):
    #     print(Counter(self.z))

# dg = DataGenerator()
# dg.load_routine(3)
# dg.generate(250000)
