import numpy as np
import pickle
import copy as cp
from segmentation.dataloadunit import DataLoadUnit
from scipy.misc import logsumexp as lse
import os.path


class EvalFB:
    def __init__(self, fname, set_range):
        self.offset = 5
        self.max_order = 4
        self.global_mms = [dict()] * (self.max_order - 1)

        self.base_path = '../../data/.logs/' + fname + '/'
        self.log_path = self.base_path + 'final_' + str(set_range[0]) + '-' + str(set_range[1]) + '.pickle'
        self.data_path = self.base_path + 'train_test.pickle'
        self.data_output_path = self.base_path + 'raw_fb_data.pickle'

        if os.path.isfile(self.data_output_path):
            with open(self.data_output_path) as f:
                self.test_packages, self.key_reg, self.action_dist, self.global_mms = pickle.load(f)
        else:
            with open(self.data_path, 'rb') as f:
                clutter, self.data, self.key_reg = pickle.load(f)
            self.__process_data()
            with open(self.data_output_path, 'wb') as f:
                pickle.dump([self.test_packages, self.key_reg, self.action_dist, self.global_mms], f)
        self.rho = 2.0
        self.L = 90
        self.boundary = -1

    def run(self):
        with open(self.log_path, 'rb') as f:
            data_info, self.model = pickle.load(f)
        self.model.theta = np.array(self.model.theta)
        self.L2 = len(self.model.key_reg.keys())

        total = 0
        top_max = 5
        correct = np.zeros(top_max)
        correct_total = 0
        correct_gmm = np.zeros(self.max_order - 1)
        # print(np.bincount(self.gt) / float(len(self.gt)))
        print('Evaluating...')
        print(len(self.test_packages))
        for test in self.test_packages:

            self.__backward_bw(test[0])
            distF = self.__forward_bw(test[0])  # [-last_x_obs:], last_x_obs)

            last_obs = test[0][-1]
            predictions = np.zeros((self.L2, 1))
            theta_t = self.model.theta[:, self.boundary, :].reshape(self.L, self.L2)
            for su_id in np.arange(self.L):
                predictions += self.model.theta[su_id, last_obs, :].reshape(self.L2, 1) * distF[su_id]
                trans_prob = np.dot(self.model.pi[su_id, :-1].reshape(1, self.L), theta_t)
                assert(np.sum(trans_prob.shape) == self.L2 + 1)
                predictions += (trans_prob.reshape(self.L2, 1) * self.model.theta[su_id, last_obs, self.boundary] * distF[su_id])
            predictions /= np.sum(predictions)
            total += 1
            # argmax -> use sampling? or weight?
            for top in np.arange(top_max):
                if test[1][0] == predictions.argmax():
                    correct[top] += 1
                    correct_total += 1
                    break
                else:
                    predictions[predictions.argmax()] = -np.inf

            for order in range(1, self.max_order):
                mm = self.global_mms[order - 1]
                content = test[0][-order:]
                diff = order - len(content)
                if diff > 0:
                    content = np.append(content, np.ones(diff) * -1)
                context = str(content)
                if context in mm.keys():
                    pred = mm[context].argmax()
                else:
                    pred = self.action_dist.argmax()
                if test[1][0] == pred:
                    correct_gmm[order - 1] += 1
            print(correct_total / np.float(total))
            print(correct / np.float(total))
            print(correct_gmm / np.float(total))

        print(correct / np.float(total))
        print(correct_gmm / np.float(total))

    def __process_data(self):
        print('Loading data...')
        self.test_packages = []
        max_val = np.max(list(set([itm for l in self.data for itm in l]))) + 1
        self.action_dist = np.zeros(max_val)

        for seq in self.data:
            prev_actions = [-1] * (self.max_order - 1)
            assert(np.sum(prev_actions) == -len(prev_actions))
            total_len = len(seq)
            if total_len < 10:
                continue
            split_pos = int(np.round(np.random.rand() * (total_len - self.offset - 1.0)) + self.offset)
            seq_input = cp.deepcopy(seq[:split_pos])
            seq_gt = cp.deepcopy(seq[split_pos:])
            self.test_packages.append([seq_input, seq_gt])
            for obs_id in np.arange(len(seq) - 1):
                obs = seq[obs_id]
                assert(obs >= 0)
                self.action_dist[obs] += 1
                prev_actions.insert(0, obs)
                prev_actions.pop()
                for order in range(1, self.max_order):
                    context = str([itm for itm in prev_actions[:order]])
                    if context in self.global_mms[order - 1]:
                        self.global_mms[order - 1][context][seq[obs_id + 1]] += 1
                    else:
                        self.global_mms[order - 1][context] = np.zeros(max_val)

    def __process_data2(self):
        self.__prepare_data()
        print('Loading data...')
        self.test_packages = []
        prev_actions = [-1] * (self.max_order - 1)
        processed = []
        self.gt = []
        self.nump = 0
        self.action_dist = np.zeros(len(self.key_reg.keys()) - 1)

        for seq in self.data:
            prev_actions[:0] = [max(self.key_reg.values())]
            total_len = len(seq)
            if total_len < 10:
                continue
            # rand = np.random.rand()
            # print rand
            # print total_len
            split_pos = int(np.round(np.random.rand() * (total_len - self.offset - 1.0)) + self.offset)
            # print split_pos
            gt = seq[split_pos:]
            seq = seq[:split_pos]
            self.nump += split_pos + 1
            c_seq = []
            # print seq
            for obs in seq:
                assert(obs < max(self.key_reg.values()))
                d = cp.deepcopy(obs)

                self.action_dist[d] += 1
                for order in range(1, self.max_order):
                    mm = self.global_mms[order - 1]
                    context = str(prev_actions[:order])
                    if context in mm:
                        if d in mm[context]:
                            mm[context][d] += 1
                        else:
                            mm[context][d] = 1
                    else:
                        mm[context] = dict()
                        mm[context][d] = 1
                    self.global_mms[order - 1] = cp.deepcopy(mm)
                prev_actions.pop()
                prev_actions[:0] = [d]

                c_seq.append(cp.deepcopy(d))
            # d = self.key_reg[self.ENDSTATE]
            # c_seq.append(cp.deepcopy(d))
            processed.append(cp.deepcopy(c_seq))
            self.test_packages.append([cp.deepcopy(c_seq), cp.deepcopy(gt)])

        # self.rev_key_reg.update(reversed(i) for i in self.key_reg.items())
        self.data = cp.deepcopy(processed)

    # Baum-Welch Backward over entire sequence...
    # @profile
    def __backward_bw(self, data):
        self.msgs = []
        self.msgs.append(np.ones((self.L, 1)))
        for obs_id in np.arange(len(data) - 2, -1, -1):
            c_obs = data[obs_id:obs_id + 2]
            c_msg = self.msgs[-1]
            self.msgs.append(cp.deepcopy(self.__backward_step(c_msg, c_obs)))
        self.msgs = self.msgs[::-1]

    def __backward_step(self, msg, c_obs):
        # (-1, y) beginning of a sequence
        if c_obs[0] < 0:
            prior = np.multiply(self.model.beta.reshape(self.L, 1), self.model.theta[:, c_obs[0], c_obs[1]].reshape(self.L, 1))
            msg_new = np.multiply(prior, msg.reshape(self.L, 1))
        else:
            # probability for intra transitions (x->y or x->-1)
            intra = self.__compute_intra(c_obs)
            # (x, -1) end of a sequence
            if c_obs[1] < 0:
                msg_new = intra
            # (x, y) in a sequence
            else:
                sos = np.multiply(self.model.beta.reshape(self.L, 1), self.model.theta[:, self.boundary, c_obs[1]].reshape(self.L, 1))
                sos = np.multiply(sos, msg.reshape(self.L, 1))
                inter = np.multiply(self.__compute_intra((c_obs[0], -1)), np.dot(self.model.pi[:-1, :-1], sos).reshape(self.L, 1))
                intra = np.multiply(intra, msg.reshape(self.L, 1))
                msg_new = inter + intra
        msg_new /= np.sum(msg_new)
        return msg_new

    # @profile
    def __compute_intra(self, c_obs):
        partMsg = np.multiply(self.model.psi[:, c_obs[0]].reshape(self.L, 1), self.model.theta[:, c_obs[0], c_obs[1]].reshape(self.L, 1))
        return partMsg

    # Compute the distribution over z at node i of observed variables
    # @profile
    def __forward_bw(self, data):
        # probs in log space
        distF = self.__init_forwardBW(data[0])
        for obs_id in np.arange(1, len(data)):
            c_obs = data[obs_id]
            p_obs = data[obs_id - 1]
            msgs = self.msgs[obs_id]

            # at end of a sequence do nothing
            if (c_obs == self.boundary):
                distF = np.multiply(distF.reshape(self.L, 1), self.model.theta[:, p_obs, c_obs].reshape(self.L, 1))
                distF /= np.sum(distF)
            # in a sequence do...
            else:
                distF = self.__forward_step(c_obs, p_obs, msgs, distF)
        return distF

    def __init_forwardBW(self, data):
        ret = np.multiply(self.model.beta.reshape(self.L, 1), self.model.theta[:, self.boundary, data].reshape(self.L, 1))
        ret = np.multiply(ret, self.msgs[0].reshape(self.L, 1))
        ret /= np.sum(ret)
        return ret

    def __forward_step(self, c_obs, p_obs, msgs, prob):
        # probability that current segment ends
        # inter_intra_end = np.multiply(self.model.psi[:, p_obs].reshape(self.L, 1), self.model.theta[:, p_obs, self.boundary].reshape(self.L, 1))
        inter = np.multiply(self.model.pi[:-1, :-1], self.model.theta[:, self.boundary, c_obs].reshape(1, self.L))
        inter = np.multiply(inter, self.model.theta[:, p_obs, self.boundary].reshape(self.L, 1))
        # intra = np.multiply(self.model.psi[:, p_obs].reshape(self.L, 1), self.model.theta[:, p_obs, c_obs].reshape(self.L, 1))
        intra = np.multiply(self.model.theta[:, p_obs, c_obs].reshape(self.L, 1), prob.reshape(self.L, 1))

        # probability of transition to another su given current observation
        # combined probability
        distF = np.multiply(inter, prob.reshape(self.L, 1))
        distF = np.sum(distF, axis=0).reshape(self.L, 1) + intra
        distF = np.multiply(distF.reshape(self.L, 1), msgs.reshape(self.L, 1))
        distF += 1E-10
        distF /= np.sum(distF)
        assert(abs(np.sum(distF)) - 1 <= 1E-4)
        return distF

    def __forward_bw2(self, data):
        ll = np.multiply(self.model.theta[:, self.boundary, data[0]].reshape(self.L, 1), self.model.beta.reshape(self.L, 1))
        ll += 1E-20
        ll /= np.float(np.sum(ll))

        endNode = data[0]
        for i in (np.arange(1, len(data))):
            startNode = cp.deepcopy(endNode)
            endNode = cp.deepcopy(data[i])

            distF = np.multiply(self.model.pi.T[:-1, :-1].reshape(self.L, self.L), self.model.theta[:, self.boundary, endNode].reshape(self.L, 1))
            distF = np.multiply(distF.T, self.model.theta[:, startNode, self.boundary])
            distF = np.sum(distF, axis=0).reshape(self.L, 1)
            distF += np.multiply(self.model.theta[:, startNode, endNode].reshape(self.L, 1), self.model.psi[:, startNode].reshape(self.L, 1))

            ll = np.multiply(distF.reshape(self.L, 1), self.msgs[i].reshape(self.L, 1))

            assert(len(ll) == self.L)

            # normalize values => probabilities
            ll = np.nan_to_num(ll)
            ll += 1E-20
            ll = ll / float(np.sum(ll))
            assert(np.abs(np.sum(ll) - 1.0) < 1E-3)
        return ll  # self.msgs[:, -1]

    def __prepare_data(self):
        data_list = []
        if self.data[0][0] < max(self.key_reg.values()):
            list_temp = [self.data[0][0]]
        else:
            list_temp = [self.data[0][1]]
        for obs in self.data:
            if obs[1] == max(self.key_reg.values()):
                data_list.append(cp.deepcopy(list_temp))
                list_temp = []
            else:
                list_temp.append(obs[1])
        if len(list_temp) > 0:
            data_list.append(cp.deepcopy(list_temp))
        self.data = cp.deepcopy(data_list)
