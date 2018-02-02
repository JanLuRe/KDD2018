from segmentation import config
from tools import percentage_bar as pb
from segmentation.eIMMC import eIMMC
from sklearn.model_selection import GridSearchCV
import copy as cp
import numpy as np
import pickle


class GlobalPatternAnalysis:
    # init LBS by inferring immc model from data
    # 1. FROM DATA: input X:data, Xd:state durations, Xgt:ground truth
    # 2. FROM FILE: input X:path
    def __init__(self, X, Xd=None, Xgt=None, L=100, num_it=100, gamma=1.0, alpha=1.0, sigma=1.0, lam=1.0, d_gamma=1.0, d_lambda=1.0,
                 d_base_prob=1E-6, rho=2.0, boundary=-1, num_it_gmm=1E7, sub_sample_gmm=5E5, k_gmm=50, raster_thresh=1E-2, d_importance_trash=200,
                 model_save_dir='data/', model_save_path='model', eval_intervals=[(40, 100)], verbose=False):
        self.MIX_BASED = 0
        self.TIME_BASED = 1
        self.COUNT_BASED = 2
        self.clf = None

        self.immc = eIMMC(debug=verbose)
        self.immc.cfg.set_attr(L, num_it, gamma, alpha, sigma, lam, d_gamma, d_lambda, d_base_prob, rho, boundary, num_it_gmm, sub_sample_gmm, k_gmm,
                               raster_thresh, d_importance_trash, model_save_dir, model_save_path, eval_intervals)

        if isinstance(X, str):
            self.mc = pickle.load(open(X, 'rb'))
            self.immc.load_model(self.mc)
        else:
            if Xgt is not None:
                Xgt = [itm for seq in Xgt for itm in seq]

            # preprocess X: time-series values 0, 1, ...
            X_set = set([ele for session in X for ele in session])
            key_reg = {}
            for idx, ele in enumerate(X_set):
                key_reg[ele] = idx
            X = [list(map(key_reg.get, session)) for session in X]

            # fit model to data
            if Xd is None:
                self.immc.cfg.collapsed = True
            self.immc.fit(X, durations=Xd, gt_display=Xgt, key_reg=key_reg)
            self.mc = pickle.load(open(model_save_dir + model_save_path + str(eval_intervals[-1][0]) + '-' + str(eval_intervals[-1][1]) +
                                       self.immc.cfg.extension,
                                       'rb'))
            self.immc.load_model(self.mc)

    # project data into vector space spanned by super states
    # modus 0: mixed, 1: time-based, 2: count-based
    def transform(self, Xe, Xed=None, modus=0, singleton=False):
        if Xed is None: modus = self.COUNT_BASED
        X_project = []
        if not all(isinstance(el, list) for el in Xe):
            Xe = [Xe]
            Xed = [Xed]
        if not all(isinstance(el, list) for seq in Xe for el in seq):
            if singleton:
                Xe = [Xe]
                Xed = [Xed]
            else:
                Xe = [[user] for user in Xe]
                Xed = [[user_durs] for user_durs in Xed]

        count = 1
        max_count = len(Xe)
        iterator = zip(Xe, Xed) if Xed is not None else [[x, []] for x in Xe]
        for X_entity, Xd_entity in iterator:
            X_transformed = self.immc.transform(X_entity, Xd_entity)
            X_flat = [ele for seq in X_transformed for ele in seq]
            if modus < self.COUNT_BASED:
                x_ent = np.zeros(self.mc.cfg.L)
                Xd_flat = [ele for seq in Xd_entity for ele in seq]
                x_ent[:np.max(X_flat) + 1] = np.bincount(X_flat, Xd_flat)
                if modus == 1: X_project.append(x_ent)
            if modus % 2 == 0:
                xd_ent = np.zeros(self.mc.cfg.L)
                idx, bincounts = np.unique(X_flat, return_counts=True)
                xd_ent[idx] += bincounts
                if modus == 2: X_project.append(xd_ent)
            if modus == 0: X_project.append(x_ent + xd_ent)
            pb.printProgressBar(count, max_count, prefix='Projecting: Progress', suffix='Complete', length=50)
            count += 1
        return np.array(X_project)

    def fit(self, fct, Xp, y, params):
        self.clf = GridSearchCV(fct, params, cv=5, scoring='f1_weighted')
        self.clf.fit(Xp, y)

    def predict(self, Xp):
        return self.clf.predict(Xp)
