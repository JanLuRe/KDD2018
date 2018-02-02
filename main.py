# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:36:04 2015

@author: jr
"""

from gpa.gpa import GlobalPatternAnalysis
import pickle
import warnings
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")


def load_data(path=None):
    if path is None:
        data, duration, data_labels, state_counts = pickle.load(open('data/time_series_twitter.pickle', 'rb'))
        # X_labels = pickle.load(open('data/extracted_labels.pckl', 'rb'))
        # X_labels = pd.DataFrame(labels, columns=['user_id', 'dset', 'label'])['label'].as_matrix()
        X = [seq for key, seq in data.items()]
        Xd = [seq for key, seq in duration.items()]

        labels = {}
        for entry in data_labels:
            if entry[0] not in labels:
                labels[entry[0]] = entry[2]

        stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for idx_train, idx_test in stratSplit.split(X, list(labels.values())):
            X_train = [x for idx, x in enumerate(X) if idx in idx_train]
            X_test = [x for idx, x in enumerate(X) if idx in idx_test]
            Xd_train = [xd for idx, xd in enumerate(Xd) if idx in idx_train]
            Xd_test = [xd for idx, xd in enumerate(Xd) if idx in idx_test]
        pickle.dump([idx_train, idx_test, X_train, X_test, Xd_train, Xd_test, labels], open('data/dataset_split.pckl', 'wb'))
    else:
        idx_train, idx_test, X_train, X_test, Xd_train, Xd_test, labels = pickle.load(open('data/dataset_split.pckl', 'rb'))

    real_id_train = [list(data.keys())[itm] for itm in idx_train]
    real_id_test = [list(data.keys())[itm] for itm in idx_test]
    train_labels = [labels[itm] for itm in real_id_train]
    test_labels = [labels[itm] for itm in real_id_test]

    return X_train, X_test, Xd_train, Xd_test, train_labels, test_labels

# load data
X_train, X_test, Xd_train, Xd_test, train_labels, test_labels = load_data()

# for grid-search
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
par = dict(gamma=gamma_range, C=C_range, class_weight=['balanced'])

# learn vector space
print('Initializing...')
gpa = GlobalPatternAnalysis(X_train, Xd_train, L=100, num_it=1000, eval_intervals=[(50, 100), (100, 200), (200, 300), (250, 500), 
                                                                                   (500, 1000)], model_save_path='model_twitter')

# transform original data into vector-space representation
Xp_train = gpa.transform(X_train, Xd_train)
print('Fitting Classification Model...')
# fit classification model to data
gpa.fit(SVC(), Xp_train, train_labels, params=par)
# predict labels of observed behavior
Xp_test = gpa.transform(X_test, Xd_test)
y_pred = gpa.predict(Xp_test)
print(classification_report(test_labels, y_pred, digits=4))
