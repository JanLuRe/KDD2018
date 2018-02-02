import pickle
import numpy as np
from gpa.gpa import GlobalPatternAnalysis
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.svm import SVC, OneClassSVM as oSVC
from sklearn.ensemble import AdaBoostClassifier as ada, RandomForestClassifier as rf
from sklearn.mixture import BayesianGaussianMixture as bgm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression as lr
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")


def feature_extraction_vis(data, cats, set_clicks, max_len):
    features_entities = []
    for entity in data:
        # temporal = np.bincount([ele for session in entity for ele in session], minlength=np.max(set_clicks) + 1)
        temporal = np.bincount([ele for session in entity for ele in session], minlength=np.max(set_clicks) + 1)

        spatial = np.bincount([i for session in entity for ele in session for i in range(len(cats)) if ele in cats[i]], minlength=len(cats))

        spatial_temporal = np.zeros(max_len)
        session_counts = [np.bincount([i for ele in session for i in range(len(cats)) if ele in cats[i]], minlength=len(cats)) for session in entity]
        st_feature = []
        for session in session_counts:
            if np.sum(session) == 0:
                print(session)
            p_session = session / np.sum(session)
            st_feature.append(-np.nansum(p_session * np.log2(p_session)))
        spatial_temporal[:len(st_feature)] = st_feature

        features = np.hstack((np.hstack((temporal, spatial)), spatial_temporal))
        features_entities.append(features)
    return np.array(features_entities)


def feature_extraction_wang(data, duration, cats):
    features_entities = []
    for entity, edurs in zip(data, duration):
        avg_clicks_session = np.average([len(session) for session in entity])
        avg_session_len = np.average([np.sum(session) for session in edurs])
        avg_time_between_clicks = np.average([ele for session in edurs for ele in session])
        avg_sessions_per_day = len(entity)
        hist_cats = np.bincount([i for session in entity for ele in session for i in range(len(cats)) if ele in cats[i]], minlength=len(cats))
        feature = [avg_clicks_session, avg_session_len, avg_time_between_clicks, avg_sessions_per_day] + list(hist_cats)
        features_entities.append(feature)
    return features_entities

def normalize(data):
    multi = len(data) > 1
    normalizer = Normalizer()
    input_data = np.vstack(data) if multi else data[0]
    normalizer.fit(input_data)
    normalized_data = [normalizer.transform(element) for element in data] if multi else normalizer.transform(data[0])
    return normalized_data

def split_data(data, class_gt):
    cl0 = [itm for itm, cl in zip(data, class_gt) if cl == 0]
    cl1 = [itm for itm, cl in zip(data, class_gt) if cl == 1]
    cl2 = [itm for itm, cl in zip(data, class_gt) if cl == 2]
    return cl0, cl1, cl2

def compute_dists(data, per=15):
    pca = PCA(n_components=10)
    pca.fit(data[0])
    dists = []
    for element in data:
        projected = pca.transform(element)
        invproj = pca.inverse_transform(projected)
        dists.append([ed(proj.reshape(1, -1), iproj.reshape(1, -1))[0][0] for proj, iproj in zip(element, invproj)])
    dists.append(np.percentile(dists.pop(0), per))
    return dists

def eval_clf(fct, params, features_data, class_gt):
    X_train, X_eval, y_train, y_eval = train_test_split(features_data, class_gt, test_size=0.2, random_state=42)
    clf = GridSearchCV(fct, params, cv=5, scoring='f1_weighted')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)
    return y_eval, y_pred

def eval_pca(dists, threshold, algo_name='Not specified'):
    y_pred = list(dists[0] > threshold)
    y_eval = np.zeros(len(dists.pop(0))).tolist()
    for element in dists:
        y_pred.append(element > threshold)
        y_eval = y_eval + np.ones(len(element)).tolist()
    y_pred = np.hstack([itm.reshape(1, -1) for itm in y_pred]).astype(np.int)
    return confusion_matrix(y_eval, list(y_pred[0]))
    # print(algo_name)
    # print(classification_report(y_eval, list(y_pred[0])))

def eval_svm(features_data, class_gt):
    params = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 'auto'], 'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 100], 'class_weight': ['balanced']},
              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 100], 'class_weight': ['balanced']}]
    return eval_clf(SVC(), params, features_data, class_gt)

def eval_logreg(features_data, class_gt):
    params = {"penalty": ['l1', 'l2'], "C": np.arange(0.01, 0.5, 0.01), "class_weight": [None, 'balanced'], "max_iter": [5000]}
    return eval_clf(lr(), params, features_data, class_gt)

def eval_rf(features_data, class_gt):
    params = {"max_depth": [None],
                 "max_features": ['auto', 'log2', None],  # , 1, 2, 3, 4, 5, 6],
                 "min_samples_split": np.arange(1, 10),
                 "min_samples_leaf": np.arange(1, 10),
                 "min_impurity_split": [1E-1, 1E-2, 1E-3, 1E-4, 1E-5],
                 "n_estimators": np.arange(2, 10),
                 "bootstrap": [True],
                 "criterion": ["gini", "entropy"],
                 "class_weight": ["balanced", None]}
    return eval_clf(rf(), params, features_data, class_gt)



VIS = True              #  0 PCA    VIS
VIS_SVM = True          #  1 SVM    VIS
VIS_ADA = True          #  2 ADA    VIS
VIS_OSVM = True
VIS_LR = True
WANG1 = True            #  3 SVM    WANG
WANG2 = True            #  4 ADA    WANG
WANG_PCA = True         #  5 PCA    WANG
WANG_OSVM = True
WANG_LR = True
OURS = True
OURS1 = True            #  6 SVM    OURS
OURS2 = True            #  7 ADA    OURS
OURS3 = True            #  8 PCA    OURS
OURS4 = True            #  9 SIMPLE OURS
OURS5 = True            # 10 OSVM   OURS
OURS6 = True            # 11 GMM    OURS
OURS_LR = True

# pool = [
#     ['data/synth_behavior_data_2c_0_10k5m.pckl', 'data/eval2040/final_model_2c_020-40.pckl', False],
#     ['data/synth_behavior_data_2c_25_10k5m.pckl', 'data/eval2040/final_model_2c_2520-40.pckl', False],
#     ['data/synth_behavior_data_2c_50_10k5m.pckl', 'data/eval2040/final_model_2c_5020-40.pckl', False],
#     ['data/synth_behavior_data_2c_75_10k5m.pckl', 'data/eval2040/final_model_2c_7520-40.pckl', False],
#     ['data/synth_behavior_data_2c_100_10k5m.pckl', 'data/eval2040/final_model_2c_10020-40.pckl', False],
#     ['data/synth_behavior_data_3c_0_10k5m.pckl', 'data/eval2040/final_model_3c_020-40.pckl', True],
#     ['data/synth_behavior_data_3c_100_10k5m.pckl', 'data/eval2040/final_model_3c_10020-40.pckl', True]
# ]
pool = [
    ['/media/robzer/48C8573DC8572888/.work/www18/run_paper/synth_behavior_data_2c_0_1k500k.pckl',
     '/media/robzer/48C8573DC8572888/.work/www18/run_1/final_model_2c_020-40.pckl', False],
    ['/media/robzer/48C8573DC8572888/.work/www18/run_paper/synth_behavior_data_2c_50_1k500k.pckl',
     '/media/robzer/48C8573DC8572888/.work/www18/run_1/final_model_2c_5020-40.pckl', False],
    ['/media/robzer/48C8573DC8572888/.work/www18/run_paper/synth_behavior_data_2c_100_1k500k.pckl',
     '/media/robzer/48C8573DC8572888/.work/www18/run_1/final_model_2c_10020-40.pckl', False],
    ['/media/robzer/48C8573DC8572888/.work/www18/run_paper/synth_behavior_data_3c_0_1k500k.pckl',
     '/media/robzer/48C8573DC8572888/.work/www18/run_1/final_model_3c_020-40.pckl', True],
    ['/media/robzer/48C8573DC8572888/.work/www18/run_paper/synth_behavior_data_3c_50_1k500k.pckl',
     '/media/robzer/48C8573DC8572888/.work/www18/run_1/final_model_3c_5020-40.pckl', True],
    ['/media/robzer/48C8573DC8572888/.work/www18/run_paper/synth_behavior_data_3c_100_1k500k.pckl',
     '/media/robzer/48C8573DC8572888/.work/www18/run_1/final_model_3c_10020-40.pckl', True]
]

results = [[[] for j in range(0, 17)] for i in range(0, len(pool))]
for dataset, eval_res in zip(pool, results):
    print(dataset[0])
    data, duration, gt, class_gt = pickle.load(open(dataset[0], 'rb'))
    max_len = np.max([len(session) for entity in data for session in entity])
    flat_data = [ele for ent in data for session in ent for ele in session]
    set_data = list(set(flat_data))
    num_cats = 8

    state_dist = np.random.dirichlet(np.ones(num_cats) * 10)
    cats = [[] for i in range(num_cats)]
    for state in set_data:
        cats[np.random.choice(range(num_cats), p=state_dist)].append(state)

    set_clicks = np.unique([ele for entity in data for session in entity for ele in session])

    if VIS:
        data_human, data_bot, data_cyborg = split_data(data, class_gt)
        f_hum = feature_extraction_vis(data_human, cats, set_clicks, max_len)
        f_bot = feature_extraction_vis(data_bot, cats, set_clicks, max_len)
        if dataset[2]:
            f_cbg = feature_extraction_vis(data_cyborg, cats, set_clicks, max_len)
            f_hum, f_bot, f_cbg = normalize([f_hum, f_bot, f_cbg])
        else:
            f_hum, f_bot = normalize([f_hum, f_bot])
        f_hum_train, f_hum_test = train_test_split(f_hum, test_size=.2, random_state=42)

        if dataset[2]:
            dists_hum, dists_bot, dists_cbg, threshold = compute_dists([f_hum_train, f_hum_test, f_bot, f_cbg])
            eval_res[0].append(eval_pca([dists_hum, dists_bot, dists_cbg], threshold, 'VISWANATH et al'))
        else:
            dists_hum, dists_bot, threshold = compute_dists([f_hum_train, f_hum_test, f_bot])
            eval_res[0].append(eval_pca([dists_hum, dists_bot], threshold, 'VISWANATH et al'))

    if VIS_SVM:
        #features VIS
        features_data = feature_extraction_vis(data, cats, set_clicks, max_len)
        #normalize
        features_data = normalize([features_data])

        # idx = np.random.choice(features_data.shape[0], 1000, replace=False)
        # y_eval, y_pred = eval_svm(features_data[idx, :], [class_gt[i] for i in idx])
        y_eval, y_pred = eval_svm(features_data, class_gt)

        eval_res[1].append(confusion_matrix(y_eval, y_pred))
        # print('VIS SVM')
        # print(classification_report(y_eval, y_pred, digits=5))

    if VIS_ADA:
        #features VIS
        features_data = feature_extraction_vis(data, cats, set_clicks, max_len)
        #normalize
        features_data = normalize([features_data])
        # split data
        X_train, X_eval, y_train, y_eval = train_test_split(features_data, class_gt, test_size=0.2, random_state=42)
        # learn classifier
        clf = ada(n_estimators=200)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_eval)
        eval_res[2].append(confusion_matrix(y_eval, y_pred))
        # print('VIS ADABOOST')
        # print(classification_report(y_eval, y_pred, digits=5))

    if VIS_OSVM:
        data_human, data_bot, data_cyborg = split_data(data, class_gt)
        f_hum = feature_extraction_vis(data_human, cats, set_clicks, max_len)
        f_bot = feature_extraction_vis(data_bot, cats, set_clicks, max_len)
        if dataset[2]:
            f_cbg = feature_extraction_vis(data_cyborg, cats, set_clicks, max_len)
            f_hum, f_bot, f_cbg = normalize([f_hum, f_bot, f_cbg])
        else:
            f_hum, f_bot = normalize([f_hum, f_bot])
        f_hum_train, f_hum_test = train_test_split(f_hum, test_size=.2, random_state=42)
        clf = oSVC(nu=0.1, kernel='rbf', gamma=0.05)
        clf.fit(f_hum_train)
        y_pred = list(clf.predict(f_hum_test))
        y_pred = y_pred + list(clf.predict(f_bot))
        if dataset[2]:
            y_pred = y_pred + list(clf.predict(f_cbg))
            y_ano = np.ones(len(f_bot) + len(f_cbg)) * -1
        else:
            y_ano = np.ones(len(f_bot)) * -1
        y_eval = np.ones(len(f_hum_test)).tolist() + y_ano.tolist()
        eval_res[3].append(confusion_matrix(y_eval, y_pred))

    if VIS_LR:
        features_data = feature_extraction_vis(data, cats, set_clicks, max_len)
        features_data = normalize([features_data])
        y_eval, y_pred = eval_logreg(features_data, class_gt)

        eval_res[4].append(confusion_matrix(y_eval, y_pred))

    if WANG1:
        # WANG et al
        features_data = feature_extraction_wang(data, duration, cats)
        #normalize
        features_data = normalize([features_data])
        y_eval, y_pred = eval_svm(features_data, class_gt)
        eval_res[5].append(confusion_matrix(y_eval, y_pred))
        # print('WANG et al')
        # print(classification_report(y_eval, y_pred, digits=5))

        if WANG2:
            clf = ada(n_estimators=200)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_eval)
            eval_res[6].append(confusion_matrix(y_eval, y_pred))
            # print('WANG et al (alternative)')
            # print(classification_report(y_eval, y_pred, digits=5))

    if WANG_PCA:
        f_hum_train, f_hum_test = train_test_split(f_hum, test_size=.2, random_state=42)

        data_human, data_bot, data_cyborg = split_data(data, class_gt)
        f_hum = feature_extraction_wang(data_human, duration, cats)
        f_bot = feature_extraction_wang(data_bot, duration, cats)
        if dataset[2]:
            f_cbg = feature_extraction_wang(data_cyborg, duration, cats)
            f_hum, f_bot, f_cbg = normalize([f_hum, f_bot, f_cbg])
        else:
            f_hum, f_bot = normalize([f_hum, f_bot])

        f_hum_train, f_hum_test = train_test_split(f_hum, test_size=.2, random_state=42)

        if dataset[2]:
            dists_hum, dists_bot, dists_cbg, threshold = compute_dists([f_hum_train, f_hum_test, f_bot, f_cbg])
            eval_res[7].append(eval_pca([dists_hum, dists_bot, dists_cbg], threshold, 'WANG PCA'))
        else:
            dists_hum, dists_bot, threshold = compute_dists([f_hum_train, f_hum_test, f_bot])
            eval_res[7].append(eval_pca([dists_hum, dists_bot], threshold, 'WANG PCA'))

    if WANG_OSVM:
        f_hum_train, f_hum_test = train_test_split(f_hum, test_size=.2, random_state=42)

        data_human, data_bot, data_cyborg = split_data(data, class_gt)
        f_hum = feature_extraction_wang(data_human, duration, cats)
        f_bot = feature_extraction_wang(data_bot, duration, cats)
        if dataset[2]:
            f_cbg = feature_extraction_wang(data_cyborg, duration, cats)
            f_hum, f_bot, f_cbg = normalize([f_hum, f_bot, f_cbg])
        else:
            f_hum, f_bot = normalize([f_hum, f_bot])

        f_hum_train, f_hum_test = train_test_split(f_hum, test_size=.2, random_state=42)
        clf = oSVC(nu=0.1, kernel='rbf', gamma=0.05)
        clf.fit(f_hum_train)
        y_pred = list(clf.predict(f_hum_test))
        y_pred = y_pred + list(clf.predict(f_bot))
        if dataset[2]:
            y_pred = y_pred + list(clf.predict(f_cbg))
            y_ano = np.ones(len(f_bot) + len(f_cbg)) * -1
        else:
            y_ano = np.ones(len(f_bot)) * -1
        y_eval = np.ones(len(f_hum_test)).tolist() + y_ano.tolist()
        eval_res[8].append(confusion_matrix(y_eval, y_pred))

    if WANG_LR:
        features_data = feature_extraction_wang(data, duration, cats)
        features_data = normalize([features_data])
        y_eval, y_pred = eval_logreg(features_data, class_gt)

        eval_res[9].append(confusion_matrix(y_eval, y_pred))


    if OURS:
        lbs = GlobalPatternAnalysis(dataset[1])
        duration = [[[float(ele) for ele in session] for session in entity] for entity in duration]
        data_vs = lbs.project(data, duration, modus=2)
        #normalize
        data_vs_norm = normalize([data_vs])

        X_train, X_eval, y_train, y_eval = train_test_split(data_vs_norm, class_gt, test_size=0.2, random_state=42)

        if OURS1:
            # learn classifier
            params = {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 'auto'], 'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 100], 'class_weight': ['balanced']}
            lbs.fit(SVC(), X_train, y_train, params)
            # predict
            y_pred = lbs.predict(X_eval)
            eval_res[10].append(confusion_matrix(y_eval, y_pred))
            # print('OURS (SVM)')
            # print(classification_report(y_eval, y_pred, digits=5))

        if OURS2:
            # learn classifier
            params = {'n_estimators': [200]}
            lbs.fit(ada(), X_train, y_train, params)
            # predict
            y_pred = lbs.predict(X_eval)
            eval_res[11].append(confusion_matrix(y_eval, y_pred))
            # print('OURS (ADAboost)')
            # print(classification_report(y_eval, y_pred, digits=5))

        if OURS3:
            # Ours: unsupervised (PCA)
            pd_hum, pd_bot, pd_cbg = split_data(data_vs_norm, class_gt)

            pd_hum_train, pd_hum_test = train_test_split(pd_hum, test_size=.2, random_state=42)

            if dataset[2]:
                dists_hum, dists_bot, dists_cbg, threshold = compute_dists([pd_hum_train, pd_hum_test, pd_bot, pd_cbg])
                eval_res[12].append(eval_pca([dists_hum, dists_bot, dists_cbg], threshold, 'OURS PCA'))
            else:
                dists_hum, dists_bot, threshold = compute_dists([pd_hum_train, pd_hum_test, pd_bot])
                eval_res[12].append(eval_pca([dists_hum, dists_bot], threshold, 'OURS PCA'))

        if OURS4:
            mean_hum = np.mean(pd_hum_train, axis=0)
            threshold = np.percentile([ed(itm.reshape(1, -1), mean_hum.reshape(1, -1))[0][0] for itm in pd_hum_train], 90)
            y_pred = ([ed(itm.reshape(1, -1), mean_hum.reshape(1, -1))[0][0] for itm in pd_hum_test] > threshold).tolist()
            y_pred = y_pred + ([ed(itm.reshape(1, -1), mean_hum.reshape(1, -1))[0][0] for itm in pd_bot] > threshold).tolist()
            y_pred = y_pred + ([ed(itm.reshape(1, -1), mean_hum.reshape(1, -1))[0][0] for itm in pd_cbg] > threshold).tolist()
            y_eval2 = np.zeros(len(pd_hum_test)).tolist()
            y_eval2 = y_eval2 + np.ones(len(pd_bot) + len(pd_cbg)).tolist()
            eval_res[13].append(confusion_matrix(y_eval2, y_pred))
            # print('OURS (simple)')
            # print(classification_report(y_eval, y_pred, digits=5))

        if OURS5:
            clf = oSVC(nu=0.1, kernel='rbf', gamma=0.05)
            clf.fit(pd_hum_train)
            y_pred = list(clf.predict(pd_hum_test))
            y_pred = y_pred + list(clf.predict(pd_bot))
            if dataset[2]:
                y_pred = y_pred + list(clf.predict(pd_cbg))
                y_ano = np.ones(len(pd_bot) + len(pd_cbg)) * -1
            else:
                y_ano = np.ones(len(pd_bot)) * -1
            y_eval2 = np.ones(len(pd_hum_test)).tolist() + y_ano.tolist()
            eval_res[14].append(confusion_matrix(y_eval2, y_pred))
            # print('OURS (OneclassSVM)')
            # print(classification_report(y_eval, y_pred, digits=5))

        if OURS6:
            estimator = bgm(50, max_iter=int(1E5))  # , covariance_type='diag')
            # print(pd_hum_train)
            model = estimator.fit(pd_hum_train)
            p_hum = estimator.predict_proba(pd_hum_test)
            p_bot = estimator.predict_proba(pd_bot)
            y_pred = [any(mix > .9 for mix in element) for element in p_hum]
            y_pred = y_pred + [any(mix > .9 for mix in element) for element in p_bot]
            y_eval2 = np.ones(len(pd_hum_test)).tolist() + np.zeros(len(pd_bot)).tolist()
            if dataset[2]:
                p_cbg = estimator.predict_proba(pd_cbg)
                y_pred = y_pred + [any(mix > .9 for mix in element) for element in p_cbg]
                y_eval2 = y_eval2 + np.zeros(len(pd_cbg)).tolist()
            eval_res[15].append(confusion_matrix(y_eval2, y_pred))
            # print('OURS (GMM)')
            # print(classification_report(y_eval, y_pred, digits=5))

        if OURS_LR:
            params = {"penalty": ['l1', 'l2'], "C": np.arange(0.01, 0.5, 0.01), "class_weight": [None, 'balanced'], "max_iter": [5000]}
            lbs.fit(lr(), X_train, y_train, params)
            y_pred = lbs.predict(X_eval)
            eval_res[16].append(confusion_matrix(y_eval, y_pred))

save_path = '/media/robzer/48C8573DC8572888/.work/www18/run_1/evaluation_results2040_all.pckl'
with open(save_path, 'wb') as f:
    pickle.dump(results, f)
