import pickle
from gpa.gpa import GlobalPatternAnalysis


id_train, id_test = pickle.load(open('data/dataset_split.pckl', 'rb'))
X, Xd, Xl, scount = pickle.load(open('data/time_series_twitter.pickle', 'rb'))

gpa = GlobalPatternAnalysis('data/final_model_all_250-80.pckl')

labels = {}
for entry in Xl:
    if entry[0] not in labels:
        labels[entry[0]] = entry[2] 

real_id_train = [list(X.keys())[itm] for itm in id_train]
train_labels = [labels[itm] for itm in real_id_train]

X_list = list(X.values())
Xd_list = list(Xd.values())

X_train = [X_list[id] for id in id_train]
Xd_train = [Xd_list[id] for id in id_train] 

X_projected = gpa.project(X_train, Xd_train)

pickle.dump(X_projected, open('users_latent_representation.pckl', 'wb'))

