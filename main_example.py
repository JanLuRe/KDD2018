from gpa.gpa import GlobalPatternAnalysis
from tools.toy_data import ToyData
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Note: train_test_split is missing in this mini-example --> results are not representative!!!

# DATA
X, Xd, y = ToyData.sample(100, durations=False)
#X, Xd, y = ToyData.sample(1000, durations=True, udist=[.25, .25, .25, .25], dmean=[10, 20, 10, 40], dvar=[1.0, 1.0, 1.0, 1.0])

# for grid-search
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
par = dict(gamma=gamma_range, C=C_range, class_weight=['balanced'])

# initialize GPA and fit/load segmentation model to data
# X_flat: X without user info --> list (sessions) of lists (events of session)
X_flat = [session for user in X for session in user]
Xd_flat = [session for user in Xd for session in user] if Xd is not None else None
print('Fitting Segmentation Model...')
gpa = GlobalPatternAnalysis(X_flat, Xd_flat, L=20, num_it=200, eval_intervals=[(100, 150), (150, 200)], model_save_path='model_example_3c_simple',
                            verbose=False)

# transform original data into vector-space representation
print('Transforming Data...')
Xp = gpa.transform(X, Xd)
print('Fitting Classification Model...')
# fit classification model to data
gpa.fit(SVC(), Xp, y, params=par)
# predict labels of observed behavior
y_pred = gpa.predict(Xp)
print(classification_report(y, y_pred, digits=4))
