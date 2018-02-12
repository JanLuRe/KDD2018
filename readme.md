from gpa.gpa import GlobalPatternAnalysis
from tools.toy_data import ToyData
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np


# DATA
X_train, Xd_train, y_train = ToyData.sample(100)
X_test, Xd_test, y_test = ToyData.sample(100)

# for grid-search
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
par = dict(gamma=gamma_range, C=C_range, class_weight=['balanced'])

# initialize GPA and fit/load segmentation model to data
# X_flat: X without user info --> list (sessions) of lists (events of session)
X_flat = [session for user in X_train for session in user]
Xd_flat = [session for user in Xd_train for session in user] if Xd_train is not None else None
gpa = GlobalPatternAnalysis(X_flat, Xd_flat, L=10, lam=5., num_it=50, model_save_path='model_example')

# transform original data into vector-space representation
Xp_train = gpa.transform(X_train, Xd_train)
Xp_test = gpa.transform(X_test, Xd_test)

# fit classification model to data
gpa.fit(SVC(), Xp_train, y_train, params=par)

# predict labels of observed behavior
y_pred = gpa.predict(Xp_test)
print(classification_report(y_test, y_pred, digits=4))
