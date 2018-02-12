# Global Pattern Analysis Documentation

_Given behavior of individual users, extract behavior patterns and represent user by the pattern they exhibit_

__Input:__

* X: Behavior of users, e.g. \[user1, user2, user3, ...\]
    * Behavior of a user = list of his sessions, e.g. \[\[\<s1>], \[\<s2>, ...]
    * Session of a user = list of events, e.g. \[a, a, b, a, c, a]

_Example:_ X = \[ \[\[a, a], \[b, a]], \[\[c, c], \[a, b]] ]

Here, X contains the behavior of two users.
<br>For user1 two sessions of behavior were recorded (\[a, a] and \[b, a]).
<br>User2 is represented by his behavior (\[c, c] and \[a,b]).
<br>So, a user is represented by a list (sessions) of lists (events of a session)
<br>And X is comprised of all users, therefore, of a list (users) of lists (sessions of a user) of lists (events of a session of a user)

* Xd (_optional_): Corresponding durations between events (same encoding as X, s.a.)
* Xgt (_optional_): Ground-truth of desired segmentation. If provided yields better verbose info.
    Xgt = a list of labels

### Functions
_Import libraries_: ``from gpa.gpa import GlobalPatternAnalysis``

```python
gpa = GlobalPatternAnalysis(X, Xd=None, Xgt=None, L=100, num_it=100, gamma=1.0, alpha=1.0, sigma=1.0,
    lam=1.0, d_gamma=1.0, d_lambda=1.0, d_base_prob=1E-5, rho=2.0, boundary=-1, num_it_gmm=1E7,
    sub_sample_gmm=2E6, k_gmm=50, raster_thresh=1E-2, d_importance_trash=200, model_save_dir='data/',
    model_save_path='model', eval_intervals=[(40, 100)], verbose=False)
```

Initialization of GPA. Here X can be input as described above or as apath to such a list stored as a pickle object.
Given the data, GPA calls the segmentation algorithm to fit the model to the data.

__model_save_dir__: path to the directory the model should be saved to
<br>__model_save_path__: filename in which to store the model

<br>

```python
Xp = gpa.transform(Xe, Xed=[], modus=0, singleton=False)
```

Based on the learned segmentation model, the algorithm transforms original data _Xe_ into a vector-space representation _Xp_.

__Modus:__ 0 = 'combined', 1 = 'time-based', 2 = 'count-based'<br>
While count-based represents the user by the frequency of how often he showed different behavior patterns,
time-based represent him by the time spent within different patterns. Combined represent a user making use of both information.

__singleton:__ default = False, True means that the data provided represents sessions of a single user.
This information is used to automatically convert data into its necessary form, if the data shows minor encoding issues.

<br>

```python
gpa.fit(fct, Xp, params, y=None)
```

Providing a sklearn-function (e.g. regression or classification), data _Xp_ for training, labels _y_ if needed and parameter-grid for model-selection, the function returns the model with the optimal parameters.
If _y_ is None the algorithm assumes that labels are not needed.

<br>

```python
y_pred = gpa.predict(fct, Xp, y, params)
```

pass

<br>

```python
y_proba = gpa.predict_proba(fct, Xp, y, params)
```

pass

<br><br>___Example___

```python
from gpa.gpa import GlobalPatternAnalysis
from tools.toy_data import ToyData
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np


# Note: train_test_split is missing in this mini-example --> results are not representative!!!

# DATA
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
```
