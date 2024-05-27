import numpy as np
from collections import Counter
import data
import utils
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

auto_data, auto_cat = data.automobileRead()

X, y = utils.first_proccess(auto_data, auto_cat)

n_train_X = np.empty((0, 25))
n_train_y = np.empty((0, 1))

for train_ix, test_ix in kfold.split(X, y):
    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
    train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
    test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
    print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
    train_X, train_y, max_class, most_common = utils.sort_by_class_quantity(train_X, train_y)
    train_X, train_y = utils.kNN(train_X, train_y, auto_cat)
    unique_classes = np.unique(train_y[:, 0])
    index = np.argwhere(unique_classes == most_common)
    unique_classes = np.delete(unique_classes, index)
    train_X, train_y = utils.preprocess(train_X, train_y, max_class, unique_classes)
    n_train_X = np.concatenate((n_train_X, train_X), axis=0)
    n_train_y = np.concatenate((n_train_y, train_y), axis=0)

svm_model = svm.SVC()
svm_model.fit(n_train_X, n_train_y.ravel())
y_pred = svm_model.predict(test_X)
print("SVM MODEL:")
print("Accuracy:", metrics.accuracy_score(test_y, y_pred))
print("Precision:", metrics.precision_score(test_y, y_pred, average='macro'))
print("Recall:", metrics.recall_score(test_y, y_pred, average='macro'))

dt_model = DecisionTreeClassifier()
dt_model.fit(n_train_X, n_train_y.ravel())
y_pred = dt_model.predict(test_X)
print("\nDecision Tree MODEL:")
print("Accuracy:", metrics.accuracy_score(test_y, y_pred))

lr_model = LogisticRegression(solver='saga', max_iter=8000)
lr_model.fit(n_train_X, n_train_y.ravel())
y_pred = lr_model.predict(test_X)
print("\nLogistic Regression MODEL:")
print("Accuracy:", metrics.accuracy_score(test_y, y_pred))