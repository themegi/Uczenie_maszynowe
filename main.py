import numpy as np
import data
import utils
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#odchylenie standardowe

kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1)
#średnia albo wartość t-statystyki
auto_data, auto_cat = data.yeastRead()

X, y = utils.first_proccess(auto_data, auto_cat)
accuracy_list_svm = []
precision_list_svm  = []
recall_list_svm  = []
f1_list_svm  = []

accuracy_list_dt = []
precision_list_dt  = []
recall_list_dt  = []
f1_list_dt  = []

accuracy_list_lr = []
precision_list_lr = []
recall_list_lr  = []
f1_list_lr = []

for train_ix, test_ix in kfold.split(X, y):
    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
    train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
    test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
    print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
    train_X, train_y, max_class, most_common = utils.sort_by_class_quantity(train_X, train_y)
    train_X, train_y = utils.kNN(train_X, train_y, auto_cat)
    unique_classes = np.unique(train_y[:, 0])
    all_classes = unique_classes
    index = np.argwhere(unique_classes == most_common)
    unique_classes = np.delete(unique_classes, index)
    train_X, train_y = utils.preprocess(train_X, train_y, max_class, unique_classes)

#### SVM ####
    svm_model = svm.SVC()
    svm_model.fit(train_X, train_y.ravel())
    y_pred = svm_model.predict(test_X)
    print("SVM MODEL:")
    print(classification_report(test_y, y_pred, zero_division=0))
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=1)
    recall = recall_score(test_y, y_pred, average='macro', zero_division=1)
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=1)
    accuracy_list_svm.append(accuracy)
    precision_list_svm.append(precision)
    recall_list_svm.append(recall)
    f1_list_svm.append(f1)

    print("Specificity: ",utils.calculate_specificity(test_y, y_pred))
    print("Sensitivity: ",utils.calculate_sensitivity(test_y, y_pred))

    # res = []
    # for l in all_classes:
    #     prec, recall, _, _ = precision_recall_fscore_support(np.array(test_y) == l,
    #                                                          np.array(y_pred) == l,
    #                                                          pos_label=True, average=None, zero_division=0)
    #     res.append([l, recall[0], recall[1]])
    #
    # print(pd.DataFrame(res, columns=['class', 'specificity', 'sensitivity']))

#### Decision tree ####
    dt_model = DecisionTreeClassifier()
    dt_model.fit(train_X, train_y.ravel())
    y_pred = dt_model.predict(test_X)
    print("\nDecision Tree MODEL:")
    print(classification_report(test_y, y_pred, zero_division=0))
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=1)
    recall = recall_score(test_y, y_pred, average='macro', zero_division=1)
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=1)
    accuracy_list_dt.append(accuracy)
    precision_list_dt.append(precision)
    recall_list_dt.append(recall)
    f1_list_dt.append(f1)
    print("Specificity: ",utils.calculate_specificity(test_y, y_pred))
    print("Sensitivity: ",utils.calculate_sensitivity(test_y, y_pred))

    # res = []
    # for l in all_classes:
    #     prec, recall, _, _ = precision_recall_fscore_support(np.array(test_y) == l,
    #                                                          np.array(y_pred) == l,
    #                                                          pos_label=True, average=None, zero_division=0)
    #     res.append([l, recall[0], recall[1]])
    #
    # print(pd.DataFrame(res, columns=['class', 'specificity', 'sensitivity']))


#### Logistic Regression ####
    lr_model = LogisticRegression(solver='saga', max_iter=8000)
    lr_model.fit(train_X, train_y.ravel())
    y_pred = lr_model.predict(test_X)
    print("\nLogistic Regression MODEL:")
    print(classification_report(test_y, y_pred, zero_division=0))
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='macro', zero_division=1)
    recall = recall_score(test_y, y_pred, average='macro', zero_division=1)
    f1 = f1_score(test_y, y_pred, average='macro', zero_division=1)
    accuracy_list_lr.append(accuracy)
    precision_list_lr.append(precision)
    recall_list_lr.append(recall)
    f1_list_lr.append(f1)
    print("Specificity: ", utils.calculate_specificity(test_y, y_pred))
    print("Sensitivity: ", utils.calculate_sensitivity(test_y, y_pred))

    # res = []
    # for l in all_classes:
    #     prec, recall, _, _ = precision_recall_fscore_support(np.array(test_y) == l,
    #                                                          np.array(y_pred) == l,
    #                                                          pos_label=True, average=None, zero_division=0)
    #     res.append([l, recall[0], recall[1]])
    #
    # print(pd.DataFrame(res, columns=['class', 'specificity', 'sensitivity']))

avg_accuracy_svm = np.mean(accuracy_list_svm)
avg_precision_svm = np.mean(precision_list_svm)
avg_recall_svm = np.mean(recall_list_svm)
avg_f1_svm = np.mean(f1_list_svm)

avg_accuracy_dt = np.mean(accuracy_list_dt)
avg_precision_dt = np.mean(precision_list_dt)
avg_recall_dt = np.mean(recall_list_dt)
avg_f1_dt = np.mean(f1_list_dt)

avg_accuracy_lr = np.mean(accuracy_list_lr)
avg_precision_lr = np.mean(precision_list_lr)
avg_recall_lr = np.mean(recall_list_lr)
avg_f1_lr = np.mean(f1_list_lr)

print("Average Accuracy SVM:", avg_accuracy_svm)
print("Average Precision SVM:", avg_precision_svm)
print("Average Recall SVM:", avg_recall_svm)
print("Average F1 SVM:", avg_f1_svm)

print("Average Accuracy DT:", avg_accuracy_dt)
print("Average Precision DT:", avg_precision_dt)
print("Average Recall DT :", avg_recall_dt)
print("Average F1 DT :", avg_f1_dt)

print("Average Accuracy LR:", avg_accuracy_lr)
print("Average Precision LR:", avg_precision_lr)
print("Average Recall LR:", avg_recall_lr)
print("Average F1 LR:", avg_f1_lr)
