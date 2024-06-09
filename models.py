import numpy as np
import utils
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def models_dataset(auto_data, auto_cat, n_splits, n_repeats):
    X, y = utils.first_proccess(auto_data, auto_cat)

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    counter = 0
    all_classes = np.unique(y)
    accuracy_list_svm = []
    precision_list_svm = []
    recall_list_svm = []
    f1_list_svm = []
    avg_sensitivity_svm = np.empty((n_splits * n_repeats, len(all_classes)))
    avg_specificity_svm = np.empty((n_splits * n_repeats, len(all_classes)))

    accuracy_list_dt = []
    precision_list_dt = []
    recall_list_dt = []
    f1_list_dt = []
    avg_sensitivity_dt = np.empty((n_splits * n_repeats, len(all_classes)))
    avg_specificity_dt = np.empty((n_splits * n_repeats, len(all_classes)))

    accuracy_list_lr = []
    precision_list_lr = []
    recall_list_lr = []
    f1_list_lr = []
    avg_sensitivity_lr = np.empty((n_splits * n_repeats, len(all_classes)))
    avg_specificity_lr = np.empty((n_splits * n_repeats, len(all_classes)))

    for train_ix, test_ix in kfold.split(X, y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        train_X, train_y, max_class, most_common = utils.sort_by_class_quantity(train_X, train_y)
        train_X, train_y = utils.kNN(train_X, train_y, auto_cat)
        unique_classes = np.unique(train_y[:, 0])
        index = np.argwhere(unique_classes == most_common)
        unique_classes = np.delete(unique_classes, index)
        train_X, train_y = utils.preprocess(train_X, train_y, max_class, unique_classes)

        #### SVM ####
        svm_model = svm.SVC()
        svm_model.fit(train_X, train_y.ravel())
        y_pred = svm_model.predict(test_X)
        # print("SVM MODEL:")
        # print(classification_report(test_y, y_pred, zero_division=0))
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='macro', zero_division=1)
        recall = recall_score(test_y, y_pred, average='macro', zero_division=1)
        f1 = f1_score(test_y, y_pred, average='macro', zero_division=1)
        accuracy_list_svm.append(accuracy)
        precision_list_svm.append(precision)
        recall_list_svm.append(recall)
        f1_list_svm.append(f1)
        avg_sensitivity_svm[counter] = utils.calculate_sensitivity(test_y, y_pred)
        avg_specificity_svm[counter] = utils.calculate_specificity(test_y, y_pred)

        #### Decision tree ####
        dt_model = DecisionTreeClassifier()
        dt_model.fit(train_X, train_y.ravel())
        y_pred = dt_model.predict(test_X)
        # print("\nDecision Tree MODEL:")
        # print(classification_report(test_y, y_pred, zero_division=0))
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='macro', zero_division=1)
        recall = recall_score(test_y, y_pred, average='macro', zero_division=1)
        f1 = f1_score(test_y, y_pred, average='macro', zero_division=1)
        accuracy_list_dt.append(accuracy)
        precision_list_dt.append(precision)
        recall_list_dt.append(recall)
        f1_list_dt.append(f1)
        avg_sensitivity_dt[counter] = utils.calculate_sensitivity(test_y, y_pred)
        avg_specificity_dt[counter] = utils.calculate_specificity(test_y, y_pred)

        #### Logistic Regression ####
        lr_model = LogisticRegression(solver='saga', max_iter=9000)
        lr_model.fit(train_X, train_y.ravel())
        y_pred = lr_model.predict(test_X)
        # print("\nLogistic Regression MODEL:")
        # print(classification_report(test_y, y_pred, zero_division=0))
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='macro', zero_division=1)
        recall = recall_score(test_y, y_pred, average='macro', zero_division=1)
        f1 = f1_score(test_y, y_pred, average='macro', zero_division=1)
        accuracy_list_lr.append(accuracy)
        precision_list_lr.append(precision)
        recall_list_lr.append(recall)
        f1_list_lr.append(f1)
        avg_sensitivity_lr[counter] = utils.calculate_sensitivity(test_y, y_pred)
        avg_specificity_lr[counter] = utils.calculate_specificity(test_y, y_pred)
        counter += 1

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

    svm_sensitivity = utils.mean_sensitivity(avg_sensitivity_svm)
    dt_sensitivity = utils.mean_sensitivity(avg_sensitivity_dt)
    lr_sensitivity = utils.mean_sensitivity(avg_sensitivity_lr)

    svm_specificity = utils.mean_specificity(avg_specificity_svm)
    dt_specificity = utils.mean_specificity(avg_specificity_dt)
    lr_specificity = utils.mean_specificity(avg_specificity_lr)

    print("Average Accuracy SVM:", avg_accuracy_svm)
    print("Average Precision SVM:", avg_precision_svm)
    print("Average Recall SVM:", avg_recall_svm)
    print("Average F1 SVM:", avg_f1_svm)
    print("Average Sensitivity SVM:", svm_sensitivity)
    print("Average Specificity SVM:", svm_specificity)

    print("Average Accuracy DT:", avg_accuracy_dt)
    print("Average Precision DT:", avg_precision_dt)
    print("Average Recall DT :", avg_recall_dt)
    print("Average F1 DT :", avg_f1_dt)
    print("Average Sensitivity DT:", dt_sensitivity)
    print("Average Specificity DT:", dt_specificity)

    print("Average Accuracy LR:", avg_accuracy_lr)
    print("Average Precision LR:", avg_precision_lr)
    print("Average Recall LR:", avg_recall_lr)
    print("Average F1 LR:", avg_f1_lr)
    print("Average Sensitivity LR:", lr_sensitivity)
    print("Average Specificity LR:", lr_specificity)

    utils.t_test(avg_sensitivity_svm, avg_sensitivity_dt, avg_sensitivity_lr)
    utils.t_test(avg_specificity_svm, avg_specificity_dt, avg_specificity_lr)

    recall_arr = np.vstack((svm_sensitivity, dt_sensitivity, lr_sensitivity))
    specificity_arr = np.vstack((svm_specificity, dt_specificity, lr_specificity))

    utils.plot_recall(recall_arr)
    utils.plot_specificity(specificity_arr)