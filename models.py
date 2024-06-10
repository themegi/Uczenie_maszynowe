import numpy as np
import utils
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate


def models_dataset(auto_data, auto_cat, n_splits, n_repeats):
    X, y = utils.first_process(auto_data, auto_cat)

    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    counter = 0
    all_classes = np.unique(y)
    avg_sensitivity_svm = np.empty((n_splits * n_repeats, len(all_classes)))
    avg_specificity_svm = np.empty((n_splits * n_repeats, len(all_classes)))
    avg_sensitivity_dt = np.empty((n_splits * n_repeats, len(all_classes)))
    avg_specificity_dt = np.empty((n_splits * n_repeats, len(all_classes)))
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
        avg_sensitivity_svm[counter] = utils.calculate_sensitivity(test_y, y_pred)
        avg_specificity_svm[counter] = utils.calculate_specificity(test_y, y_pred)

        #### Decision tree ####
        dt_model = DecisionTreeClassifier()
        dt_model.fit(train_X, train_y.ravel())
        y_pred = dt_model.predict(test_X)
        avg_sensitivity_dt[counter] = utils.calculate_sensitivity(test_y, y_pred)
        avg_specificity_dt[counter] = utils.calculate_specificity(test_y, y_pred)

        #### Logistic Regression ####
        lr_model = LogisticRegression(solver='saga', max_iter=9000)
        lr_model.fit(train_X, train_y.ravel())
        y_pred = lr_model.predict(test_X)
        avg_sensitivity_lr[counter] = utils.calculate_sensitivity(test_y, y_pred)
        avg_specificity_lr[counter] = utils.calculate_specificity(test_y, y_pred)
        counter += 1

    svm_sensitivity, svm_sens_std = utils.mean_sensitivity(avg_sensitivity_svm)
    dt_sensitivity, dt_sens_std = utils.mean_sensitivity(avg_sensitivity_dt)
    lr_sensitivity, lr_sens_std = utils.mean_sensitivity(avg_sensitivity_lr)

    svm_specificity, svm_spec_std = utils.mean_specificity(avg_specificity_svm)
    dt_specificity, dt_spec_std = utils.mean_specificity(avg_specificity_dt)
    lr_specificity, lr_spec_std = utils.mean_specificity(avg_specificity_lr)

    print("P-values for sensitivity: ")
    utils.t_test(avg_sensitivity_svm, avg_sensitivity_dt, avg_sensitivity_lr)
    print("P-values for specificity: ")
    utils.t_test(avg_specificity_svm, avg_specificity_dt, avg_specificity_lr)

    recall_arr = np.vstack((svm_sensitivity, dt_sensitivity, lr_sensitivity))
    specificity_arr = np.vstack((svm_specificity, dt_specificity, lr_specificity))
    recall_std = np.vstack((svm_sens_std, dt_sens_std, lr_sens_std))
    specificity_std = np.vstack((svm_spec_std, dt_spec_std, lr_spec_std))

    class_labels = utils.plot_recall(recall_arr)
    utils.plot_specificity(specificity_arr)

    recall_arr = utils.std(recall_arr, recall_std)
    specificity_arr = utils.std(specificity_arr, specificity_std)


    headers = np.array([['SVM'], ['DT'], ['LR']])
    recall_arr = np.column_stack((headers, recall_arr))
    specificity_arr = np.column_stack((headers, specificity_arr))
    print(tabulate(recall_arr, class_labels, tablefmt="latex"))
    print(tabulate(specificity_arr, class_labels, tablefmt="latex"))
