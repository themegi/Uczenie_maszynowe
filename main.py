import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter

import data
import utils
import hvdm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

auto_data, auto_class, auto_cat, classes = data.automobileRead()
X = auto_data.drop(["Class"], axis=1)
y = auto_data["Class"]


y = y.to_numpy()
label_encoder = preprocessing.LabelEncoder()
for i in auto_cat:
    X[i] = label_encoder.fit_transform(X[i])
X = X.to_numpy()
for train_ix, test_ix in kfold.split(X, y):
    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
    train_0, train_1 = len(train_y[train_y == 0]), len(train_y[train_y == 1])
    test_0, test_1 = len(test_y[test_y == 0]), len(test_y[test_y == 1])
    print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
    train_X, train_y, max_class, most_common = utils.sort_by_class_quantity(train_X,train_y)
    train_X, train_y = utils.kNN(train_X,train_y, auto_cat)
    #print(train_y)
    unique_classes = np.unique(train_y[:, 0])
    print(unique_classes)
    index = np.argwhere(unique_classes == most_common)
    unique_classes = np.delete(unique_classes, index)
    print(unique_classes)
    for k in unique_classes:
        selected_examples_x, selected_examples_y, other_x, other_y, class_count = utils.select_examples_by_class_and_type(train_X, train_y, k)
        if len(selected_examples_x) > 0:
            new_samples = np.zeros((max_class - class_count, train_X.shape[1]))
            new_samples_y = np.zeros((max_class - class_count, 2))
            for i in range(len(new_samples)):
            # class_indices = np.where((train_y[0] == class_label) & (train_y[1] == 1))[0]
            #print(class_indices)
            # selected_examples_x, selected_examples_y = train_X[class_indices], train_y[class_indices]
            #selected_examples = train_X[(train_y[:, 0] == class_label) & (train_y[:, 1] == 1)]
            # other_x, other_y = train_X[(train_y[:, 0] == class_label) & (train_y[:, 1] != 1)], train_y[(train_y[:, 0] == class_label) & (train_y[:, 1] != 1)]
            #print(len(selected_examples))
                random_index = np.random.choice(selected_examples_x.shape[0])
                x_x, x_y = selected_examples_x[random_index], selected_examples_y[random_index]
                if other_x.shape[0] > 0:
                    random_index = np.random.choice(other_x.shape[0])
                    y_x, y_y = other_x[random_index], other_y[random_index]
                else:
                    y_x, y_y = selected_examples_x[random_index], selected_examples_y[random_index]
                if y_y[1] == 0:  # safe
                    new_samples[i] = x_x * 0.85 + y_x * 0.15
                    new_samples_y[i] = x_y
                elif y_y[1] == 1:  # borderline
                    new_samples[i] = x_x * 0.5 + y_x * 0.5
                    new_samples_y[i] = x_y
                elif y_y[1] == 2:  # rare
                    new_samples[i] = x_x * 0.7 + y_x * 0.3
                    new_samples_y[i] = x_y
                else:  # outlier
                    new_samples[i] = x_x * 0.9 + y_x * 0.1
                    new_samples_y[i] = x_y
            print(new_samples)
            train_X = np.concatenate((train_X, new_samples), axis=0)
            train_y = np.concatenate((train_y, new_samples_y), axis=0)
    train_y = np.delete(train_y, 1, 1)
    print(Counter(train_y[:, 0]))


# cars = utils.encode(auto_data, auto_cat)
# sorted_data, max_class = utils.countClasses(cars, auto_class)
# cars = utils.kNN(cars, auto_class, auto_cat)
# cars = utils.preprocess(cars, sorted_data, max_class, auto_class)
#
#
# print(cars.shape)
# labels = cars[:, auto_class]
# unique_classes, counts = np.unique(labels, return_counts=True)
# class_counts = np.column_stack((unique_classes, counts))
# print(class_counts)
