import numpy as np
import hvdm
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from collections import Counter


def sortDf(df):
    sorted_df = df.sort_values(by='Class', key=lambda x: df['Class'].value_counts().sort_values(ascending=True)[x])
    return sorted_df


def getCatIndex(df):
    cat_values = list(df.columns.get_indexer(df.select_dtypes(include=['object']).columns))
    return cat_values


def encode(data, categories):
    label_encoder = preprocessing.LabelEncoder()
    for i in categories:
        data[i] = label_encoder.fit_transform(data[i])
    processed = data.to_numpy()
    return processed


def countClasses(data, class_idx):
    labels = data[:, class_idx]
    unique_classes, counts = np.unique(labels, return_counts=True)
    class_counts = np.column_stack((unique_classes, counts))
    class_counts_sorted = np.argsort(class_counts[:, 1])
    sorted_data = class_counts[class_counts_sorted][:-1]
    max_class = class_counts[class_counts_sorted][-1]
    max_class = (int(max_class[1]))
    return sorted_data, max_class


def kNN(train_X, train_y, categories):
    hvdm_metric = hvdm.HVDM(train_X, train_y, categories, [np.nan, 0])
    neighbor = NearestNeighbors(metric=hvdm_metric.hvdm)
    neighbor.fit(train_X)
    results = np.zeros((len(train_X), 6))
    for i in range(len(train_X)):
        result = neighbor.kneighbors(train_X[i].reshape(1, -1), n_neighbors=6, return_distance=False)
        results[i] = result.copy()
    types = np.zeros(len(train_y))
    train_y = train_y.reshape(-1, 1)
    train_y = np.insert(train_y, train_y.shape[1], types, axis=1)
    for j in range(len(train_y)):
        class_counter = 0
        a_class = train_y[int(results[j][0])][0]
        for i in range(1, 6):
            if a_class == train_y[int(results[j][i])][0]:
                class_counter += 1
        if class_counter >= 4:
            train_y[j][1] = 0  # safe
        elif class_counter >= 2:
            train_y[j][1] = 1  # borderline
        elif class_counter == 1:
            train_y[j][1] = 2  # rare
        else:
            train_y[j][1] = 3  # outlier
    return train_X, train_y


def sort_by_class_quantity(train_X, train_y):
    # Count the occurrences of each class in train_y
    class_counts = Counter(train_y)
    most_common_class, max_count = class_counts.most_common(1)[0]
    print(most_common_class,max_count)

    # Create a list of classes sorted by their count (ascending)
    sorted_classes = sorted(class_counts, key=class_counts.get)

    # Create an array of indices that would sort train_y by class frequency
    sorted_indices = np.argsort([sorted_classes.index(cls) for cls in train_y])

    # Use the sorted indices to reorder train_X and train_y
    sorted_train_X = train_X[sorted_indices]
    sorted_train_y = train_y[sorted_indices]

    return sorted_train_X, sorted_train_y, max_count, most_common_class

def preprocess(data, sorted_data, max_class, class_idx):
    for k in range(len(sorted_data)):
        selected_examples = []
        other = []
        new_samples = np.zeros((max_class - int(sorted_data[k][1]), data.shape[1]))
        for i in range(max_class - int(sorted_data[k][1])):
            class_label = sorted_data[k][0]
            selected_examples = data[(data[:, class_idx] == class_label) & (data[:, 26] == 1)]
            other = data[(data[:, class_idx] == class_label) & (data[:, 26] != 1)]
            random_index = np.random.choice(selected_examples.shape[0])
            x = selected_examples[random_index]
            if other.shape[0] > 0:
                random_index = np.random.choice(other.shape[0])
                y = other[random_index]
            else:
                y = selected_examples[random_index]
            if y[26] == 0:  # safe
                new_samples[i] = x * 0.85 + y * 0.15
            elif y[26] == 1:  # borderline
                new_samples[i] = x * 0.5 + y * 0.5
            elif y[26] == 2:  # rare
                new_samples[i] = x * 0.7 + y * 0.3
            else:  # outlier
                new_samples[i] = x * 0.9 + y * 0.1
        data = np.concatenate((data, new_samples), axis=0)
    return data

def select_examples_by_class_and_type(train_X, train_y, target_class):

    # Filter indices based on class and type
    type_1_indices = np.where((train_y[:, 0] == target_class) & (train_y[:, 1] == 1))[0]
    other_types_indices = np.where((train_y[:, 0] == target_class) & (train_y[:, 1] != 1))[0]

    class_count = len(np.where((train_y[:, 0] == target_class))[0])


    # Select examples based on filtered indices
    type_X = train_X[type_1_indices]
    type_y = train_y[type_1_indices]
    other_X = train_X[other_types_indices]
    other_y = train_y[other_types_indices]

    return type_X, type_y, other_X, other_y, class_count