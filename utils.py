import numpy as np
import hvdm
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from collections import Counter


def sortDf(df):
    sorted_df = df.sort_values(by='Class', key=lambda x: df['Class'].value_counts().sort_values(ascending=True)[x])
    return sorted_df


def getCatIndex(df):
    cat_values = list(df.columns.get_indexer(df.select_dtypes(include=['object']).columns))
    return cat_values


def first_proccess(data, categories):
    X = data.drop(["Class"], axis=1)
    y = data["Class"]
    y = y.to_numpy()
    label_encoder = preprocessing.LabelEncoder()
    for i in categories:
        X[i] = label_encoder.fit_transform(X[i])
    X = X.to_numpy()
    return X, y


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

    # Create a list of classes sorted by their count (ascending)
    sorted_classes = sorted(class_counts, key=class_counts.get)

    # Create an array of indices that would sort train_y by class frequency
    sorted_indices = np.argsort([sorted_classes.index(cls) for cls in train_y])

    # Use the sorted indices to reorder train_X and train_y
    sorted_train_X = train_X[sorted_indices]
    sorted_train_y = train_y[sorted_indices]

    return sorted_train_X, sorted_train_y, max_count, most_common_class


def preprocess(train_X, train_y, max_class, unique_classes):
    for k in unique_classes:
        selected_examples_x, selected_examples_y, other_x, other_y, class_count = select_examples_by_class_and_type(
            train_X, train_y, k)
        if len(selected_examples_x) > 0:
            new_samples = np.zeros((max_class - class_count, train_X.shape[1]))
            new_samples_y = np.zeros((max_class - class_count, 2))
            for i in range(len(new_samples)):
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
            train_X = np.concatenate((train_X, new_samples), axis=0)
            train_y = np.concatenate((train_y, new_samples_y), axis=0)
    train_y = np.delete(train_y, 1, 1)
    return train_X, train_y


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
