import numpy as np
import hvdm
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing


def sortDf(df):
    sorted_df = df.sort_values(by='Class', key=lambda x: df['Class'].value_counts().sort_values(ascending=True)[x])
    return sorted_df


def addType(df):
    df.insert(len(df.columns), 'Type', 'NaN')


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


def kNN(data, class_idx, categories):
    hvdm_metric = hvdm.HVDM(data, class_idx, categories, [np.nan, 0])
    neighbor = NearestNeighbors(metric=hvdm_metric.hvdm)
    neighbor.fit(data)
    results = np.zeros((len(data), 6))
    for i in range(len(data)):
        result = neighbor.kneighbors(data[i].reshape(1, -1), n_neighbors=6, return_distance=False)
        results[i] = result.copy()
    types = np.zeros(len(data))
    data = np.insert(data, data.shape[1], types, axis=1)
    print(data.shape)
    for j in range(len(data)):
        class_counter = 0
        a_class = data[int(results[j][0])][class_idx]
        for i in range(1, 6):
            if a_class == data[int(results[j][i])][class_idx]:
                class_counter += 1
        if class_counter >= 4:
            data[j][data.shape[1] - 1] = 0  # safe
        elif class_counter >= 2:
            data[j][data.shape[1] - 1] = 1  # borderline
        elif class_counter == 1:
            data[j][data.shape[1] - 1] = 2  # rare
        else:
            data[j][data.shape[1] - 1] = 3  # outlier
    return data


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
