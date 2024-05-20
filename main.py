import numpy as np
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import vdm
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import data


class HVDM(vdm.VDM):
    def __init__(self, X, y_ix, cat_ix, nan_equivalents=[np.nan, 0], normalised="variance"):
        """ Heterogeneous Value Difference Metric
        Distance metric class which initializes the parameters
        used in hvdm() function

        Parameters
        ----------
        X : array-like of shape = [n_rows, n_features]
            Dataset that will be used with HVDM. Needs to be provided
            here because minimum and maximimum values from numerical
            columns have to be extracted

        y_ix : int array-like, list of shape [1]
            Single element array with indices for the categorical output variable
            If y is numerical it should be converted to categorical (if it makes sense)

        cat_ix : array-like of shape = [cat_columns_number]
            List containing categorical feature indices

        cat_ix : array-like of shape = [x]
            List containing missing values indicators

        normalised: string
            Normalises euclidan distance function for numerical variables
            Can be set as "std". The other option is a column range

        Returns
        -------
        None
        """
        # Initialize VDM object
        super().__init__(X, y_ix, cat_ix)
        self.nan_eqvs = nan_equivalents
        self.cat_ix = cat_ix
        self.col_ix = [i for i in range(X.shape[1])]
        # Get the normalization scheme for numerical variables
        if normalised == "std":
            self.range = 4 * np.nanstd(X, axis=0)
        else:
            self.range = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)

    def hvdm(self, x, y):
        """ Heterogeneous Value Difference Metric
        Distance metric function which calculates the distance
        between two instances. Handles heterogeneous data and missing values.
        For categorical variables, it uses conditional probability 
        that the output class is given 'c' when attribute 'a' has a value of 'n'.
        For numerical variables, it uses a normalized Euclidan distance.
        It can be used as a custom defined function for distance metrics
        in Scikit-Learn

        Parameters
        ----------
        x : array-like of shape = [n_features]
            First instance 

        y : array-like of shape = [n_features]
            Second instance
        Returns
        -------
        result: float
            Returns the result of the distance metrics function
        """
        # Initialise results array
        results_array = np.zeros(x.shape)

        # Get indices for missing values, if any
        nan_x_ix = np.flatnonzero(np.logical_or(np.isin(x, self.nan_eqvs), np.isnan(x)))
        nan_y_ix = np.flatnonzero(np.logical_or(np.isin(y, self.nan_eqvs), np.isnan(y)))
        nan_ix = np.unique(np.concatenate((nan_x_ix, nan_y_ix)))
        # Calculate the distance for missing values elements
        results_array[nan_ix] = 1

        # Get categorical indices without missing values elements
        cat_ix = np.setdiff1d(self.cat_ix, nan_ix)
        # Calculate the distance for categorical elements
        results_array[cat_ix] = super().vdm(x, y, nan_ix)[cat_ix]
        # Get numerical indices without missing values elements
        num_ix = np.setdiff1d(self.col_ix, self.cat_ix)
        num_ix = np.setdiff1d(num_ix, nan_ix)
        # Calculate the distance for numerical elements
        results_array[num_ix] = np.abs(x[num_ix] - y[num_ix]) / self.range[num_ix]

        # Return the final result
        # Square root is not computed in practice
        # As it doesn't change similarity between instances
        return np.sum(np.square(results_array))


auto_data, auto_class, auto_cat, classes = data.automobileRead()
x = []




label_encoder = preprocessing.LabelEncoder()
for i in  auto_cat:
    auto_data[i] = label_encoder.fit_transform(auto_data[i])
autos = auto_data.to_numpy()
results = np.zeros((len(autos), 6))

labels = autos[:, 25]
unique_classes, counts = np.unique(labels, return_counts=True)
class_counts = np.column_stack((unique_classes, counts))
class_counts_sorted = np.argsort(class_counts[:, 1])
sorte_data = class_counts[class_counts_sorted][:-1]
print(sorte_data)
max_class = class_counts[class_counts_sorted][-1]
max_class = (int(max_class[1]))


# boston = load_boston()
# boston_data = boston['data']
# print (boston_data)
# categorical_ix = [3,8]
# hvdm_metric = HVDM(boston_data, 8, categorical_ix, [np.nan, 0])
# neighbor = NearestNeighbors(metric=hvdm_metric.hvdm)
# neighbor.fit(boston_data)
# result = neighbor.kneighbors(boston_data[0].reshape(1, -1), n_neighbors = 6, return_distance=False)
# print(result)


hvdm_metric = HVDM(autos, auto_data.columns.get_loc('Class'), auto_cat, [np.nan, 0])
neighbor = NearestNeighbors(metric=hvdm_metric.hvdm)
neighbor.fit(autos)
for i in range(len(autos)):
    result = neighbor.kneighbors(autos[i].reshape(1, -1), n_neighbors = 6, return_distance=False)
    results[i] = result.copy()
types = np.zeros(len(autos))
autos = np.insert(autos, autos.shape[1], types, axis=1)
for j in range(len(autos)):
    class_counter = 0
    a_class = autos[int(results[j][0])][auto_class]
    for i in range(1, 6):
        if a_class == autos[int(results[j][i])][auto_class]:
            class_counter += 1

    if class_counter >= 4:
        autos[j][autos.shape[1] - 1] = 0  # safe
    elif class_counter >= 2:
        autos[j][autos.shape[1] - 1] = 1  # borderline
    elif class_counter == 1:
        autos[j][autos.shape[1] - 1] = 2  # rare
    else:
        autos[j][autos.shape[1] - 1] = 3  # outlier


for k in range(len(sorte_data)):
    selected_examples = []
    other = []
    for i in range(max_class - int(sorte_data[k][1])):
        #print(sorte_data[k][1])
        class_label = sorte_data[k][0]
        selected_examples = autos[(autos[:, auto_class] == class_label) & (autos[:, 26] == 1)]
        other = autos[(autos[:, auto_class] == class_label) & (autos[:, 26] != 1)]
    # for j in range(int(sorte_data[k][1])):
        random_index = np.random.choice(selected_examples.shape[0])
        x = selected_examples[random_index]
        if other.shape[0] > 0:
            random_index = np.random.choice(other.shape[0])
            y = other[random_index]
        else:
            y = selected_examples[random_index]
        print(x, y, "\n")



# result = neighbor.kneighbors(autos[0].reshape(1, -1), n_neighbors=6, return_distance=False)
# print(result[0][0])
# print(autos[0])




