import numpy as np
import pandas as pd
import data
import utils

auto_data, auto_class, auto_cat, classes = data.automobileRead()

cars = utils.encode(auto_data, auto_cat)
sorted_data, max_class = utils.countClasses(cars, auto_class)
cars = utils.kNN(cars, auto_class, auto_cat)
cars = utils.preprocess(cars, sorted_data, max_class, auto_class)


print(cars.shape)
labels = cars[:, auto_class]
unique_classes, counts = np.unique(labels, return_counts=True)
class_counts = np.column_stack((unique_classes, counts))
print(class_counts)




