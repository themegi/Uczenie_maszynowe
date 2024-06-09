import data
import models

# print("####Automobile Dataset#####")
# auto_data, auto_cat = data.automobileRead()
# models.models_dataset(auto_data, auto_cat, 3, 5)

print("####Cleveland Dataset#####")
cleveland_data, cleveland_cat = data.clevelandRead()
models.models_dataset(cleveland_data, cleveland_cat, 10, 3)

print("####Hayes-Roth Dataset#####")
hayes_data, hayes_cat = data.hayesrothRead()
models.models_dataset(hayes_data, hayes_cat, 10, 3)

print("####Lymphography Dataset#####")
lympho_data, lympho_cat = data.lymphographyRead()
models.models_dataset(lympho_data, lympho_cat, 2, 8)

print("####Yeast Dataset#####")
yeast_data, yeast_cat = data.yeastRead()
models.models_dataset(yeast_data, yeast_cat, 5, 4)



#recall = sensitivity
