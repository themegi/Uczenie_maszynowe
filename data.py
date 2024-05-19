import pandas as pd
import utils


def automobileRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/autos.dat', sep=",", header=None)
    class_index = 25
    data.rename(columns={25: 'Class'}, inplace=True)
    utils.addType(data)
    utils.sortDf(data)
    cat_values = utils.getCatIndex(data) # find categorical columns
    #encoded = pd.get_dummies(data)
    #dataframe = pd.concat([data, encoded], axis=1)


    return data, class_index, cat_values


def clevelandRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/cleveland-0_vs_4.dat', sep=",", header=None)
    class_index = 13
    data.rename(columns={class_index: 'Class'}, inplace=True)
    utils.addType(data)
    return data, class_index


def hayesrothRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/hayes-roth.data', sep=",", header=None)
    class_index = 5
    data.rename(columns={class_index: 'Class'}, inplace=True)
    utils.addType(data)
    return data, class_index


def lymphographyRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/lymphography.dat', sep=",", header=None)
    class_index = 18
    data.rename(columns={class_index: 'Class'}, inplace=True)
    utils.addType(data)
    return data, class_index


def yeastRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/yeast.dat', sep=",", header=None)
    class_index = 8
    data.rename(columns={class_index: 'Class'}, inplace=True)
    utils.addType(data)
    return data, class_index


# df = utils.sortDf(automobileRead())
# df.info()
# test = utils.getClassIndex(df)
# print(test)

#print(df.columns.get_indexer(df.select_dtypes(include=['object']).columns))