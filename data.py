import pandas as pd
import utils


def automobileRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/autos.dat', sep=",", header=None)
    data.rename(columns={25: 'Class'}, inplace=True)
    cat_values = utils.getCatIndex(data) # find categorical columns
    return data, cat_values


def clevelandRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/cleveland-0_vs_4.dat', sep=",", header=None)
    data.rename(columns={13: 'Class'}, inplace=True)
    cat_values = utils.getCatIndex(data) # find categorical columns
    return data, cat_values


def hayesrothRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/hayes-roth.data', sep=",", header=None)
    data.rename(columns={5: 'Class'}, inplace=True)
    cat_values = utils.getCatIndex(data) # find categorical columns
    return data, cat_values


def lymphographyRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/lymphography.dat', sep=",", header=None)
    data.rename(columns={18: 'Class'}, inplace=True)
    cat_values = utils.getCatIndex(data) # find categorical columns
    return data, cat_values


def yeastRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/yeast.dat', sep=",", header=None)
    data.rename(columns={8: 'Class'}, inplace=True)
    cat_values = utils.getCatIndex(data) # find categorical columns
    return data, cat_values
