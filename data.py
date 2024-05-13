import pandas as pd

def automobileRead():

    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/autos.dat', sep =",", header=None)
    print(len(data.columns))
    data.rename(columns={25: 'Class'}, inplace=True)
    data.insert(len(data.columns), 'Type', 'NaN')
    data.sort_values(by='Class', inplace=True)
    print(data['Class'].value_counts())
    print(data)
    return data

def clevelandRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/cleveland-0_vs_4.dat', sep=",", header=None)
    print(len(data.columns))
    data.rename(columns={13: 'Class'}, inplace=True)
    data.insert(len(data.columns), 'Type', 'NaN')
    return data

def hayesrothRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/hayes-roth.data', sep=",", header=None)
    print(len(data.columns))
    data.rename(columns={5: 'Class'}, inplace=True)
    data.insert(len(data.columns), 'Type', 'NaN')
    return data

def lymphographyRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/lymphography.dat', sep=",", header=None)
    print(len(data.columns))
    data.rename(columns={18: 'Class'}, inplace=True)
    data.insert(len(data.columns), 'Type', 'NaN')
    return data

def yeastRead():
    data = pd.read_csv('C:/Users/Megi/Studia/UM/Projekt/Datasets/yeast.dat', sep=",", header=None)
    print(len(data.columns))
    data.rename(columns={8: 'Class'}, inplace=True)
    data.insert(len(data.columns), 'Type', 'NaN')
    return data
automobileRead()