def sortDf(df):
    sorted_df = df.sort_values(by='Class', key=lambda x: df['Class'].value_counts().sort_values(ascending=True)[x])
    return sorted_df


def addType(df):
    df.insert(len(df.columns), 'Type', 'NaN')
