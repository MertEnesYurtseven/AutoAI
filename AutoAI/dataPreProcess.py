import pandas as pd
import numpy as np
from sklearn import preprocessing
import pathlib

def load_data(filename,type):
    global columns
    global data
    if type=="apriori":
        data = pd.read_csv(filename,header=None)
        columns=None
        return data
    else:
        extension=pathlib.Path(filename).suffix
        if extension==".csv":
            data= pd.read_csv(filename)
        elif extension==".xslx":
            data= pd.read_excel(filename)
        columns = data.columns
        print(data.head(10))
        print(data.describe())
        return data




def code_to_number(dfColumnList):
    uniques=set(dfColumnList)
    encoder=list(range(0,len(uniques)))
    return dict(zip(uniques,encoder))
def replacer(df,code):
    for i in range(len(list(code.keys()))):
        df=df.replace(to_replace=list(code.keys())[i], value=list(code.values())[i])

    return df

def PreProcess(data,categorical_columns,target_column,X_columns):
    data = data.dropna()
    data = data.reset_index(drop=True)
    codes=[]
    for column in categorical_columns:
        encoding=code_to_number(data[column].values.tolist())
        codes.append(encoding)
        data=replacer(data,encoding)
    return np.array(data[target_column].values.tolist()),np.array(data[X_columns].values.tolist()),codes,data[X_columns].columns.tolist()


