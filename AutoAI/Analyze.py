import pandas as pd
import numpy as np
from dataPreProcess import *
from AssociationFinder import *
from Clusturer import *
from ImageRegression import *
from ImageClassification import *
from StatisticalRegressor import *
from StatisticalClassifier import *
from StructuredDataRegression import *
from StructuredDataClassification import *
from TextRegression import *
from TextClassification import *
from TimeSeriesClassification import *
from TimeSeriesForecasting import *
from visualizer import *
def Analyze(
    isApriori,
    problemType,
    useStatistical,
    useDeepLearning,
    dataType,#structured,text,image
    isTimeSeries,
    useAdvancedMethods,
    yColumn,
    categoricalColumns,
    XColumns,
    filename,
    isMultiInput,
    test_size):

    data = load_data(filename,isApriori)


    if problemType=="association":
        X_train=data.values.tolist()
        X_train2=[]
        for i in X_train:
            cell = []
            for j in i:
                if isinstance(j,str):
                    cell.append(j)
            X_train2.append(cell)
        X_train=X_train2
        frequent_itemset=AprioriAssociation(X_train)
        AssociationRules( frequent_itemset)
    elif problemType=="clustering":
        X_train = data.values.tolist()
        columns=data.columns
        with open('output/helpers/columns.txt', 'w') as filehandle:
            filehandle.write(str(columns))
        HyperClustering(X_train)
    else:
        Y, X, codes, columns = PreProcess(data, categoricalColumns, yColumn, XColumns)
        with open('output/helpers/columns.txt', 'w') as filehandle:
            filehandle.write(str(columns))
        with open('output/helpers/codes.txt', 'w') as filehandle:
            filehandle.write(str(codes))
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test =  train_test_split(X, Y, test_size=test_size, random_state=42)
        if problemType=="regression":
            if isTimeSeries:
                if isMultiInput:
                    MultiVariateTimeSeriesForecasting(X,Y)
                else:
                    SingleVariableTimeSeriesForecaster(X,Y)
            else:
                if useStatistical:
                    if dataType=="structured":
                        HyperRegression(X_train, X_test, y_train, y_test)
                if useDeepLearning:
                    if useAdvancedMethods:
                        if dataType == "structured":
                            AdvancedStructuredRegressor(X_train, X_test, y_train, y_test)
                        if dataType == "image":
                            AdvancedImageRegressor(X_train, X_test, y_train, y_test)
                        if dataType == "text":
                            AdvancedTextRegressor(X_train, X_test, y_train, y_test)
                    else:
                        if dataType == "structured":
                            StandartStructuredRegressor(filename,filename,yLabel=yColumn[0])
                        if dataType == "image":
                            StandartImageRegressor(X_train, X_test, y_train, y_test)
                        if dataType == "text":
                            StandartTextRegressor(X_train, X_test, y_train, y_test)
        elif problemType=="classification":
            if isTimeSeries:
                if len(categoricalColumns)>0:
                    MultiInputCategoricalNumericalSupportedTimeSeriesClassifier(X_train, X_test, y_train, y_test)
                else:
                    NumericalSupportedTimeSeriesClassifier(X_train, X_test, y_train, y_test)
            else:
                if useStatistical:
                    if dataType == "structured":
                        HyperClassification(X_train, X_test, y_train, y_test)
                if useDeepLearning:
                    if useAdvancedMethods:
                        if dataType == "structured":
                            AdvancedStructuredClassifier(X_train, X_test, y_train, y_test)
                        if dataType == "image":
                            AdvancedImageClassifier(X_train, X_test, y_train, y_test)
                        if dataType == "text":
                            AdvancedTextClassifier(X_train, X_test, y_train, y_test)
                    else:
                        if dataType == "structured":
                            StandartStructuredClassifier(filename,filename,yLabel=yColumn[0])
                        if dataType == "image":
                            StandartImageClassifier(X_train, X_test, y_train, y_test)
                        if dataType == "text":
                            StandartTextClassifier(X_train, X_test, y_train, y_test)
        else:
            pass




