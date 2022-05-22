from tkinter import filedialog as fd
import tkinter.messagebox
import tkinter as tk
import pandas as pd
from dataPreProcess import load_data
from visualizer import get_profile


root=tk.Tk()


def EDAf():
    global data
    get_profile(data)
EDA = tk.Button(root,text='EDA',command=EDAf)

EDA.pack(expand=True)


useStatistical=tk.IntVar()
useDeepLearning=tk.IntVar()
isTimeSeries=tk.IntVar()
useAdvancedMethods=tk.IntVar()
isMultiInput=tk.IntVar()
isUnsupervised=tk.IntVar()


isUnsupervisedCheck=tk.Checkbutton(root, text='isUnsupervised',variable=isUnsupervised, onvalue=1, offvalue=0)
useStatisticalCheck=tk.Checkbutton(root, text='useStatistical',variable=useStatistical, onvalue=1, offvalue=0)
useDeepLearningCheck=tk.Checkbutton(root, text='useDeepLearning',variable=useDeepLearning, onvalue=1, offvalue=0)
isTimeSeriesCheck=tk.Checkbutton(root, text='isTimeSeries',variable=isTimeSeries, onvalue=1, offvalue=0)
useAdvancedMethodsCheck=tk.Checkbutton(root, text='useAdvancedMethods',variable=useAdvancedMethods, onvalue=1, offvalue=0)
isMultiInputCheck=tk.Checkbutton(root, text='isMultiInput',variable=isMultiInput, onvalue=1, offvalue=0)
useStatisticalCheck.pack()
useDeepLearningCheck.pack()
isTimeSeriesCheck.pack()
useAdvancedMethodsCheck.pack()
isMultiInputCheck.pack()
isUnsupervisedCheck.pack()
file=""
data=[]
independent=[]
dependent=[]
categorical=[]
independentVars=[]
dependentVars=[]
categoricalVars=[]
columns=[]
chooseFolderFormat=[]
DatasetFormat=[]
def select_file():
    import tkinter.filedialog
    global root
    global independent
    global chooseFolderFormat
    global file
    global DatasetFormat
    global data
    global dependent
    global categorical
    global isUnsupervised
    if dataType.get()=="structured":
        file = fd.askopenfilename()
        data=load_data(file,isUnsupervised)


        independent = tk.Listbox(root, selectmode="multiple")
        independent.pack(padx=10, pady=10, expand=tk.YES, fill="both")
        x=data.columns
        for each_item in range(len(x)):
            independent.insert(tk.END, x[each_item])
            independent.itemconfig(each_item, bg="lime")
        tk.Button(root, text='select independent', command=select_independent).pack()
        dependent = tk.Listbox(root, selectmode="multiple")
        dependent.pack(padx=10, pady=10, expand=tk.YES, fill="both")
        x = data.columns
        for each_item in range(len(x)):
            dependent.insert(tk.END, x[each_item])
            dependent.itemconfig(each_item, bg="lime")
        tk.Button(root, text='select dependent', command=select_dependent).pack()
        categorical = tk.Listbox(root, selectmode="multiple")
        categorical.pack(padx=10, pady=10, expand=tk.YES, fill="both")
        x = data.columns
        for each_item in range(len(x)):
            categorical.insert(tk.END, x[each_item])
            categorical.itemconfig(each_item, bg="lime")
        tk.Button(root, text='select categorical', command=select_categorical).pack()
    elif dataType.get()=="image":
        if problemType.get()=="classification":
            DatasetFormat= tk.StringVar()
            DatasetFormat.set("fromsubfolders")  # default value
            DatasetFormatMenu = tk.OptionMenu(root,  DatasetFormat, "fromsubfolders", "fromImgFolderAndcCsv","fromImgAndTxtFilesInFolders")
            DatasetFormatMenu.pack()
        elif problemType.get()=="regression":
            DatasetFormat= tk.StringVar()
            DatasetFormat.set("fromImgAndTxt")  # default value
            DatasetFormatMenu = tk.OptionMenu(root,  DatasetFormat, "fromImgAndTxt")
            DatasetFormatMenu.pack()
    elif dataType.get() == "text":
        if problemType.get() == "classification":
            DatasetFormat = tk.StringVar()
            DatasetFormat.set("fromsubfolders")  # default value
            DatasetFormatMenu = tk.OptionMenu(root, DatasetFormat, "fromsubfolders", "fromTextAndLabelCsv")
            DatasetFormatMenu.pack()
        elif problemType.get() == "regression":
            DatasetFormat = tk.StringVar()
            DatasetFormat.set("fromTextAndValueCsv")  # default value
            DatasetFormatMenu = tk.OptionMenu(root, DatasetFormat, "fromTextAndValueCsv")
            DatasetFormatMenu.pack()


open_button = tk.Button(root,text='Open a File',command=select_file)

open_button.pack(expand=True)

problemType = tk.StringVar()
problemType.set("regression") # default value

problemTypeMenu = tk.OptionMenu(root, problemType, "regression", "classification","association","clustering")
problemTypeMenu.pack()

dataType = tk.StringVar()
dataType.set("structured") # default value

dataTypeMenu = tk.OptionMenu(root, dataType, "structured", "image","text")
dataTypeMenu.pack()


def select_independent():
    global independent
    global independentVars
    for i in independent.curselection():
        independentVars.append(independent.get(i))
    print(independentVars)
def select_dependent():
    global dependent
    global dependentVars
    for i in dependent.curselection():
        dependentVars.append(dependent.get(i))
    print(dependentVars)
def select_categorical():
    global categorical
    global categoricalVars
    for i in categorical.curselection():
        categoricalVars.append(categorical.get(i))

    print(categoricalVars)


def select_FolderFormat():
    global chooseFolderFormat
    global categoricalVars
    for i in categorical.curselection():
        categoricalVars.append(categorical.get(i))

    print(categoricalVars)
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
from visualizer import get_profile
from TextRegression import *
from TextClassification import *
from TimeSeriesClassification import *
from TimeSeriesForecasting import *
from visualizer import *
def Analyze():
    from joblib import parallel_backend0

    with parallel_backend('threading', n_jobs=-1):
        global isUnsupervised
        global problemType
        global useStatistical
        global useDeepLearning
        global dataType
        global isTimeSeries
        global useAdvancedMethods
        global dependentVars
        global categoricalVars
        global independentVars
        global file
        global isMultiInput

        print("isUnsupervised",isUnsupervised.get())
        print("problemType",problemType.get())
        print("useStatistical",useStatistical.get())
        print("useDeepLearning",useDeepLearning.get())
        print("dataType",dataType.get())
        print("isTimeSeries",isTimeSeries.get())
        print("useAdvancedMethod",useAdvancedMethods.get())
        print("dependentVars",dependentVars)
        print("categoricalVars",categoricalVars)
        print("independentVars",independentVars)
        print("file",file)
        print("isMultiInput",isMultiInput.get())
        isApriori=isUnsupervised.get()
        problemType=problemType.get()
        useStatistical=useStatistical
        useDeepLearning=useDeepLearning.get()
        dataType=dataType.get()
        isTimeSeries=isTimeSeries.get()
        useAdvancedMethods=useAdvancedMethods.get()
        yColumn=dependentVars
        categoricalColumns=categoricalVars
        XColumns=independentVars
        filename=file
        isMultiInput=isMultiInput.get()
        test_size = 0.1
        global data
        print("starting")

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
            print("association completed")
            tkinter.messagebox.showinfo("Results are ready", "Check out the output folder")
        elif problemType=="clustering":
            X_train = data.values.tolist()
            columns=data.columns
            with open('output/helpers/columns.txt', 'w') as filehandle:
                filehandle.write(str(columns))
            HyperClustering(X_train)
            print("clustering completed")
            tkinter.messagebox.showinfo("Results are ready", "Check out the output folder")
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
                        print("MultiVariateTimeSeriesForecasting completed")

                    else:
                        SingleVariableTimeSeriesForecaster(X,Y)
                        print("SingleVariableTimeSeriesForecaster completed")
                    tkinter.messagebox.showinfo("Results are ready", "Check out the output folder")
                else:
                    if useStatistical:
                        if dataType=="structured":
                            HyperRegression(X_train, X_test, y_train, y_test)
                            print("Statistical regression completed")
                            tkinter.messagebox.showinfo("Results are ready", "Check out the output folder")
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
                    tkinter.messagebox.showinfo("Results are ready", "Check out the output folder")
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
                            print("Statistical classification completed")
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
                    tkinter.messagebox.showinfo("Results are ready", "Check out the output folder")
            else:
                print("none")



tk.Button(root, text='Analyze', command=Analyze).pack()
root.mainloop()