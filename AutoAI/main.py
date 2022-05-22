import streamlit as st
import pandas as pd
import pandas_profiling
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
from joblib import parallel_backend
import shutil
import os
if not os.path.exists("output"):
    os.makedirs("output")
    os.makedirs("output/findings")
    os.makedirs("output/images")
    os.makedirs("output/helpers")
    os.makedirs("output/models")
def Analyze(data, isUnsupervised, problemType, useStatistical, useDeepLearning, dataType,
            isTimeSeries, dependentVars,
            categoricalVars, independentVars, isMultiInput, test_size=0.1):



    categoricalColumns = categoricalVars
    XColumns = independentVars
    yColumn = dependentVars
    with parallel_backend('threading', n_jobs=-1):
        if problemType == "association":
            X_train = data.values.tolist()
            X_train2 = []
            for i in X_train:
                cell = []
                for j in i:
                    if isinstance(j, str):
                        cell.append(j)
                X_train2.append(cell)
            X_train = X_train2
            frequent_itemset = AprioriAssociation(X_train)
            AssociationRules(frequent_itemset)
            print("association completed")
        elif problemType == "clustering":
            X_train = data.values.tolist()
            columns = data.columns
            with open('output/helpers/columns.txt', 'w') as filehandle:
                filehandle.write(str(columns))
            HyperClustering(X_train)
            print("clustering completed")
        else:
            Y, X, codes, columns = PreProcess(data, categoricalColumns, yColumn, XColumns)
            with open('output/helpers/columns.txt', 'w') as filehandle:
                filehandle.write(str(columns))
            with open('output/helpers/codes.txt', 'w') as filehandle:
                filehandle.write(str(codes))
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
            if problemType == "regression":
                if isTimeSeries:
                    if isMultiInput:
                        MultiVariateTimeSeriesForecasting(X, Y)
                        print("MultiVariateTimeSeriesForecasting completed")

                    else:
                        SingleVariableTimeSeriesForecaster(X, Y)
                        print("SingleVariableTimeSeriesForecaster completed")
                else:
                    if useStatistical:
                        if dataType == "structured":
                            HyperRegression(X_train, X_test, y_train, y_test)
                            print("Statistical regression completed")
                    if useDeepLearning:
                        if dataType == "structured":
                            AdvancedStructuredRegressor(X_train, X_test, y_train, y_test)
                        if dataType == "image":
                            AdvancedImageRegressor(X_train, X_test, y_train, y_test)
                        if dataType == "text":
                            AdvancedTextRegressor(X_train, X_test, y_train, y_test)

            elif problemType == "classification":
                if isTimeSeries:
                    if len(categoricalColumns) > 0:
                        MultiInputCategoricalNumericalSupportedTimeSeriesClassifier(X_train, X_test, y_train, y_test)

                    else:
                        NumericalSupportedTimeSeriesClassifier(X_train, X_test, y_train, y_test)

                else:
                    if useStatistical:
                        if dataType == "structured":
                            HyperClassification(X_train, X_test, y_train, y_test)
                            print("Statistical classification completed")
                    if useDeepLearning:
                        if dataType == "structured":
                            AdvancedStructuredClassifier(X_train, X_test, y_train, y_test)
                        if dataType == "image":
                            AdvancedImageClassifier(X_train, X_test, y_train, y_test)
                        if dataType == "text":
                            AdvancedTextClassifier(X_train, X_test, y_train, y_test)

            else:
                print("none")


def eda(df):
    pandas_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True).to_file("output/findings/EDAReport.html")
    with open("output/findings/EDAReport.html", "r", encoding="utf-8") as f:
        st.warning("Explorative Data Analysis created")
        st.download_button("Download Explorative Data Analysis", f, mime="text/html")

def load_data(file, _isUnsupervised):
    if _isUnsupervised:
        data = pd.read_csv(file, header=None)
        columns = None
        return data
    else:
        if file.name[-4:] == ".csv":
            data = pd.read_csv(file)
        elif file.name[-5:] == ".xslx":
            data = pd.read_excel(file)
        columns = data.columns
        # TODO file output columns
        return data

if __name__ == "__main__":
    st.write("""
    ### MRT
    **Turn your data into a miracle**
    """)

    isUnsupervised = st.checkbox('isUnsupervised')
    useStatistical = st.checkbox('useStatistical')
    useDeepLearning = st.checkbox('useDeepLearning')
    isTimeSeries = st.checkbox('isTimeSeries')
    isMultiInput = st.checkbox('isMultiInput')

    problemType = st.selectbox('Problem Type', ("regression", "classification","association","clustering"))
    dataType = st.selectbox('Data Type', ("structured", "image","text"))

    uploaded_file = st.file_uploader("Choose a spreadsheet")
    if uploaded_file is not None:
        # TODO datatype image text check
        if dataType == "structured":
            df = load_data(uploaded_file, isUnsupervised)
            eda_button = st.button("Explorative Data Analysis", on_click=lambda: eda(df))
            if not isUnsupervised:
                independent = st.multiselect(
                    'Choose independent variables', df.columns)
                dependent = st.multiselect(
                    'Choose dependent variables', df.columns)
                categorical = st.multiselect(
                    'Choose categorical variables', df.columns)
                analyze_btn = st.button("Analyze", on_click=lambda:Analyze(df, isUnsupervised, problemType, useStatistical,
                                                                           useDeepLearning, dataType, isTimeSeries, dependent,
                                                                           categorical, independent, isMultiInput, test_size=0.1))
                import os
                import zipfile
                def zip_directory(folder_path, zip_path):
                    with zipfile.ZipFile(zip_path, mode='w') as zipf:
                        len_dir_path = len(folder_path)
                        for root, _, files in os.walk(folder_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, file_path[len_dir_path:])


                zip_directory("output",'output.zip')
                with open('output.zip', 'rb') as f:
                    st.download_button('Download Zip', f, file_name='output.zip')


