import streamlit as st
from joblib import dump,load
import os
modelChooseVars=[]
models=[]
file=[]

def select_modelChoose():
    global modelChoose
    global modelChooseVars
    global model
    for i in modelChoose.curselection():
        modelChooseVars.append(modelChoose.get(i))
        models.append(load("output/models/"+modelChoose.get(i)))

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose a model")
    if uploaded_file is not None:
        modelNames = [uploaded_file.name]#st.multiselect('models',[_ for _ in os.listdir("output/models/") if _.endswith("joblib")])
        for i in modelNames:
            models.append(load("output/models/"+i))
        parameters = st.text_input('infer', 'enter values seperating with comma')
        commas = parameters.split("\n")
        commass = []
        try:
            for i in commas:
                commass.append([float(x) for x in i.split(",")])
            commas = commass
            mytxt = ""
            for model, modelName in zip(models, modelNames):
                mytxt += modelName + " : " + str(model.predict(commas)) + "\n"
            st.write(mytxt)
        except Exception as e:
            print(e)

