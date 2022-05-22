from joblib import dump,load
import os
from numpy import array
from dataPreProcess import load_data
import tkinter as tk
import numpy as np
root=tk.Tk()
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
    print(modelChooseVars)



modelNames=[_ for _ in os.listdir("output/models/" ) if _.endswith("joblib")]
print(modelNames)

modelChoose = tk.Listbox(root, selectmode="multiple")
modelChoose.pack(padx=10, pady=10, expand=tk.YES, fill="both")
x=modelNames
for each_item in range(len(x)):
    modelChoose.insert(tk.END, x[each_item])
    modelChoose.itemconfig(each_item, bg="lime")
tk.Button(root, text='select modelChoose', command=select_modelChoose).pack()

label = tk.Label(root, text=open("output/helpers/columns.txt").read()).pack() # Creating a label
text=tk.Text(root)
text.pack()
text.insert(tk.END, "After delete this sentence please enter parameters according to the order above seperating them with comma")

def infer():
    global text
    global models
    global modelChooseVars
    commas= text.get("1.0", "end-1c")
    commas=commas.split("\n")
    commass=[]
    for i in commas:
        commass.append([ float(x) for x in i.split(",")])
    print(commass)

    commas=commass
    mytxt=""
    for model,modelName in zip(models,modelChooseVars):

        mytxt+=modelName+" : "+str(model.predict(commas)) + "\n"
    text.delete('1.0', tk.END)
    text.insert(tk.END,mytxt)

tk.Button(root, text='infer', command=infer).pack()



root.mainloop()