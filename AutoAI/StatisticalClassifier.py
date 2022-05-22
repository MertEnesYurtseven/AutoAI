
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import sklearn
import scikitplot as skplt
import matplotlib.pyplot as plt
from joblib import dump,load
import pandas as pd



accuracy_scores=[]
def saver(model,filename):
    dump(model, filename+".joblib")
def makeClassification(model,X_train, X_test, y_train, y_test):
    global accuracy_scores
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(prediction, y_test)
    accuracy_scores.append([accuracy,type(model).__name__])
    saver(model,"output/models/"+type(model).__name__)



def HyperClassification(
    X_train, X_test, y_train, y_test,
    GradientBoostingLoss="deviance", #deviance,exponantial
    GradientBoostingLr=0.1,
    GradientBoostingNumberOfEstimators=100,
    GradientBoostingMaxDepth=3,
    GradientBoostingEarlyStoppingEpoch=10,
    GaussianProcessClassifierKernel=1.0 * RBF(1.0),
    GaussianProcessClassifierMaxIteration=100,
    SGDLoss="hinge",#hinge,log,modified_huber,squared_hinge,perceptron,squared_error,huber
    SGDPenalty="l2",#l2,l1,elasticnet
    SGDAlpha=0.0001,
    SGDEpochs=1000,
    SGDEarlyStoppingEpochs=5,
    SGDEarlyStopping=True,
    SGDInitialLr=0.01,
    SGDLrSchedule="optimal",#optimal,constant,invscaling,adaptive
    SVCC=1.0,
    SVCKernel="rbf",#linear,poly,rbf,sigmoid,precomputed
    SVCPolyDegree=3,
    SVCGamma="scale",#scale,auto
    KNNneighborsNumberOfNeighbors=5,
    KNNneighborsWeights="uniform",#uniform,distance
    KNNneighborsAlgorithm="auto",#auto,ball_tree,kd_tree,brute
    LogisticPenalty="elasticnet",
    LogisticSolver="saga",
    LogisticMaxIteration=100,
    MLPClassifierHiddenLayerSizes=(100,),
    MLPClassifierActivation="relu",#identity, logistic, tanh’, relu
    MLPClassifierSolver="adam",#lbfgs, sgd, adam
    MLPClassifierBatchSize=16,
    MLPClassifierLrInıt=0.001,
    MLPClassifierLrSchedule="constant",#constant, invscaling, adaptive
    MLPClassifierMomentum=0.9,
    MLPClassifierEarlyStopping=True,
    MLPClassifierEarlyStoppingEpoch=10,
    MLPClassifierNumberOfEpochs=200,
    AdaBoostEstimatorType=None,
    AdaBoostNumberOfEstimators=50,
    AdaBoostLr=1.0):
    makeClassification(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0),X_train, X_test, y_train, y_test)
    makeClassification(GaussianProcessClassifier(kernel=1.0 * RBF(1.0),random_state=0),X_train, X_test, y_train, y_test)
    makeClassification(SGDClassifier() ,X_train, X_test, y_train, y_test)
    makeClassification(SVC() ,X_train, X_test, y_train, y_test)
    makeClassification(KNeighborsClassifier(n_neighbors=5) ,X_train, X_test, y_train, y_test)
    makeClassification(LogisticRegression(),X_train, X_test, y_train, y_test)
    makeClassification(GaussianNB(),X_train, X_test, y_train, y_test)
    makeClassification(MLPClassifier(),X_train, X_test, y_train, y_test)
    makeClassification(RandomForestClassifier(),X_train, X_test, y_train, y_test)
    makeClassification(AdaBoostClassifier(),X_train, X_test, y_train, y_test)
    fig = plt.figure()

    unzipped_object = zip(*accuracy_scores)
    unzipped_list = list(unzipped_object)
    plt.bar(unzipped_list[1],unzipped_list[0])
    plt.savefig("output/findings/"+"classification_models_performance_comparison.png")
    df=pd.DataFrame([unzipped_list[0]],columns=unzipped_list[1])
    df.to_csv("output/findings/"+"classification_models_performance_comparison.csv")
