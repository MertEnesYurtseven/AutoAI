from sklearn import datasets
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump,load
import seaborn as sns
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge

def saver(model,filename):
    dump(model, filename+".joblib")

def create_comparison_plot(y1,y2,name1,name2):
    t = np.arange(0, len(y1), 1)
    data1 = y1
    data2 = y2

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('')
    ax1.set_ylabel(name1, color=color)
    ax1.scatter(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    color = 'tab:blue'

    ax1.scatter(t, data2, color=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.ylim(0, max(max(data1)*1.1,max(data2)*1.1))
    plt.savefig("output/images/"+name2+".png")
    #plt.show()

def makeRegression(regressor,regressorName,X_train,y_train,X_test,y_test):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    create_comparison_plot(y_test,y_pred,"real",regressorName)
    saver(regressor, "output/models/"+regressorName)
    return pd.DataFrame([[regressorName, mae, mse, rmse, r2,mape]],columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score',"MAPE"])

def HyperRegression(
    X_train, X_test, y_train, y_test,
    polyDegree=2,
    SVRKernel="rbf",
    randomForestEstimators=200,
    adaBoostEstimatorType=None,
    adaBoostNumberOfEstimators=50,
    adaBoostLr=1.0,
    gradientBoostingLr=1.0,
    gradientBoostingNumberOfEstimators=100,
    gradientBoostingMaxDepth=3,
    xgbBooster="gbtree" ,#gptree,gblinear,dart
    xgbLr=0.5,
    lassoCvNumberOfAlphas=100,
    lassoCvMaxIteration=1000,
    lassoCvCv=5,
    elasticNetL1Ratio=0.5,
    ElasticNetCVCv=5,
    ransacRegressionBaseEstimator=None,#sklearn model object
    SGDRegressorMaxIteration=1000,
    SGDEarlyStopping=True,
    KNNNumberOfNeighbors=5,
    KNNWeights="uniform",#uniform,distance
    larsCvMaxIteration=500,
    larsCvCv=5,
    larsCvMaxNumberOfAlphas=1000,
    orthogonalMatchingPursuitCvCv=5,
    passiveAggressiveRegressorC=1.0,
    passiveAggressiveRegressorNumberOfIteration=1000,
    MLPRegressorHiddenLayerSizes=(100,),
    MLPRegressorActivation="relu",#identity, logistic, tanh’, relu
    MLPRegressorSolver="adam",#lbfgs, sgd, adam
    MLPRegressorBatchSize=16,
    MLPRegressorLrInıt=0.001,
    MLPRegressorLrSchedule="constant",#constant, invscaling, adaptive
    MLPRegressorMomentum=0.9,
    MLPRegressorEarlyStopping=True,
    MLPRegressorEarlyStoppingEpoch=10,
    MLPRegressorNumberOfEpochs=200):
    regressor=LinearRegression()
    results = makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test)
    poly_reg = PolynomialFeatures(degree = polyDegree)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = SVR(kernel = SVRKernel)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = DecisionTreeRegressor(random_state=0)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = RandomForestRegressor(n_estimators=randomForestEstimators, random_state=0)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = AdaBoostRegressor(base_estimator=adaBoostEstimatorType,n_estimators=adaBoostNumberOfEstimators,learning_rate=adaBoostLr,random_state=0)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = GradientBoostingRegressor(learning_rate=gradientBoostingLr,n_estimators=gradientBoostingNumberOfEstimators,max_depth=gradientBoostingMaxDepth)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = XGBRegressor(booster=xgbBooster,learning_rate=xgbLr)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    model = LassoCV(n_alphas=lassoCvNumberOfAlphas,cv=lassoCvCv, random_state=0, max_iter=lassoCvMaxIteration)
    model.fit(X_train, y_train)
    regressor = Lasso(alpha=model.alpha_)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor = ElasticNetCV(cv=ElasticNetCVCv, l1_ratio=elasticNetL1Ratio)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor=RANSACRegressor(base_estimator=ransacRegressionBaseEstimator)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor=SGDRegressor(max_iter=SGDRegressorMaxIteration,early_stopping=SGDEarlyStopping)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor=BayesianRidge()
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor= KNeighborsRegressor(n_neighbors=KNNNumberOfNeighbors,weights=KNNWeights)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor=LarsCV(cv=larsCvCv,max_iter=larsCvMaxIteration,max_n_alphas=larsCvMaxNumberOfAlphas)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor= OrthogonalMatchingPursuitCV(cv=orthogonalMatchingPursuitCvCv)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor=PassiveAggressiveRegressor(max_iter=passiveAggressiveRegressorNumberOfIteration, random_state=0,C=passiveAggressiveRegressorC)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    regressor= MLPRegressor(random_state=0,hidden_layer_sizes=MLPRegressorHiddenLayerSizes,activation=MLPRegressorActivation,solver=MLPRegressorSolver,batch_size=MLPRegressorBatchSize,learning_rate=MLPRegressorLrSchedule,learning_rate_init=MLPRegressorLrInıt,max_iter=MLPRegressorNumberOfEpochs,momentum=MLPRegressorMomentum,early_stopping=MLPRegressorEarlyStopping,n_iter_no_change=MLPRegressorEarlyStoppingEpoch)
    results = results.append(makeRegression(regressor,type(regressor).__name__,X_train,y_train,X_test,y_test))
    result=results.sort_values(by=['MAPE'], ascending=True)
    result.to_csv("output/findings/"+"regression_models_performance_comparison.csv")


