


def saver(model,filename):
    from joblib import dump, load
    dump(model,    "output/"+filename+".joblib")

def makeCluestering(X,model):
    import sklearn
    import pandas as pd
    from numpy import unique
    from numpy import where
    import matplotlib.pyplot as plt
    model.fit(X)
    # assign a cluster to each example
    yhat = model.fit_predict(X)
    #df=pd.DataFrame({"features":X,"class":yhat})
    mylist=[]
    for index in range(len(yhat)):
        cell=[]
        for feature in X[index]:
            cell.append(feature)
        cell.append(yhat[index])
        mylist.append(cell)
    df = pd.DataFrame(mylist)
    saver(model,(type(model).__name__))
    df.to_csv("output/models/"+type(model).__name__+".csv")


def HyperClustering(X):
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import Birch
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import KMeans
    from sklearn.cluster import SpectralClustering
    from sklearn.mixture import GaussianMixture
    model = AffinityPropagation(damping=0.9)
    makeCluestering(X,model)

    #model = AgglomerativeClustering(n_clusters=2)
    #makeCluestering(X,model)

    model = Birch(threshold=0.01, n_clusters=2)
    makeCluestering(X,model)

    model = DBSCAN(eps=0.30, min_samples=9)
    makeCluestering(X,model)

    model = KMeans(n_clusters=2)
    makeCluestering(X,model)

    model = SpectralClustering(n_clusters=2)
    makeCluestering(X,model)

    model = GaussianMixture(n_components=2)
    makeCluestering(X,model)

