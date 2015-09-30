import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.datasets import load_iris

#sepal length, sepal width, petal length, petal width
import numpy as np

def majclass(k):
    data = load_iris()
    
    d = data['data']
    dd = pd.DataFrame(d)[[0,1]]

    rnd_item =  int(random.random() * len(dd))
    nbrs = NearestNeighbors(n_neighbors=len(dd)).fit(dd)
    distances, indices = nbrs.kneighbors(dd[rnd_item:rnd_item+1])
    ee = indices.tolist()
    flower = pd.DataFrame(data['target'].tolist())
    dd['flower'] = flower


    nearest_points = ee[0][:10]
    ff = dd.as_matrix()
    gg = []
    for x in nearest_points:
        gg.append(ff[x][2])
    dataz = Counter(gg)
    flower_num =  dataz.most_common(1)[0][0]
    print data['target_names'][flower_num]

majclass(11)
#sepal length, sepal width, petal length, petal width