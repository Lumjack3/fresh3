
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt

from sklearn import svm
import numpy as np
svc = svm.SVC(kernel='linear', C=1)
from sklearn import datasets
import pandas as pd
from matplotlib.colors import ListedColormap

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.title(iris['feature_names'][a] + " vs. " +iris['feature_names'][b]  )
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
i2data = pd.DataFrame(iris.data)
i2targ = pd.DataFrame(iris.target)
i2data['target']= i2targ
#2 flower svc
for c in range(0,3):
    for d in range(0,3):
        for a in range (0,4):
            for b in range (0,4):
                if b > a and d > c:
                    x1 = iris.data[(c*50):((c+1)*50),[a,b]]
                    x2 = iris.data[(d*50):((d+1)*50),[a,b]]
                    X = np.vstack([x1,x2])
                    y1 = iris.target[(c*50):((c+1)*50)]
                    y2 = iris.target[(d*50):((d+1)*50)]
          
                    y = np.vstack([y1,y2])
                    svc.fit(X, y.ravel())
                    plot_estimator(svc, X, y.ravel())
#3 flower svc
for a in range (0,4):
    for b in range (0,4):
        if b > a:
            X = iris.data[:,[a,b]]
            y = iris.target[:]
            svc.fit(X, y)
            plot_estimator(svc, X, y)
