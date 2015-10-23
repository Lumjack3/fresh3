
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.lda import LDA
import pandas as pd
from pylab import plot,show
from scipy.cluster.vq import kmeans,vq
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()


KM = kmeans(X_r2, 3)   
centroids,_ = kmeans(X_r2, 3)

idx,_ = vq(X_r2,centroids)
idx2 = pd.DataFrame(idx)
x3 = np.hstack((X_r2, idx2))


plt.figure()
for c, i in zip("rgb", [0,1,2]):
    plt.scatter(x3[idx == i,0], x3[idx == i,1], c=c)
plt.title('Kmeans of IRIS dataset')
plt.show()

print idx - y
print "As you can see... there is not uniformity across the first 50 points, second 50 points, and the third 50 points"
print "This indicates some different predictions by the kmeans clustering method."