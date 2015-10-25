from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import svm
from sklearn import cross_validation

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
x2 = np.hstack((X,pd.DataFrame(y)))
training1, test1 = train_test_split(x2, test_size = 0.4)

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(training1[:,0:4], training1[:,4])
print "Training and Test data split:"
print "Training set points: %s" % str(training1.shape[0])
print "Test set points: %s" % str(test1.shape[0])
print "Score on training data: %s" % str(clf.score(training1[:,0:4], training1[:,4]))
print "Score on test data: %s" % str(clf.score(test1[:,0:4], test1[:,4]))

scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print "5 FOLD TEST"
print "Average score: %s" % scores.mean()
print "Standard Deviation score: %s" % scores.std()

