import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
import numpy as np

#loading x training values
with open('C:/Thinkful/UCI HAR Dataset/train/X_train.txt') as f:
    reader  = csv.reader(f)
    d= list(reader)
dd =[]

df = pd.DataFrame(d)
for row in d:
    dd.append(row[0].split())

dd2 = pd.DataFrame(dd)


#loading y training values
with open('C:/Thinkful/UCI HAR Dataset/train/y_train.txt') as f:
    reader  = csv.reader(f)
    d= list(reader)


df = pd.DataFrame(d)
dd2['activity'] = df

#loading subject training values
with open('C:/Thinkful/UCI HAR Dataset/train/subject_train.txt') as f:
    reader  = csv.reader(f)
    d= list(reader)


df = pd.DataFrame(d)
dd2['subject'] = df.astype(int)

dd2cols = []
for x in dd2.columns:
    dd2cols.append("x" + str(x))

dd2.columns = dd2cols

dd2test = dd2[dd2['xsubject'] <= 6]
dd2train = dd2[dd2['xsubject'] >= 27]
dd2validateA = dd2[dd2['xsubject'] >=21]
dd2validate = dd2validateA[dd2validateA['xsubject'] < 27]

#random forest classifier
rf = RandomForestClassifier(n_estimators=50, oob_score=True)
cols = dd2test[dd2cols]
colsRes = dd2test['xactivity']
rf.fit(cols, colsRes)
results_test = rf.predict(dd2test[dd2cols])
model_score = rf.oob_score_
print "Model score: %s" % model_score
resultsscoretest = rf.score(dd2test[dd2cols],dd2test['xactivity'])
print "Mean avg accuracy test: %s" % resultsscoretest

results_train = rf.predict(dd2train[dd2cols])
resultsscoretrain = rf.score(dd2train[dd2cols],dd2train['xactivity'])
print "Mean avg accuracy test test datat: %s" % resultsscoretrain

results_validate = rf.predict(dd2validate[dd2cols])
resultsscorevalidate = rf.score(dd2validate[dd2cols],dd2validate['xactivity'])
print "Mean avg accuracy test validation data: %s" % resultsscorevalidate

collist= []

with open('C:/Thinkful/UCI HAR Dataset/features.txt') as f:
    
    for row in f:
        collist.append(row.split())

dfA =pd.DataFrame(collist, columns={"Act","x"})

	#feauture ranking
imps = rf.feature_importances_
ind_imps = np.argsort(imps)[::-1]
print "Top 10 Features:"
for f in range(10):
    print("%d.  %s (%f)" % (f + 1, collist[ind_imps[f]-1][1], imps[ind_imps[f]]))
    
import sklearn.metrics as sklm
import pylab as plt
confusion_results = sklm.confusion_matrix(dd2test['xactivity'],results_test)
plt.figure()
plt.matshow(confusion_results)
plt.title('confusion matrix - test')
plt.colorbar()
plt.show()

plt.figure()
confusion_results = sklm.confusion_matrix(dd2validate['xactivity'],results_validate)
plt.matshow(confusion_results)
plt.title('confusion matrix - validate')
plt.colorbar()
plt.show()

accuracy_train = sklm.accuracy_score(dd2train['xactivity'], results_train)
accuracy_test = sklm.accuracy_score(dd2test['xactivity'], results_test)
accuracy_validate = sklm.accuracy_score(dd2validate['xactivity'], results_validate)

print "Accuracy Score on the train set:  %s" % accuracy_train
print "Accuracy Score on the test set:  %s" % accuracy_test
print "Accuracy Score on the validation: %s" % accuracy_validate

precision_train = sklm.precision_score(dd2train['xactivity'], results_train)
precision_test = sklm.precision_score(dd2test['xactivity'], results_test)
precision_validate = sklm.precision_score(dd2validate['xactivity'], results_validate)

print "Precision Score on the train set:  %s" % precision_train
print "Precision Score on the test set:  %s" % "Precision Score on the train set:  %s" % precision_test
print "Precision Score on the validation set:  %s" % precision_validate

accuracy_train = sklm.recall_score(dd2train['xactivity'], results_train)
accuracy_test = sklm.recall_score(dd2test['xactivity'], results_test)
accuracy_validate = sklm.recall_score(dd2validate['xactivity'], results_validate)

print "Accuracy Score on the train set:  %s" % accuracy_train
print "Accuracy Score on the test set:  %s" % accuracy_test
print "Accuracy Score on the validation set:  %s" % accuracy_validate

f1_train = sklm.f1_score(dd2train['xactivity'], results_train)
f1_test = sklm.f1_score(dd2test['xactivity'], results_test)
f1_validate = sklm.f1_score(dd2validate['xactivity'], results_validate)

print "F1 Score on the train set:  %s" % f1_train
print "F1 Score on the test set:  %s" % f1_test
print "F1 Score on the validation set:  %s" % f1_validate