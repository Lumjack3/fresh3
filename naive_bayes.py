import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt

def getrid(a):
    return a.replace("'","")

#loading x training values
with open('C:/Thinkful/Naive/ideal_weight.csv') as f:
    reader  = csv.reader(f)
    d= list(reader)

cols = []
#clean column names
for x in d[0]:
    cols.append(x.replace("'",""))

df = pd.DataFrame(d[1:], columns = cols, dtype = int)

df['s2'] = map(getrid, df['sex'])

plt.figure()
plt.hist(df['ideal'], label = 'ideal')
plt.hist(df['actual'], label = 'actual')
plt.legend(loc='upper left')
plt.xlabel("Actual vs. Ideal Weights")

plt.show()

plt.figure()
plt.hist(df['diff'])
plt.xlabel('weight differences distribution')
plt.show()

#categoricalize gender
s= pd.get_dummies(df['sex'])
for x in s.columns:
    df[x.replace("'","")] = s[x]
    
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
data = df[['actual','ideal','diff']]
gnb = GaussianNB()
y_pred = gnb.fit(data, df['Female']).predict(data)

# summarize the fit of the model
#print(metrics.classification_report(df['Female'], y_pred))
#print(metrics.confusion_matrix(df['Female'], y_pred))

print "Number of mislabeled points: %s" % (df['Female'] != y_pred).sum()

#predict sex
df2 = [145,160,-15]
df3 = [160, 145, 15]

y_pred2 = gnb.predict(df2)
y_pred3 = gnb.predict(df3)

print "Predict the sex for an actual weight of 145, an ideal weight of 160, and a diff of -15."
print "The predicted sex is: "
if y_pred2 == 0:
    print "Male"
else:
    print "Female"
    
print "Predict the sex for an actual weight of 160, an ideal weight of 145, and a diff of 15."
print "The predicted sex is: "
if y_pred3 == 0:
    print "Male"
else:
    print "Female"