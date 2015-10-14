import pandas as pd
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv

#load un data
df = pd.read_csv('C:\Thinkful\cluster\un.csv')
print 'rows: ' + str(df.shape[0])

#screen out necessary columns
df2 = df[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]
df2.dropna(inplace = True)
print 'countries used: %s' % df2.shape[0]
df3 = df2.as_matrix()

#gnerate within cluster sum of squares
klist = range(1,11)
KM = [ scv.kmeans(df2.values, k) for k in klist ]    
avwcss = [ var for (cent, var) in KM ]

plt.figure()
plt.scatter(klist, avwcss)
plt.xlabel('k clusters')
plt.ylabel('within cluster sum of squares')
plt.show()

#cluster in 3
centroids,_ = kmeans(df3, 3)

idx,_ = vq(df3,centroids)
df2['cen'] = idx

#GDP vs infant mortality
x1 = df2['GDPperCapita'][df2['cen'] == 0]
y1 = df2['infantMortality'][df2['cen'] == 0]
x2 = df2['GDPperCapita'][df2['cen'] == 1]
y2 = df2['infantMortality'][df2['cen'] == 1]
x3 = df2['GDPperCapita'][df2['cen'] == 2]
y3 = df2['infantMortality'][df2['cen'] == 2]


plt.figure()
plt.scatter(x1, y1, color = ['red'])
plt.scatter(x2, y2, color = ['blue'])
plt.scatter(x3, y3, color = ['green'])
plt.xlabel('GDPperCapita')
plt.ylabel('infant mortality')
plt.title('GDP vs Infant Mortality')
plt.show()

#GDP vs male life expectancy
x1 = df2['GDPperCapita'][df2['cen'] == 0]
y1 = df2['lifeMale'][df2['cen'] == 0]
x2 = df2['GDPperCapita'][df2['cen'] == 1]
y2 = df2['lifeMale'][df2['cen'] == 1]
x3 = df2['GDPperCapita'][df2['cen'] == 2]
y3 = df2['lifeMale'][df2['cen'] == 2]


plt.figure()
plt.scatter(x1, y1, color = ['red'])
plt.scatter(x2, y2, color = ['blue'])
plt.scatter(x3, y3, color = ['green'])
plt.xlabel('GDPperCapita')
plt.ylabel('male life expectancy')
plt.title('GDP vs Male Life Expectancy')
plt.show()

#GDP vs female life expectancy
x1 = df2['GDPperCapita'][df2['cen'] == 0]
y1 = df2['lifeFemale'][df2['cen'] == 0]
x2 = df2['GDPperCapita'][df2['cen'] == 1]
y2 = df2['lifeFemale'][df2['cen'] == 1]
x3 = df2['GDPperCapita'][df2['cen'] == 2]
y3 = df2['lifeFemale'][df2['cen'] == 2]


plt.figure()
plt.scatter(x1, y1, color = ['red'])
plt.scatter(x2, y2, color = ['blue'])
plt.scatter(x3, y3, color = ['green'])
plt.xlabel('GDPperCapita')
plt.ylabel('female life expectancy')
plt.title('GDP vs Female Life Expectancy')
plt.show()
