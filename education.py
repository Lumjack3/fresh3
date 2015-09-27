from bs4 import BeautifulSoup
import statsmodels.api as sm
import requests
import sqlite3 as lite
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import numpy as np
#put all imports at top
#removed unnecessary spaces between lines
#removed unnecessary commented lines
#added comments throughout the code to indicate what's going on

con = lite.connect('getting_started.db')
url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"
r = requests.get(url)
soup = BeautifulSoup(r.content)
#Create and load table with school life expectancy data
with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS school_life") 
    cur.execute("CREATE TABLE school_life (country text, male int, female int, year int)")            
    for row in soup('table')[8].find_all('tr'):
        tds = row.find_all('td')
        if len(tds) == 12:
            sql = "INSERT INTO school_life VALUES (?,?,?,?)" 
            cur.execute(sql, (tds[0].text, tds[7].text, tds[10].text, tds[1].text))
#Query database for male and female statistics
with con:
    cur.execute("SELECT * FROM school_life")
rows = cur.fetchall()
df = pd.DataFrame(data=rows)
df.columns = ['country','male', 'female','year' ]
print "female [Median: %s, Mean: %s]" %( df['female'].median(), df['female'].mean())
print "male [Median: %s, Mean: %s]" %( df['male'].median(), df['male'].mean())
#Load GDP data into database
with con:
    cur.execute("DROP TABLE IF EXISTS gdp") 
    cur.execute("CREATE TABLE gdp (country_name TEXT, _1999 FLOAT, _2000 FLOAT, _2001 FLOAT, _2002 FLOAT, _2003 FLOAT, _2004 FLOAT, _2005 FLOAT, _2006 FLOAT, _2007 FLOAT, _2008 FLOAT, _2009 FLOAT, _2010 FLOAT)")            
with open('c:/Thinkful/Educ/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv','rU') as inputFile:
    next(inputFile) # skip the first 5 lines
    next(inputFile)
    next(inputFile)
    next(inputFile)
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        with con:
            cur.execute('INSERT INTO gdp (country_name, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010) VALUES ("' + line[0] + '","' + '","'.join(line[42:-6]) + '");')
#Query database to compare gdp and male school expentancy
with con:
    cur.execute("SELECT country, _2010, male FROM gdp INNER JOIN school_life ON country_name = country;")
rows = cur.fetchall()
df = pd.DataFrame(rows)
df['gdp2'] = df[1].convert_objects(convert_numeric = True)
df['gdp3'] = np.log(df['gdp2'])
df.dropna(inplace = True)
y = np.matrix(df[2]).transpose()
x = np.matrix(df['gdp3']).transpose()
X= sm.add_constant(x)
model = sm.OLS(y,x)
f=model.fit()
print "The R-Squared between GDP and School Life Expectancy is: %s" % f.rsquared
print "This is a high r-squared indicating a strong connection"
print "The correlation is %s" % np.corrcoef(df['gdp3'], df[2])[0,1]
print "This correlation isn't extremely high but maybe considered very high for economic values."
print "Perhaps with richer economies there is more need for education"
print "On the other hand, high resource countries can buck these trends."