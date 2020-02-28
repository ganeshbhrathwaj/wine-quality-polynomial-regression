import numpy as np
import pandas as pd

ds=pd.read_csv('wine quality polynomial reg.csv')
x=ds.iloc[:,0:11].values
y=ds.iloc[:,-1].values



from sklearn.linear_model  import LinearRegression
r=LinearRegression()
r.fit(x,y)

ypred=r.predict(x)


from sklearn.preprocessing import PolynomialFeatures
preg=PolynomialFeatures(degree=4)
xpoly=preg.fit_transform(x)
reg1=LinearRegression()
reg1.fit(xpoly,y)

ypred1=reg1.predict(xpoly)
a=0
a1=0
for i in range(0,1599):
    a+=abs(y[i]-ypred[i])
a=a/1599

for i in range(0,1599):
    a1+=abs(y[i]-ypred1[i])
a1=a1/1599
    