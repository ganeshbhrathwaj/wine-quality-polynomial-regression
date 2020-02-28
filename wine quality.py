import numpy as np
import pandas as pd

ds=pd.read_csv('wine quality polynomial reg.csv')
x=ds.iloc[:,0:11].values
y=ds.iloc[:,-1].values


#prediction with linear regression
from sklearn.linear_model  import LinearRegression
r=LinearRegression()
r.fit(x,y)

ypred=r.predict(x)

#prediction with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
preg=PolynomialFeatures(degree=4)
xpoly=preg.fit_transform(x)
reg1=LinearRegression()
reg1.fit(xpoly,y)
ypred1=reg1.predict(xpoly)

    
