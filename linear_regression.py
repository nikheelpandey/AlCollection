
# coding: utf-8

# In[36]:

import pandas as pd 
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
##the dataset used was GOOGLE's stock price from quandls. You can get any other data set
#just make sure it is in the right format or you can use your label...whatever, moving on...

df = pd.read_csv('GOOG-NASDAQ_GOOGL-Copy1.csv')

#Custom labeling for easy in use 
df = df[['Open','Close','Low','High','Volume']]


#additon if column to check the percentae of High_Low
df['PCT_HL']= ((df['High']-df['Low'])/df['High'])*100.0


#additon if column to check the percentae of Opeining price and high price
df['PCT_OC']= ((df['Close']-df['Open'])/df['Open'])*100.0


#now I am tring to predict the Closing price of the stock using Linear Regression
forecast_col= 'Close'
df.fillna(-99999, inplace = True)
 
forecast_out= int(math.ceil(0.005*len(df)))

df['label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'],1))
y = np.array(df['label'])


X = preprocessing.scale(X)
y = np.array(df['label'])

#cross validation for automatic shuffling of the data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


#training time :I will use two different algorithms
#classifier 
clf =LinearRegression()  ##you can give the argument (n_jobs= int) for multithreading as scikitlearn(LR) supports multithreading
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)
##0.98812731826 :)  (using simple linear regression)

clf =svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)
##.870378210184  :( (Using suppport vector Regression)





# In[ ]:




# In[ ]:



