
import pandas as pd 
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style




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


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
#X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)


style.use('ggplot')
df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()  ##This is to update the time and it screws with the graph... suggest solution if you have any.
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



