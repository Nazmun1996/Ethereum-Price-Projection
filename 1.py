import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle 

style.use('ggplot')

df = quandl.get("BITFINEX/ETHUSD")
df['HL_PCT'] = (df['High']-df['Low'])/df['Low'] * 100.0
df['Change_PCT'] = (df['High']-df['Last'])/df['Last'] * 100.0

df = df[['Last', 'High', 'Low', 'HL_PCT', 'Change_PCT', 'Volume']]
#print(df.tail())

forecast = 'Last'

df.fillna(0000, inplace=True)

fc_out =  int(math.ceil(0.0178*len(df)))
print(fc_out)
df['label'] = df[forecast].shift(-fc_out)
#print(df.head())
#print(df.tail())
X = np.array(df.drop(['label', 'Volume'],1)) #label is output, so should be in y
X = preprocessing.scale(X)
X_late = X[-fc_out:]
X = X[:-fc_out]


df.dropna(inplace=True)

y = np.array(df['label'])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)
classifier = LinearRegression()
classifier.fit(X_train, y_train)
#with open ('linear_regression.pickle', 'wb') as f:
#    pickle.dump(classifier, f)
#pickle_in = open('linear_regression.pickle', 'rb')
#classifier = pickle.load(pickle_in)


accuracy = classifier.score(X_test,y_test)
accuracy = accuracy*100
forecast_set = classifier.predict(X_late)
with open('out.txt', 'wb') as f:
    print(forecast_set, fc_out)
    print(accuracy)
    f.write(forecast_set)
    
    
df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + 86400

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Last'].plot()
df['forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

