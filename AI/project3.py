import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

#for oneHot encoding of output 
from sklearn.preprocessing import OneHotEncoder

# Scale the data
from sklearn.preprocessing import MinMaxScaler
#for spliting data
from sklearn.model_selection import train_test_split

# agglomerative clustering
# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

#for building simple neural network(only a fully connected)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
#calculate  metrics of model performace
from sklearn import metrics  


#get and prepare data
# Set up End and Start times for data grab
end = datetime.now()
# get data of BTC-USD from 14 years ago
start = datetime(end.year - 14, end.month, end.day)

BTC_USD = yf.download("BTC-USD", start, end)#calculate daily return
Open_Candle_from_2014_up_2022=BTC_USD['Open']
Close_Candle_from_2014_up_2022=BTC_USD['Close']
daily_stock_return=100*((Close_Candle_from_2014_up_2022-Open_Candle_from_2014_up_2022)/Open_Candle_from_2014_up_2022)

#convert daily return to a numpy array for passing it to model
daily_stock_return_np_array=daily_stock_return.to_numpy()
X=daily_stock_return_np_array.reshape((-1,1))


#first clusters the data with unsupervised algorithm to 3 class (like project 1),for extracting label of data
# define the model(use AgglomerativeClustering like project 1)
model = AgglomerativeClustering(n_clusters=3)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)

#set X,Y of dataset
Y=yhat.reshape((-1,1))
print('dims of X :'+str(X.shape))
print('dims of Y :'+str(Y.shape))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(X)

#create dataset including: 4 previous value as input and the label of cluster as output
x_ = []
y_ = []

for i in range(4, len(scaled_data)):
    x_.append(scaled_data[i-4:i, 0])
    y_.append(Y[i, 0])

# Convert the x_ and y_ to numpy arrays 
x_, y_ = np.array(x_), np.array(y_)
y_=np.reshape(y_,(-1,1))
#oneHot y
ohe = OneHotEncoder()
y_ = ohe.fit_transform(y_).toarray()

X_train, X_tv, y_train, y_tv = train_test_split(x_, y_, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)

print('shape of X_train: '+str(X_train.shape))
print('shape of y_train: '+str(y_train.shape))
print('shape of X_val: '+str(X_val.shape))
print('shape of y_val: '+str(y_val.shape))
print('shape of X_test: '+str(X_test.shape))
print('shape of y_test: '+str(y_test.shape))

#creat model
# Neural network
model = Sequential()
model.add(Dense(30, input_dim=4, activation='relu'))
model.add(Dense(20, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

#model compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','AUC'])


#use  earlyStopping method for find best epoch
#set earlyStopping callback
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=1
    )

#save  best weight of model
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        'nn.h5',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode="min",
        monitor="val_loss",
    )


history = model.fit(X_train, y_train, epochs=40, batch_size=128,validation_data=(X_val,y_val),callbacks=[earlyStoppingCallback, modelCheckpoint])


y_pred=model.predict(X_test)

threshold=0.5
f1=metrics.f1_score(y_test, y_pred > threshold, average="micro")
acc=metrics.accuracy_score(y_test, y_pred > threshold)
precision=metrics.precision_score(y_test, y_pred > threshold, zero_division=0,average='micro')
recall=metrics.recall_score(y_test, y_pred > threshold, zero_division=0,average='micro')


print('acc of test part is :'+str(acc))
print('f1 of test part is :'+str(f1))
print('precision of test part is :'+str(precision))
print('recall of test part is :'+str(recall))