from itsdangerous import BadTimeSignature
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

# For time stamps
from datetime import datetime


# Set up End and Start times for data grab
end = datetime.now()
# get data of BTC-USD from 14 years ago
start = datetime(end.year - 14, end.month, end.day)

BTC_USD = yf.download("BTC-USD", start, end)
# show data
print(BTC_USD)

# calculate daily return
Open_Candle_from_2014_up_2022 = BTC_USD["Open"]
Close_Candle_from_2014_up_2022 = BTC_USD["Close"]
daily_stock_return = 100 * (
    (Close_Candle_from_2014_up_2022 - Open_Candle_from_2014_up_2022)
    / Open_Candle_from_2014_up_2022
)

# prepare data as X vector
daily_stock_return_np_array = daily_stock_return.to_numpy()
data_length = np.arange(daily_stock_return_np_array.shape[0])
X = daily_stock_return_np_array.reshape((-1, 1))

# define the clustering model
model = AgglomerativeClustering(n_clusters=3)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)


x_y = np.array([daily_stock_return_np_array, data_length])

# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    plt.scatter((x_y).T[row_ix, 1], (x_y).T[row_ix, 0])
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("daily return rate (%)", fontsize=16)

# show the plot
plt.show()
