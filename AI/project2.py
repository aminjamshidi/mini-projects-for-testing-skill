import pandas as pd
import json
from urllib.request import (
    urlopen,
)

# For reading stock data from yahoo
import yfinance as yf

# For time stamps
from datetime import datetime

# for calculating correlation with lag
import statsmodels.api as sm


# an other way for getting dataset

# Pulling Bitcoin's price history
btc_url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&api_key=d1875a3943f6f2ee83a90ac2e05d5fa018618e00724e9018f9bd08c0ac932cc6"
btc_data = urlopen(btc_url).read()  # Open the API contents

# Transform the contents of our response into a manageable JSON format
btc_json = json.loads(btc_data)


##Transform Bitcoin data so we can run analysis
btc_price = btc_json["Data"][
    "Data"
]  ##Extract only the relevant data from the JSON variable we created earlier
btc_df = pd.DataFrame(
    btc_price
)  ##Convert the json format into a Pandas dataframe so we can make it easier to work with
btc_df["btc_returns"] = (
    (btc_df["close"] / btc_df["open"]) - 1
) * 100  # We create a coloumn for daily returns of Bitcoin that we'll need for later when we calculate the correlation.
btc_df["Date"] = btc_df["time"].apply(
    lambda x: datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")
)  # Formatting the date into a human-readable format
btc_returns = btc_df[["Date", "btc_returns"]]


##Pulling Oil's price history
oil = yf.Ticker("oil")
oil_df = oil.history(period="max")

##Transform Oil data so we can run analysis
oil_df = (
    oil_df.reset_index()
)  # In the original dataframe, the date is part of the index which means we can't select it later. reset_index shifts the date into a normal column
oil_df["Date"] = oil_df["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
oil_df["oil_returns"] = ((oil_df["Close"] / oil_df["Open"]) - 1) * 100
oil_returns = oil_df[["Date", "oil_returns"]]


##Pulling gold's price history
gold = yf.Ticker("GC=F")
gold_df = gold.history(period="max")

##Transform gold data so we can run analysis
gold_df = gold_df.reset_index()
gold_df["Date"] = gold_df["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
gold_df["gold_returns"] = ((gold_df["Close"] / gold_df["Open"]) - 1) * 100
gold_returns = gold_df[["Date", "gold_returns"]]


# make both dataset a unique size
joint_df_btc_oil = pd.merge(btc_returns, oil_returns)
joint_df_btc_gold = pd.merge(btc_returns, gold_returns)

# convert them to numpy array format
# for Oil
btc_oil_returns_array = joint_df_btc_oil[["btc_returns"]].to_numpy()
oil_btc_returns_array = joint_df_btc_oil[["oil_returns"]].to_numpy()

# for gold
btc_gold_returns_array = joint_df_btc_gold[["btc_returns"]].to_numpy()
gold_btc_returns_array = joint_df_btc_gold[["gold_returns"]].to_numpy()


# calculate correlation
# BTC-USD /Oil

# positive lag
correlation_btc_oil = sm.tsa.stattools.ccf(
    btc_oil_returns_array[:, 0], oil_btc_returns_array[:, 0]
)
# negative lag
correlation_oli_btc = sm.tsa.stattools.ccf(
    oil_btc_returns_array[:, 0], btc_oil_returns_array[:, 0]
)

print("correlation between BTC-USD/OIL:")
print("ten positive lag ,from 0 until 9 is : " + str(correlation_btc_oil[0:10]))
print('ten negativ lag ,from 0 until 9 is : "' + str(correlation_oli_btc[0:10]))

# BTC-USD /Gold

# positive lag
correlation_btc_gold = sm.tsa.stattools.ccf(
    btc_gold_returns_array[:, 0], gold_btc_returns_array[:, 0]
)
# negative lag
correlation_gold_btc = sm.tsa.stattools.ccf(
    gold_btc_returns_array[:, 0], btc_gold_returns_array[:, 0]
)

print("correlation between BTC-USD/GOLD:")
print("ten positive lag ,from 0 until 9 is : " + str(correlation_btc_gold[0:10]))
print('ten negativ lag ,from 0 until 9 is : "' + str(correlation_gold_btc[0:10]))
