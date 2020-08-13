import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('fivethirtyeight')

AAPL = pd.read_csv('./aapl.csv')
print(AAPL)

# Create the simple moving average with a 30 day window
SMA30 = pd.DataFrame()
SMA30['Adj Close'] = AAPL['Adj Close'].rolling(window=3).mean()
print(SMA30)

# Create a simple moving 100 day average
SMA100 = pd.DataFrame()
SMA100['Adj Close'] = AAPL['Adj Close'].rolling(window=15).mean()
print(SMA100)

#Visualize the data
# plt.figure(figsize=(12.5, 4.5))
# plt.plot(AAPL['Adj Close'], label = 'AAPL')
# plt.plot(SMA30['Adj Close'], label='SMA30')
# plt.plot(SMA100['Adj Close'], label='SMA100')
# plt.title('Apple Adj. Close Price History')
# plt.xlabel('Oct. 02, 2006 - Dec. 30, 2011')
# plt.ylabel('Adj. Close Price ($)')
# plt.legend(loc='upper left')

# Create a new data frame to store all the data
data = pd.DataFrame()
data['AAPL'] = AAPL['Adj Close']
data['SMA30'] = SMA30['Adj Close']
data['SMA100'] = SMA100['Adj Close']
print(data)

# Create a function to signal to buy and sell the asset/stock
def buy_sell(data):
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1
    profit = 0

    for i in range(len(data)):
        if data['SMA30'][i] > data['SMA100'][i]:
            if flag != 1:
                sigPriceBuy.append(data['AAPL'][i])
                sigPriceSell.append(np.nan)
                if i < 300:
                    profit -= data['AAPL'][i] 
                flag = 1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif data['SMA30'][i] < data['SMA100'][i]:
            if flag != 0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data['AAPL'][i])
                if i < 300:
                    profit += data['AAPL'][i]
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)
    return (sigPriceBuy, sigPriceSell, profit)   

# Store the buy and sell data into a variable
buy_sell = buy_sell(data)
data['Buy_Signal_Price'] = buy_sell[0]
data['Sell_Signal_Price'] = buy_sell[1]

marker = pd.DataFrame()
marker['Buy'] = data['Buy_Signal_Price'][:100]
print(marker['Buy'])
print(data)
print(type(data))

# Show the data
fig = plt.figure(figsize=(12.6, 4.6))
plt.plot(data['AAPL'], label = 'AAPL', alpha = 0.35)
plt.plot(data['SMA30'], label = 'SMA30', alpha = 0.35)
plt.plot(data['SMA100'], label = 'SMA100', alpha = 0.35)
print(type(data.index))
plt.scatter(data.index, data['Buy_Signal_Price'], label = 'Buy', marker = '^', color = 'green')
plt.scatter(data.index, data['Sell_Signal_Price'], label = ' Sell', marker = 'v', color = 'red')
plt.title('Apple Adj. CLose Price History Buy & Sell Signals')
plt.xlabel('Oct. 02, 2006 - Dec. 30, 2011')
plt.ylabel('Adj. Close Price USD ($)')
plt.legend(loc='upper right')


print("Overall Profit: ", buy_sell[2])
# show plt
plt.show()

