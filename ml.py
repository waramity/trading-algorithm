import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
plt.style.use('bmh')

# Store the data into a data frame
df = pd.read_csv('aapl.csv')
print(df.head(6))

# Get the number (row) of trading days
print(df.shape)

# Visualize the close price data
# plt.figure(figsize=(16,8))
# plt.title('Apple')
# plt.xlabel('Days')
# plt.ylabel('Close Price USD ($)')
# plt.plot(df['Adj Close'])
# plt.show()

# GEt the Close Price
df = df[['Adj Close']]
print(df.head(4))

# Create a variable to predict the 'x' days out into the future
future_days = 50
# Create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Adj Close']].shift(-future_days)
print(df.head(4))
print(df.tail(4))

# Create the future data set (x) and convert it to a numpy array and remove the last 'x'rows/days
x = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(x)

# Create the target data set (y) and convert it to a numpy array and get all of the target values except the last 'x' rows/days
y = np.array(df['Prediction'])[:-future_days]
print(y)

# Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Create the models
# Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
# Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)

# Get the last 'x' rows of the future data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
print(x_future)

# Show the model tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()

# Show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)


# Change prediction method here VVVV
# predictions = tree_prediction
predictions = tree_prediction

valid = df[x.shape[0]:]
valid['Predictions'] =  predictions

## Create a new data frame to store all the data
data = pd.DataFrame()
print(valid['Predictions'])
print(type(valid['Predictions']))
data['Predictions'] = valid['Predictions']
data['Adj Close'] = df['Adj Close']
data['Buy Price'] = pd.Series([])
data['Sell Price'] = pd.Series([])
print(df['Adj Close'])
print(type(df['Adj Close']))


def buy_sell(data):
    sigPriceBuy = []
    sigPriceSell = []

    flag = -1
    profit = 0
    i = 973
    print('kuy')
    print(data['Predictions'][997])

    while i < 997:
        print(data['Predictions'][i])
        if data['Predictions'][i] < data['Predictions'][i+1]:
            if flag != 1:
                data['Buy Price'][i] = data['Adj Close'][i]
                data['Sell Price'][i] = np.nan
                sigPriceBuy.append(data['Adj Close'][i])
                sigPriceSell.append(np.nan)
                profit -= data['Adj Close'][i]
                flag = 1
            else: 
                data['Buy Price'][i] = np.nan
                data['Sell Price'][i] = np.nan
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif data['Predictions'][i] > data['Predictions'][i+1]:
            if flag != 0:
                data['Buy Price'][i] = np.nan
                data['Sell Price'][i] = data['Adj Close'][i]

                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data['Adj Close'][i])
                profit += data['Adj Close'][i]
                flag = 0
            else:
                data['Buy Price'][i] = np.nan
                data['Sell Price'][i] = np.nan
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            data['Buy Price'][i] = np.nan
            data['Sell Price'][i] = np.nan
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)
        i += 1
    return (sigPriceBuy, sigPriceSell, profit)

buy_sell = buy_sell(data)
print(buy_sell[0])
marker = pd.DataFrame()
marker['Buy_Signal_Price'] = buy_sell[0]
marker['Sell_Signal_Price'] = buy_sell[1]

print("Overall Profit: ", buy_sell[2])

# Visualize data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.scatter(data.index, data['Buy Price'], label = 'Buy', marker = '^', color = 'green')
plt.scatter(data.index, data['Sell Price'], label = 'Sell', marker = 'v', color = 'red')
plt.legend(['Orig', 'Val', 'Pred'])
plt.show()

