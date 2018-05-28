import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd

from sklearn.metrics import mean_squared_error
from math import sqrt


def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(80, inplace=True)
    return df


# read data from dataset
dataset_url = '../data/B_ticker.csv'
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)

X = []
for i in range(1,9) :
    s = 'b' +str(i)
    #print s
    X.append(df[s].values)

# remove outliers
for i in range (0 ,len(X)):
        for j in range (1,(X[i].size) - 1) :
            #print j
            if X[i][j] < 2 :
                X[i][j] =  ( X[i][j - 1] )

msk = np.random.rand(len(df['b1'])) < 0.8
err = []
#for step in range(1, 10):
step = 1
#print step

#chaneg time serie to supervise
supervised = timeseries_to_supervised(X[0], step)
supervised_values = supervised.values

#split data to x and y
x = supervised_values[:, :step]
y = supervised_values[:, step]

# set train and test randomly (using featuer a1)
X_train = x[msk]
y_train = y[msk]
X_test = x[~msk]
y_test = y[~msk]
# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train[1:], y_train[1:])
predictions = reg.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, predictions))
print("rmse : \n",rmse)

# # regression coefficients
# print('Coefficients: \n', reg.coef_)
#
# # variance score: 1 means perfect prediction
# print('Variance score: {}'.format(reg.score(X_test, y_test)))
#
# # plot for residual error
#
# # line plot of observed vs predicted


# plt.plot(X_train, y_train,label = 'real data  ')
# plt.show()
# plt.plot(X_train, reg.predict(X_train),label = 'line which is fited')
#
# plt.show()

# plt.plot(y_test, label='Expected Value')
# plt.plot(predictions, label='Predicted Value')
# # pyplot.plot(err, label='rmse')
# plt.legend()
# plt.show()

plt.plot(X_train, y_train, 'o')
#plt.plot(X_train[1:], y_train[1:], 'o')
#plt.legend()
plt.show()