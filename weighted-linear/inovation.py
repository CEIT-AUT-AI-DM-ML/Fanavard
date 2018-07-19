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
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

def setAdd(s, x):
    if x not in s:
        s[x] = 1
    else:
        s[x] += 1

# read data from dataset
dataset_url = '../data/A_ticker.csv'
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)
prefix = 'a'
X = []
step = 1
for i in range(1,9) :
    s = prefix +str(i)
    #print s
    X.append(df[s].values)

# remove outliers
for i in range (0 ,len(X)):
        for j in range (1,(X[i].size) - 1) :
            #print j
            if X[i][j] < 2 :
                X[i][j] =  ( X[i][j - 1] )
            #X[i][j] =X[i][j] -  X[i][j - 1]


err = []
#for step in range(1, 10):

#print step


#chaneg time serie to supervise
supervised =   timeseries_to_supervised(X[0], step)
supervised_values = supervised.values

#split data to x and y
x = supervised_values[:, :step]
y = supervised_values[:, step]


# set train and test randomly (using featuer a1)
for r in range(5):
    print r
    x = supervised_values[:, :step]
    y = supervised_values[:, step]
    msk = np.random.rand(len(df[prefix+'1'])) < 0.8
    X_train = x[msk]
    y_train = y[msk]
    X_test = x[~msk]
    y_test = y[~msk]
    #
    # plt.plot(x, y, 'x')
    # plt.show()

    values = {}
    numbers = {}
    for i in range(len(X_train)):

        numbers[int(X_train[i])] = 0
        values[int(X_train[i])] = 0#{}

    for i in range(len(X_train)):
        numbers[int(X_train[i])] += 1
        values[int(X_train[i])] += y_train[i]
        #setAdd(values[int(X_train[i])], int(y_train[i]))


    # print values
    # print numbers

    avg = {}
    for x in values:
        #print x,values[x],numbers[x]
        avg[x] = float(values[x]) / float(numbers[x])
        #print len(values[x])

    #print values

    # print avg

    pred = []
    for x in X_test:
        #print avg[int(x)]
        pred.append(avg[int(x)])

    # plt.plot(X_test, y_test,'x', label = 'Expected Value')
    # plt.plot(X_test, pred,'o', label = 'predicted Value')
    # plt.xlabel('last time step price', fontsize=18)
    # plt.ylabel('price', fontsize=16)
    # plt.show()

    rmse = sqrt(mean_squared_error(pred, y_test))
    #print rmse
    err.append(rmse)
    # print rmse
print float(sum(err)) / float(len(err))



