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

# read data from dataset
dataset_url = '../data/I_ticker.csv'
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)
prefix = 'i'
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

msk = np.random.rand(len(df[prefix+'1'])) < 0.8
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
X_train = x[msk]
y_train = y[msk]
X_test = x[~msk]
y_test = y[~msk]

# create linear regression object
reg = linear_model.LinearRegression()
reg1 = linear_model.LinearRegression()
reg2 = linear_model.LinearRegression()
reg3 = linear_model.LinearRegression()


l = len(X_train)
l1 = l/3
l2 = (2*l)/3
# train the model using the training sets
reg1.fit(X_train, y_train)
print sqrt(mean_squared_error(y_test, reg1.predict(X_test)))

pin = x[len(x)-1]
print type(pin)
for j in range(step - 1):
    pin[j] = pin[j + 1]
pin[step-1] = y[len(y) - 1]

pout = []
px = []

for i in range(10):
    pred = reg1.predict(pin.reshape(1, -1))
    pout.append(pred)
    print pred
    for j in range(step-1):
        pin[j] = pin[j+1]
    pin[step-1] = pred


#
# plt.plot(x, y, 'x')
# plt.plot(pout[:-1],pout[1:], 'o')
# reg2.fit(X_train[l1:l2], y_train[l1:l2])
# reg3.fit(X_train[l2:], y_train[l2:])
# reg.fit(X_train[1:], y_train[1:])

#rmse = sqrt(mean_squared_error(y_test, predictions))
#print("rmse : \n",rmse)


# plt.plot(X_train, y_train,label = 'real data  ')
# plt.show()
# plt.plot(X_train, reg.predict(X_train),label = 'line which is fited')
#
# plt.show()


# plt.plot(y_test, label='Expected Value')
# plt.plot(predictions, label='Predicted Value')
# plt.legend()
# plt.show()



# plt.plot(X_train[1:l1], y_train[1:l1], 'o')
# plt.plot(X_train[l1:l2], y_train[l1:l2], 'x')
# plt.plot(X_train[l2:], y_train[l2:], 'y')

# print tl1, tl2
#
# plt.plot(X_test[1:tl1], y_test[1:tl1], 'x')
# plt.plot(X_test[1:tl1], reg1.predict(X_test[1:tl1]), 'o')
# print sqrt(mean_squared_error(y_test[1:tl1], reg1.predict(X_test[1:tl1])))
# #plt.plot(X_train[1:], y_train[1:], 'o')
# #plt.legend()
# plt.show()
#
# plt.plot(X_test[tl1:tl2], y_test[tl1:tl2], 'x')
# plt.plot(X_test[tl1:tl2], reg1.predict(X_test[tl1:tl2]), 'o')
# print sqrt(mean_squared_error(y_test[tl1:tl2], reg2.predict(X_test[tl1:tl2])))
# #plt.plot(X_train[1:], y_train[1:], 'o')
# #plt.legend()
# plt.show()
#
# plt.plot(X_test[tl2:], y_test[tl2:], 'x')
# plt.plot(X_test[tl2:], reg1.predict(X_test[tl2:]), 'o')
# print sqrt(mean_squared_error(y_test[tl1:tl2], reg3.predict(X_test[tl1:tl2])))
# plt.show()
#
# plt.plot(X_test, y_test, 'x')
# plt.plot(X_test, reg1.predict(X_test), 'o')
# print sqrt(mean_squared_error(y_test, reg2.predict(X_test)))
l = len(X_train)
l1 = l/2

# fig = plt.figure(1)
# # pyplot.xticks(np.arange(10), np.arange(1, 11))
# fig.suptitle('price based on last time step price', fontsize=20)
# plt.plot(X_train, y_train, 'x')
# plt.xlabel('last time step price', fontsize=18)
# plt.ylabel('price', fontsize=18)
# plt.plot(X_train[l1:], y_train[l1:], 'o')
#plt.plot(X_train, reg.predict(X_train), 'o')
plt.show()