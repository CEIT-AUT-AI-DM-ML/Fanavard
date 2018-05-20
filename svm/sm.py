from sklearn import svm
import pandas as pd
from matplotlib import pyplot
from pandas import DataFrame
from math import sqrt
from sklearn.metrics import mean_squared_error


from pandas import concat

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


dataset_url = 'A_ticker.csv'
def parser(x):
	return int(x) - 1510555672
#df = pd.read_csv(dataset_url).drop('a2',axis = 1).drop('a3',axis = 1).drop('a4',axis = 1).drop('a5',axis = 1).drop('a6',axis = 1)
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)
# summarize first few rows
# line plot
#df = df.drop('a6',axis = 1)
# pyplot.figure(1)
# pyplot.subplot(211)
# df['a1'].plot()
#
# pyplot.subplot(212)
# df['a2'].plot()
#
# pyplot.figure(2)
# pyplot.subplot(211)
# df['a3'].plot()
#
# pyplot.subplot(212)
# df['a4'].plot()
#
# pyplot.figure(3)
# pyplot.subplot(211)
# df['a5'].plot()
#
# pyplot.subplot(212)
# df['a6'].plot()
#
# pyplot.figure(4)
# pyplot.subplot(211)
# df['a7'].plot()
#
# pyplot.subplot(212)
# df['a8'].plot()
#
# pyplot.show()


# X = [data.a7,data.a1],
# y = data.a8

dataset_url = 'C:/Users/Fateme/Desktop/ai/contests/fanavard_97/data/A_ticker.csv'
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)

X = df[['a6', 'a7']].values


pyplot.figure(2)
pyplot.subplot(211)
pyplot.plot(X, label = 'Predicted Value')

l = 0
# for i in range (1 ,(X.size) - 1):
# 		if X[i] == 0 :
# 			l = l + 1
# 			X[i] = X[i-1]
# print l

pyplot.subplot(212)
pyplot.plot(X, label = 'Predicted Value')

supervised = timeseries_to_supervised(X, 1)
supervised_values = supervised.values


X = supervised_values[:,0:1]
Y = supervised_values[:,2]

print supervised

n = X.size
test_size = int(0.2*n)

X_train, X_test = X[0:-test_size], X[-test_size:] #12 taye akhar ro migire test
y_train, y_test = Y[0:-test_size], Y[-test_size:] #12 taye akhar ro migire test

clf = svm.SVC()
print clf.fit(X_train, y_train).score(X_train, y_train)

predictions = clf.predict(X_test)

pyplot.figure(1)
pyplot.plot(y_test,label = 'Expected Value')
pyplot.plot(predictions, label = 'Predicted Value')

rmse = sqrt(mean_squared_error(y_test, predictions))
print('RMSE: %.3f' % rmse)


# pyplot.subplot(211)
# pyplot.plot(y_test,label = 'Expected Value')
#
# pyplot.subplot(212)
# pyplot.plot(predictions, label = 'Predicted Value')

#pyplot.legend()
pyplot.show()

print "09"