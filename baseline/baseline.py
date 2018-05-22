import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

dataset_url = 'C:/Users/Fateme/Desktop/ai/contests/fanavard_97/data/A_ticker.csv'
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)

df['a1'].plot()
pyplot.show()

X = []

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


for i in range(1,9) :
    s = 'a' +str(i)
    #print s
    X.append(df[s].values)

l = 0
print len(X)
for i in range (0 ,len(X)):

		for j in range (1,(X[i].size) - 1) :
		    #print j
		    if X[i][j] < 2 :
			    l = l + 1
			    X[i][j] =  ( X[i][j - 1] )



for i in range (0 ,len(X)  ) :
    print i
    pyplot.figure(i)
    pyplot.plot(X[i], label='Value')

pyplot.legend(['data', 'linear', 'cubic'], loc='best')
pyplot.show()

n = len(X)
test_size = int(0.2*n)
print test_size
train, test = X[0:-test_size], X[-test_size:] #12 taye akhar ro migire test

# tmp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print tmp[0:-2]
# print tmp[-2:]

history = [x for x in train]
predictions = list()
nb_correct_predict = 0
for i in range(len(test)):
    # get the history last row as predictions
    predictions.append(history[-1])
    # append the test set to the history
    history.append(test[i])
    # expected price
    expected = history[-1]
    #predicted price
    yhat = predictions[-1]
    #calculate number of correct trend predictions
    if i != 0:
        if (expected > old_expected) and (yhat > old_yhat):
            nb_correct_predict = nb_correct_predict+1
        elif (expected < old_expected) and (yhat < old_yhat):
            nb_correct_predict = nb_correct_predict+1
        elif (expected == old_expected) and (yhat == old_yhat):
            nb_correct_predict = nb_correct_predict+1
    print('Date=%s, Predicted=%.2f, Expected=%.2f' % (df.index[-12+i], yhat, expected))
    old_yhat = yhat
    old_expected = expected
# calculate rmse
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# print correct number of trend predictions

# line plot of observed vs predicted
pyplot.plot(test[-200:], label = 'Expected Value')
pyplot.plot(predictions[-200:], label = 'Predicted Value')
pyplot.legend()
pyplot.show()