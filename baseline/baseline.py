from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from pandas import DataFrame
from pandas import concat

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(77, inplace=True)
    return df
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

class baseLineModel :

    def __init__(self):
        self.df = []
        self.dataset_url=''

    def baseLine(self):

        #read data from dataset
        self.dataset_url = '../data/B_ticker.csv'
        self.df = pd.read_csv(self.dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)

        # add features to X
        X = []
        for i in range(1,9) :
            s = 'b' +str(i)
            #print s
            X.append(self.df[s].values)

        # remove outliers
        for i in range (0 ,len(X)):
                for j in range (1,(X[i].size) - 1) :
                    #print j
                    if int(X[i][j]) < 2 :
                        X[i][j] =  ( X[i][j - 1] )

        # set train and test randomly (using featuer a1)
        msk = np.random.rand(len(self.df['b1'])) < 0.8
        # print msk
        test = X[0][~msk]
        # # smooth data
        # for i in range(0, len(X)):
        #     for j in range(1, (X[i].size) - 1):
        #         X[i][j] = (X[i][j - 1] + X[i][j] + X[i][j + 1]) / 3

        train = X[0][msk]

        supervised = difference(timeseries_to_supervised(X[0], 1))
        supervised_values = supervised.values

        x = supervised_values[:, :1]
        y = supervised_values[:, 1]

        x_train = x[msk]
        y_train = y[msk]



        pyplot.plot(x_train,y_train,'o')
        pyplot.show()
        # plot data after removing outliers
        # for i in range (0 ,len(X)  ) :
        #     # print (i)
        #     pyplot.figure(i)
        #     pyplot.plot(X[i], label='Value')
        # pyplot.show()




        # print (train)
        # print(len(train))
        # print (test)
        # print(len(test))


        # baseline model
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
            # print('Date=%s, Predicted=%.2f, Expected=%.2f' % (self.df.index[-12+i], yhat, expected))
            old_yhat = yhat
            old_expected = expected

        # calculate rmse
        rmse = sqrt(mean_squared_error(test, predictions))
        print('RMSE: %.3f' % rmse)

        # line plot of observed vs predicted
        pyplot.plot(test[-100:], label = 'Expected Value')
        pyplot.plot(predictions[-100:], label = 'Predicted Value')
        pyplot.xlabel('time', fontsize=18)
        pyplot.ylabel('b1', fontsize=16)
        pyplot.legend()
        pyplot.show()

        pyplot.plot(train);
        pyplot.show()

if __name__ == "__main__":
    blm = baseLineModel()
    blm.baseLine()
