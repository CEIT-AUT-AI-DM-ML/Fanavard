from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn import svm
from pandas import DataFrame
from pandas import concat

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
class SvmModel :


    def __init__(self):
        self.df = []
        self.dataset_url=''

    def smodel(self):

        #read data from dataset
        self.dataset_url = '../data/A_ticker.csv'
        self.df = pd.read_csv(self.dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)

        # add features to X
        X = []
        for i in range(1,9) :
            s = 'a' +str(i)
            #print s
            X.append(self.df[s].values)

        # remove outliers
        for i in range (0 ,len(X)):
                for j in range (1,(X[i].size) - 1) :
                    #print j
                    if X[i][j] < 2 :
                        X[i][j] =  ( X[i][j - 1] )

        d = self.df[['a1', 'a6']]

        msk = np.random.rand(len(self.df['a1'])) < 0.8
        err = []

        #for step in range(1, 10):
        step = 1
        numbeOfFetures = 2
        #print step
        supervised = timeseries_to_supervised(d, step)
        print supervised
        supervised_values = supervised.values

        x = supervised_values[:, :step*numbeOfFetures]
        y = supervised_values[:, step*numbeOfFetures]

        print x
        print y

        # set train and test randomly (using featuer a1)

        x_train = x[msk]
        y_train = y[msk]
        x_test = x[~msk]
        y_test = y[~msk]

        # we create an instance of SVM and fit out data. We do not scale our
        # data since we want to plot the support vectors
        C = 1.0

        # models = (svm.SVC(kernel='linear', C=C),
        #           svm.LinearSVC(C=C),
        #           svm.SVC(kernel='rbf', gamma=0.7, C=C),
        #           svm.SVC(kernel='poly', degree=3, C=C))
        # models = (clf.fit(x_train, y_train) for clf in models)
        #
        # i = 0
        # for m in models:
        #     print i
        #     i+=1
        #     p = m.predict(x_test)
        #     rmse = sqrt(mean_squared_error(y_test, p))
        #     print('RMSE: %.3f' % rmse)

        #print "finish"
        clf = svm.SVC(kernel='rbf', gamma=0.7, C=C) #svm.SVC(kernel='linear', C=C)
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)
        rmse = sqrt(mean_squared_error(y_test, predictions))
        err.append(rmse)
        print rmse


        #line plot of observed vs predicted
        pyplot.plot(y_test[-100:], label='Expected Value')
        pyplot.plot(predictions[-100:], label='Predicted Value')
        #pyplot.plot(err, label='rmse')
        pyplot.legend()
        pyplot.show()

if __name__ == "__main__":
    model = SvmModel()
    model.smodel()