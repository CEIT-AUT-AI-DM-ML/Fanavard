from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn import svm
from pandas import DataFrame
from pandas import concat
from sklearn import datasets, linear_model, metrics

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(50, inplace=True)
    return df

class test :

    def __init__(self):
        self.df = []
        self.dataset_url='639'

    def test(self):
        dataset_url = []
        #read data from datasets
        self.dataset_url[0] = '../data/A_ticker.csv'
        self.dataset_url[1] = '../data/B_ticker.csv'
        self.dataset_url[2] = '../data/C_ticker.csv'
        self.dataset_url[3] = '../data/D_ticker.csv'
        self.dataset_url[4] = '../data/E_ticker.csv'
        self.dataset_url[5] = '../data/F_ticker.csv'
        self.dataset_url[6] = '../data/G_ticker.csv'
        self.dataset_url[7] = '../data/H_ticker.csv'
        self.dataset_url[8] = '../data/I_ticker.csv'
        prefix = []
        prefix[0] = 'a'
        prefix[1] = 'b'
        prefix[2] = 'c'
        prefix[3] = 'd'
        prefix[4] = 'e'
        prefix[5] = 'f'
        prefix[6] = 'g'
        prefix[7] = 'h'
        prefix[8] = 'i'
        for i in range(9):
            self.df[i] = pd.read_csv(self.dataset_url[i], header=0, parse_dates=[0], index_col=0, squeeze=True)

        # add features to X
        X = []
        for k in range(9):
            for i in range(1,9) :
                s = prefix[k] +str(i)
                X[k].append(self.df[k][s].values)

        input_data = input("enter data")
        data = []
        data_size = 100
        for k in range(9):

            data[k] = input_data[k*data_size + 1]
            # remove outliers
            for i in range (0 ,len(X)):
                    for j in range (1,(X[k][i].size) - 1) :
                        #print j
                        if X[k][i][j] < 2 :
                            X[k][i][j] =  ( X[k][i][j - 1] )

            step = 1
            #preparing data

            # chaneg time serie to supervise
            supervised = timeseries_to_supervised(X[0], step)
            supervised_values = supervised.values

            # split data to x and y
            x = supervised_values[:, :step]
            y = supervised_values[:, step]

            msk = np.random.rand(len(self.df[k][prefix[k] + '1'])) < 0.8
            # set train and test randomly (using featuer a1)
            X_train = x[msk]
            y_train = y[msk]
            X_test = x[~msk]
            y_test = y[~msk]

            #linear model
            reg = linear_model.LinearRegression()
            reg.fit(X_train, y_train)
            err = sqrt(mean_squared_error(y_test, reg.predict(X_test)))
            model = reg

            #svm model
            C = 1.0
            clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)  # svm.SVC(kernel='linear', C=C)
            clf.fit(X_train, y_train)
            tmp_err = sqrt(mean_squared_error(y_test, clf.predict(X_test)))

            if tmp_err < err:
                model = clf

            pdata = data[k]

            print 'for ', prefix[k]
            for i in range(5):
               pred =  model.predict(pdata)
               pdata = pred
               print pred

if __name__ == "__main__":
    tst = test()
    tst.test()
