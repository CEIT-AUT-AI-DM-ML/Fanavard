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

class pred :

    def __init__(self):
        self.df = []
        self.dataset_url= []

    def pred(self):

        #read data from datasets
        self.dataset_url.append('../data/A_ticker.csv')
        self.dataset_url.append('../data/B_ticker.csv')
        self.dataset_url.append('../data/C_ticker.csv')
        self.dataset_url.append('../data/D_ticker.csv')
        self.dataset_url.append('../data/E_ticker.csv')
        self.dataset_url.append('../data/F_ticker.csv')
        self.dataset_url.append('../data/G_ticker.csv')
        self.dataset_url.append('../data/H_ticker.csv')
        self.dataset_url.append('../data/I_ticker.csv')
        prefix = []
        prefix.append('a')
        prefix.append('b')
        prefix.append('c')
        prefix.append('d')
        prefix.append('e')
        prefix.append('f')
        prefix.append('g')
        prefix.append('h')
        prefix.append('i')


        for i in range(9):

            self.df.append(pd.read_csv(self.dataset_url[i], header=0, parse_dates=[0], index_col=0, squeeze=True))

        # add features to X
        X = []
        for k in range(9):
            X.append(self.df[k][prefix[k]+'1'].values)


        #input_data = pd.read_csv('input.csv', parse_dates=[0], index_col=0, squeeze=True)
        text_file = open('input.txt', "r")
        lines = text_file.read().split(',')
        msize = 638

        x = int(lines[0])
        for t in range(x):
            data = []

            for k in range(9):

               adr = t*(msize)*9 + k*msize + 2
               print lines[adr]
               data.append(int(lines[adr]))

            print data
            for k in range(9):

                #data[k] = input_data[k*data_size + 1]
                # remove outliers
                for i in range (0 ,len(X[k])):
                    if X[k][i] < 2:
                        X[k][i] = (X[k][i-1])


                step = 1
                #preparing data

                # chaneg time serie to supervise
                supervised = timeseries_to_supervised(X[k], step)
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
                    #print tmp_err
                    model = clf


                pdata = np.array(data[k])

                print 'for ', prefix[k]
                for i in range(5):
                   pred =  model.predict(pdata.reshape(1, -1))
                   pdata = pred
                   print pred

if __name__ == "__main__":
    tst = pred()
    tst.pred()
