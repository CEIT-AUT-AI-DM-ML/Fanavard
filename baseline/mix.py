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

class baseLineModel :

    def __init__(self):
        self.df = []
        self.dataset_url=''

    def baseLine(self):

        #read data from dataset
        self.dataset_url = '../data/B_ticker.csv'
        prefix = 'b'
        self.df = pd.read_csv(self.dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)
        print 'step, model, mean, variance'
        # add features to X
        X = []
        for i in range(1,9) :
            s = prefix +str(i)
            #print s
            X.append(self.df[s].values)

        # remove outliers
        for i in range (0 ,len(X)):
                for j in range (1,(X[i].size) - 1) :
                    #print j
                    if X[i][j] < 2 :
                        X[i][j] =  ( X[i][j - 1] )


        # for k in range(20):
        #     print k
        #     # set train and test randomly (using featuer a1)
        #     msk = np.random.rand(len(self.df['a1'])) < 0.8
        #     train = X[0][msk]
        #     test = X[0][~msk]
        #
        #
        #     # baseline model
        #     history = [x for x in train]
        #     predictions = list()
        #     nb_correct_predict = 0
        #     for i in range(len(test)):
        #         # get the history last row as predictions
        #         predictions.append(history[-1])
        #         # append the test set to the history
        #         history.append(test[i])
        #         # expected price
        #         expected = history[-1]
        #         #predicted price
        #         yhat = predictions[-1]
        #
        #         #calculate number of correct trend predictions
        #         if i != 0:
        #             if (expected > old_expected) and (yhat > old_yhat):
        #                 nb_correct_predict = nb_correct_predict+1
        #             elif (expected < old_expected) and (yhat < old_yhat):
        #                 nb_correct_predict = nb_correct_predict+1
        #             elif (expected == old_expected) and (yhat == old_yhat):
        #                 nb_correct_predict = nb_correct_predict+1
        #         # print('Date=%s, Predicted=%.2f, Expected=%.2f' % (self.df.index[-12+i], yhat, expected))
        #         old_yhat = yhat
        #         old_expected = expected
        #
        #     # calculate rmse
        #     rmse = sqrt(mean_squared_error(test, predictions))
        #     baselineErr.append(rmse)

        baselineErr = []
        svmErr = []
        linearSvmErr = []
        linearErr = []

        avg_baselineErr = []
        avg_svmErr = []
        avg_linearSvmErr = []
        avg_linearErr = []

        var_baselineErr = []
        var_svmErr = []
        var_linearSvmErr = []
        var_linearErr = []

        for step in range(1, 6):
            baselineErr = []
            svmErr = []
            linearSvmErr = []
            linearErr = []
            for i in range(5):

                supervised = timeseries_to_supervised(X[0], step)
                supervised_values = supervised.values

                x = supervised_values[:, :step]
                y = supervised_values[:, step]

                # set train and test randomly (using featuer a1)
                msk = np.random.rand(len(self.df[prefix+'1'])) < 0.8
                x_train = x[msk]
                y_train = y[msk]
                x_test = x[~msk]
                y_test = y[~msk]

                # we create an instance of SVM and fit out data. We do not scale our
                # data since we want to plot the support vectors
                C = 1.0
                clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)  # svm.SVC(kernel='linear', C=C)
                clf.fit(x_train, y_train)
                # pyplot.plot(x_train, y_train, 'o')
                # pyplot.plot(x_train, clf.predict(x_train), 'x')
                predictions = clf.predict(x_test)
                rmse = sqrt(mean_squared_error(y_test, predictions))
                svmErr.append(rmse)


                # clfLinear = svm.SVC(kernel='linear', C=C)
                # clfLinear.fit(x_train, y_train)
                # predictions = clfLinear.predict(x_test)
                # rmse = sqrt(mean_squared_error(y_test, predictions))
                # linearSvmErr.append(rmse)

                # create linear regression object
                reg = linear_model.LinearRegression()
                # train the model using the training sets
                reg.fit(x_train, y_train)
                predictions = reg.predict(x_test)
                # pyplot.plot(x_train, reg.predict(x_train), 'y')
                # pyplot.show()
                rmse = sqrt(mean_squared_error(y_test, predictions))
                linearErr.append(rmse)

                #normal
                predictions = []
                for i in range(step, len(x_test)):
                    predictions.append(np.mean(x_test[i-step: i]))
                rmse = sqrt(mean_squared_error(y_test[step:], predictions))
                baselineErr.append(rmse)

            print step, ', baseline,', '%.4f,' % np.mean(baselineErr), '%.4f' % np.var(baselineErr)
            avg_baselineErr.append(np.mean(baselineErr))
            var_baselineErr.append(np.var(baselineErr))

            print step, ', linear,', '%.4f,' % np.mean(linearErr), '%.4f' % np.var(linearErr)
            avg_linearErr.append(np.mean(linearErr))
            var_linearErr.append(np.var(linearErr))

            print step, ', svm,', '%.4f,' % np.mean(svmErr), '%.4f' % np.var(svmErr)
            avg_svmErr.append(np.mean(svmErr))
            var_svmErr.append(np.var(svmErr))
            # linearErr.append(rmse)

        fig = pyplot.figure(1)
        pyplot.xticks(np.arange(5), np.arange(1, 6))
        fig.suptitle('error based on time steps', fontsize=20)
        pyplot.plot(avg_linearErr, label = 'linear rmse')
        pyplot.xlabel('steps', fontsize=18)
        pyplot.ylabel('rmse error', fontsize=18)

        # fig = pyplot.figure(2)
        # pyplot.xticks(np.arange(10), np.arange(1,11))
        # fig.suptitle('svm model error based on time steps', fontsize=20)
        pyplot.plot(avg_svmErr, label='svm rmse')
        pyplot.xlabel('steps', fontsize=18)
        pyplot.ylabel('rmse error', fontsize=18)

        # fig = pyplot.figure(3)
        # pyplot.xticks(np.arange(10), np.arange(1, 11))
        # fig.suptitle('normal model error based on time steps', fontsize=20)
        pyplot.plot(avg_baselineErr, label='normal rmse')
        pyplot.xlabel('steps', fontsize=18)
        pyplot.ylabel('rmse error', fontsize=18)

        # fig = pyplot.figure(3)
        # fig.suptitle('baseline model error based on time steps', fontsize=20)
        # pyplot.plot(avg_baselineErr, label='baseline rmse')
        # pyplot.xlabel('steps', fontsize=18)
        # pyplot.ylabel('rmse error', fontsize=18)


        # pyplot.plot(svmErr, label = 'svm(rbf kernel) rmse')
        # pyplot.plot(linearSvmErr, label='svm(linear) rmse')
        # pyplot.plot(linearErr, label='linear rmse')

        # print ("baseline mean %.3f" % np.mean(baselineErr))
        # print ("baseline var %.4f" % np.var(baselineErr))
        # print ("svm mean %.3f" % np.mean(svmErr))
        # print ("svm var %.4f" % np.var(svmErr))
        # print ("Linear  svm mean %.3f" % np.mean(linearSvmErr))
        # print ("Linear  svm var %.4f" % np.var(linearSvmErr))
        # print ("linear mean %.3f" % np.mean(linearErr))
        # print ("linear var %.4f" % np.var(linearErr))

        pyplot.legend()
        pyplot.show()

if __name__ == "__main__":
    blm = baseLineModel()
    blm.baseLine()
