from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


class baseLineModel :

    def __init__(self):
        self.df = []
        self.dataset_url=''

    def baseLine(self):


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


        # plot data after removing outliers
        # for i in range (0 ,len(X)  ) :
        #     # print (i)
        #     pyplot.figure(i)
        #     pyplot.plot(X[i], label='Value')
        # pyplot.show()


        # set train and test randomly (using featuer a1)
        msk = np.random.rand(len(self.df['a1'])) < 0.8
        print (msk)
        train = X[0][msk]
        test = X[0][~msk]

        # print (train)
        # print(len(train))
        # print (test)
        # print(len(test))


        # normal model
        history = [x for x in train]
        predictions = list()
        nb_correct_predict = 0
        for i in range(len(test)):
            #print( i)
            # get the average of history of 3 last row as predictions
            if (i > 0 and i < len(test) - 1 ) :
                predictions.append((history[-1] + history[-2] + history[-3] ) / 3 )
            elif (i == 0 or i == len(test) - 1):
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
        pyplot.legend()
        pyplot.show()

if __name__ == "__main__":
    blm = baseLineModel()
    blm.baseLine()
