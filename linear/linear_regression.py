from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
from pandas import DataFrame
from math import sqrt


def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df



# read data from dataset
dataset_url = '../data/A_ticker.csv'
df = pd.read_csv(dataset_url, header=0, parse_dates=[0], index_col=0, squeeze=True)

X = []
for i in range(1,9) :
    s = 'a' +str(i)
    #print s
    X.append(df[s].values)

# remove outliers
for i in range (0 ,len(X)):
        for j in range (1,(X[i].size) - 1) :
            #print j
            if X[i][j] < 2 :
                X[i][j] =  ( X[i][j - 1] )

msk = np.random.rand(len(df['a1'])) < 0.8
err = []
#for step in range(1, 10):
step = 2
#print step

#chaneg time serie to supervise
supervised = timeseries_to_supervised(X[0], step)
supervised_values = supervised.values

#split data to x and y
x = supervised_values[:, :step]
y = supervised_values[:, step]

# set train and test randomly (using featuer a1)
x_train = x[msk]
y_train = y[msk]
x_test = x[~msk]
y_test = y[~msk]

class linear_regression:
    def __init__(self, bias=False):
        self.bias = bias
        self.coefficient = None

    def fit(self, X, Y):
        if self.bias:
            X = np.insert(X, X.shape[1], 1, axis=1)
        a = np.dot(X.T, X)
        b = np.dot(X.T, Y)
        a_invese = np.linalg.inv(a)
        self.coefficient = np.dot(a_invese, b)

    def predict(self, X):
        if self.bias:
            X = np.insert(X, X.shape[1], 1, axis=1)
        return np.dot(X, self.coefficient)

    def RSS(self, X, Y):
        if self.bias:
            X = np.insert(X, X.shape[1], 1, axis=1)
        predicted = np.dot(X, self.coefficient)
        square_diff = (y_test - predicted) ** 2
        return np.sum(square_diff)



housing_classifier = linear_regression(bias=True)
housing_classifier.fit(x_train, y_train)

rss = housing_classifier.RSS(x_test, y_test)
print("The RSS of the housing validation set is %f" % rss)

predicted = housing_classifier.predict(x_test)
min_price = min(predicted)
max_price = max(predicted)
print("The median home value range from %f to %f" % (min_price, max_price))


#### Plotting Coefficients
coefficients = plt.figure(1)
plt.title("Regression coefficients")
indices = range(len(housing_classifier.coefficient)-1)

plt.plot(indices, housing_classifier.coefficient[0:-1], linestyle='-', marker='o')
plt.xlabel('Indices', fontsize=14, color='blue')
plt.ylabel('Coefficient', fontsize=14, color='blue')
plt.grid(True)


#### Plotting residuals
residuals = plt.figure(2)
plt.title("Residuals")

residuals_val = housing_classifier.predict(x_test) - y_test
plt.hist(residuals_val, 75, facecolor='green')

