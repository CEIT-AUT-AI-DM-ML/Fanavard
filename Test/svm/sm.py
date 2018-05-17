from sklearn import svm
import pandas as pd
from matplotlib import pyplot

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
clf = svm.SVC()
print clf.fit(df[['a1']], df['a8']).score(df[['a1']], df['a8'])
print "09"