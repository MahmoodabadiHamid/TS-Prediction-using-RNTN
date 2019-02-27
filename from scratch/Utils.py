# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:09:09 2019

@author: hamidm
"""


  
def readData():    
    from pandas import read_csv
    from sklearn.linear_model import LinearRegression
    from matplotlib import pyplot as plt
    from matplotlib import pyplot
    from sklearn import preprocessing
    import pandas 
    
    
    X = pandas.read_csv('sp500_original.csv', header=0)
    print(X.shape)
    X = X.drop('Date', axis=1)
    print(X.shape)
    X = X.dropna()
    y = X['Close'].shift(-1).dropna()
    X = X.iloc[1:]
    pyplot.plot(X['Close'])
    pyplot.show()
    pyplot.plot(X['Open'])
    pyplot.show()
    pyplot.plot(X['Volume'])
    pyplot.show()
    pyplot.plot(X['High'])
    pyplot.show()
    pyplot.plot(X['Low'])
    pyplot.show()
    
    X = X.values
    
    
    
    for j in range(1, len(X)): 
        X[j] = X[j] - X[j-1]
    
    X = X[1:]
    
    y = y.values
    diff = []
    for j in range(1, len(y)): 
        diff.append( y[j] - y[j-1])
    
    #y = diff
    print('Done...............')
    pyplot.figure(figsize=(20,8))
    pyplot.subplot(311)
    plt.title('Close price')
    plt.plot(y)
    pyplot.subplot(313)
    plt.title('Detrended Close price')
    plt.plot(diff)
    plt.savefig('aa')
    plt.show()
    
    print(X.shape)
    print(len(y))
    
    
    
if __name__ == '__main__':
    
    data = readData()
    
    
    
    
    
    
    
    
    
    
    