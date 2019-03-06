# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:09:09 2019

@author: hamidm
"""
  
def readData():    
    
    import pandas 
    
    X = pandas.read_csv('sp500_original.csv', header=0)
    X = X.drop('Date', axis=1)
    X = X.dropna()
    
    y = X['Close'].shift(-1).dropna()
    X = X.iloc[1:]
    
    
    X = X.values    
    for j in range(1, len(X)): 
        X[j] = X[j] - X[j-1] 
    X = X[1:]
    
    y = y.values
    diff = []
    for j in range(1, len(y)): 
        diff.append( y[j] - y[j-1])    
    y = diff
    
    
    
    return X, y 
    
    
if __name__ == '__main__':
    
    X, y = readData()
    
    
    
    
    
    
    
    
    
    
    