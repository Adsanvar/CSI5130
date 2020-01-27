import pandas as pd
import numpy as np
from scipy.stats import pearsonr
###VARS###
data = pd.read_csv('alkali_data.csv')
data = data.drop(columns=['time'], axis=1)
v_out = data.iloc[:,7:13]
naoh_out = data['output_naoh_1-6']
attributes = data.drop(columns=['output_naoh_1-6'], axis=1) #drops output_naoh_1-6 from our X
attributes = attributes.drop(attributes.columns[7:13], axis=1) #drops Voltages 1-6 from out X


#mean
def mean(data):
    return data.sum()/len(data)

#median requires values to be sorted, using pandas to sort the input dataframe
def median(data):
    data = data.sort_values()
    l = len(data)
    #even
    if(l % 2 == 0 ):
        loc = l/2
        loc_1 = loc +1 
        return (data.iloc[int(loc)] + data.iloc[int(loc_1)])/2
    #odd
    else:
        return data.iloc[int(np.ceil(l/2))]

def variance(data):
    x_bar = mean(data)
    var = 0
    for i in data:
        var += (i - x_bar) ** 2
    var = var / (len(data) -1)
    return var 

def stdev(data):
    return np.sqrt(variance(data))

def cov_var(x, y):
    
    if len(x) == len(y):
        x_bar = mean(x)
        y_bar = mean(y)
        _sum = 0 
        for i in range(len(x)):
            _sum += ((x.iloc[i] - x_bar)*(y.iloc[i]-y_bar))
        return _sum / (len(x)-1)

def p_cov(x, y):
    if len(x) == len(y):
        x_bar = mean(x)
        y_bar = mean(y)
        x_y = 0
        x_2 = 0
        y_2 = 0
        for i in range(len(x)):
            x_ = (x.iloc[i] - x_bar)
            y_ = (y.iloc[i]- y_bar)
            x_y += x_ * y_
            x_2 += x_ **2
            y_2 += y_ **2

        return x_y/ (np.sqrt(x_2) * np.sqrt(y_2))

#cost function
def cost(X, y, w):
    m = len(y)
    y_pred = X.dot(w)
    cost = (1/(2*m)) * np.sum(np.square(y_pred-y))
    return cost

#Gradient Descent
'''
X: Data
Y: label
w: initial weights
alpha: step or learning rate
iter: number of itterations
'''
def gd(X, y, w, alpha, iter):
    m = len(y)
    for i in range(iter):
        y_pred = np.dot(X, w)
        summation = np.dot(X.T, y_pred - y)
        w = w - alpha * (1/m) * summation
    return w

if __name__ == '__main__':
    
    #Hyperparameters:
    alpha = 0.01
    iterations = 1000
    np.random.seed(123)
    w = np.random.rand(2)
    weights = []
    cost = []

    for i in attributes:
        X = attributes[i]
        X = (X - mean(X)) / stdev(X)
        X = np.c_[np.ones(X.shape[0]), X]
        
        
        
    #w = gd(attributes, naoh_out, w, alpha, iterations)

    #print(w)

