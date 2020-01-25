import numpy as np
import pandas as pd

# data = pd.read_csv('train.csv')

# x = data['GrLivArea']
# y = data['SalePrice']

# x = (x - x.mean()) / x.std()
# x = np.c_[np.ones(x.shape[0]), x] 

s = np.array([(1,1), (2,2), (3,3)])
x = np.c_[np.ones(3), s.T[0]]
y = np.c_[np.ones(3), s.T[1]]

alpha = 0.1
iterations = 3
m = 3
w_init = np.array([0,0])

def gd(x, y, theta, iter, alpha):
    costs = []
    thetas = [theta]
    for i in range(iter):
        #y_pred = np.dot(v[i][0], weights)
        y_pred = np.dot(x, theta)
        err = y_pred - y
        cost = 1/(2*m) * np.dot(err.T, err)
        theta = theta - (alpha * (1/m) * np.dot(x.T, err) )
        print(theta)
    



gd(x, s.T[1], w_init, iterations, alpha)