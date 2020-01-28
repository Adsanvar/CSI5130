import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


###VARS###
data = pd.read_csv('alkali_data.csv')
data = data.drop(columns=['time' , 'output_nacl_PH_1', 'output_nacl_PH_2', 'output_nacl_PH_3', 'output_nacl_PH_4', 'output_nacl_PH_5','output_nacl_PH_6'], axis=1)
v_out = data.iloc[:,6:12]
naoh_out = data['output_naoh_1-6']
attributes = data.drop(columns=['output_naoh_1-6'], axis=1) #drops output_naoh_1-6 from our X
attributes = attributes.drop(attributes.columns[7:13], axis=1) #drops Voltages 1-6 from out X

#Standardized data
# std_v_out = StandardScaler().fit_transform(v_out)
# std_naoh_out = StandardScaler().fit_transform(naoh_out)
# std_attr = StandardScaler().fit_transform(attributes)

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
    cost_history = []
    for i in range(iter):
        y_pred = np.dot(X, w)
        error = y_pred - y
        cost = 1/(2*m) * np.dot(error.T, error)
        cost_history.append(cost)
        cost_fn = np.dot(X.T, y_pred - y)
        w = w - (alpha * (1/m) * cost_fn)
    return w, cost_history


def sgd(X,y,w,alpha, iter):
    m = len(y)
    w = w
    for i in range(iter):
        r_index = np.random.choice(m, replace=True)
        x = X[r_index]
        y_i = y[r_index]
        y_pred = np.dot(x, w)
        error = y_pred - y_i
        cost_fn = np.dot(x.T, error)
        w = w - (alpha * (1/m) * cost_fn)
    
    return w


def linear_regression(x, y, alpha, w, b, itter):
    m = len(y)

    for i in range(itter):
        o_train = np.dot(w, x) + b
        cost_train = (1/(2*m)) * (np.sum(np.square(o_train, y)))
        


    


if __name__ == '__main__':

    #Hyperparameters:
    alpha = 0.01 #step
    iterations = 750 # 750 iterations is good enough
    np.random.seed(123) #sets random seed
    w = np.random.rand(attributes.shape[1]+1) #random variables for the weights
    b = w # for linear regression
    weights = []
    cost = []

    #Normalize Data
    n_attributes = (attributes - mean(attributes)) / attributes.std()
    n_attributes = np.c_[np.ones(attributes.shape[0]), n_attributes]
    y_norm = (naoh_out - mean(naoh_out)) / naoh_out.std()
    sgd_n_attributes = (attributes - mean(attributes)) / attributes.std()

    #gradient descent
    #weights, cost = gd(n_attributes, v_out['voltage_6'], w,alpha, iterations)

    # plt.title("voltage_6 Cost Function")
    # plt.ylabel("Cost")
    # plt.xlabel("iterations")
    # plt.plot(cost)
    # # plt.show()
    # plt.savefig('voltage_6_Cost_Function.png')


    w = np.delete(w, 0,0) # removes initial weight
    #Stochastic Gradient Descent
    #sgd(sgd_n_attributes.to_numpy(), y_norm, w, alpha, 10000)

    #Linear Regression

    
    x_test = attributes.sample(n=1000) #select 1000 random samples to test
    y_test = naoh_out.loc[x_test.index.tolist()] # selects the corresponding y_test values
    x_train = attributes.drop(x_test.index.tolist())#drops x_test values and sets to x_train
    y_test = naoh_out.drop(y_test.index.tolis())
    
    #linear_regression(attributes, naoh_out)

    
