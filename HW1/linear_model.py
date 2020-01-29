import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

###Variables###
#loads data
data = pd.read_csv('alkali_data.csv')
#drops unecessay data 
data = data.drop(columns=['time' , 'output_nacl_PH_1', 'output_nacl_PH_2', 'output_nacl_PH_3', 'output_nacl_PH_4', 'output_nacl_PH_5','output_nacl_PH_6'], axis=1)
#selects voltage_1, ... , voltage_6
v_out = data.iloc[:,6:12]
#selects 'output_naoh_1-6'
naoh_out = data['output_naoh_1-6']
#drops 'output_naoh_1-6' and only contains attributes
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

#variance
def variance(data):
    x_bar = mean(data)
    return np.sum(np.square(data - x_bar))/ (len(data) -1)

#standard deviation
def stdev(data):
    return np.sqrt(variance(data))

#cov variance
def cov_var(x, y):
    if len(x) == len(y):
        x_bar = mean(x)
        y_bar = mean(y)
        _sum = 0 
        for i in range(len(x)):
            _sum += ((x.iloc[i] - x_bar)*(y.iloc[i]-y_bar))
        return _sum / (len(x)-1)

#Pearson's Correlation
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
            x_2 += np.square(x_)
            y_2 += np.square(y_)

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
    cost_history = [] # For visualizations
    for i in range(iter):
        y_pred = np.dot(X, w) #iterates thru all point in dataset
        error = y_pred - y
        cost = 1/(2*m) * np.dot(error.T, error) #For visualizations 
        cost_history.append(cost)
        cost_fn = np.dot(X.T, y_pred - y) #cost function
        w = w - (alpha * (1/m) * cost_fn) #updates weights
    return w, cost_history

#Stochastic Gradien Descent
'''
X: Data
Y: label
w: initial weights
alpha: step or learning rate
iter: number of itterations
'''
def sgd(X, y, w, alpha, iter):
    m = len(y)
    w = w
    for i in range(iter):
        r_index = np.random.choice(m, replace=True) #selects random record in dataset
        x = X[r_index]
        y_i = y[r_index]
        y_pred = np.dot(x, w) #predicts
        error = y_pred - y_i
        cost_fn = np.dot(x.T, error) #cost function
        w = w - (alpha * (1/m) * cost_fn) #updats weights
    return w

#Linear Regression using ordinary least squares formula
'''
x_train: training data_set
y_train: training labels
'''
def linear_regression(x_train, y_train):
    #Ordinary least sqaures formula: 
    # ß = (X^T . X)^-1 . X^T . y
    # y = X * ß
    w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    #y_pred = x_train.dot(w)

    return w

#Mean Squared Error
def mse(acutal, predicted):
    return np.sum(np.square(acutal - predicted)) / len(acutal)

#Root Mean Squared Error
def rmse(acutal, predicted):
    return np.sqrt(np.sum(np.square(acutal - predicted)) / len(acutal))

#Saves Image for analysis
def saveimg(actual, predicted, name):
    
    index = np.arange(len(actual)) 
    width = 0.3       
    plt.bar(index, predicted, width, label='Y\'')
    plt.bar(index + width, actual, width,label='Y')

    plt.ylabel('Y')
    plt.xlabel('Test Sample\nMSE: '+ str(np.round(mse(actual, predicted), 3)) + '\nRMSE: ' + str(np.round(rmse(actual, predicted),3)))
    plt.title(str(name)+' vs predicted')

    plt.xticks(index + width / 2, index)
    plt.legend(loc='best')
    plt.savefig(str(name)+'.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    #Hyperparameters:
    alpha = 0.0001 #step
    iterations = 10000 # iterations
    np.random.seed(123) #sets random seed
    w = np.random.rand(attributes.shape[1]+1) #random variables for the weights
    
    #Normalize Data to test on GD and SGD
    n_attributes = (attributes - mean(attributes)) / stdev(attributes)
    n_attributes = np.c_[np.ones(attributes.shape[0]), n_attributes]
    y_norm = (naoh_out - mean(naoh_out)) / stdev(naoh_out)
    #sgd doesn't require a bias so n_attributes are differen from n_attributes
    sgd_n_attributes = (attributes - mean(attributes)) / stdev( attributes)

    #gradient descent on standardized data
    start = timeit.default_timer() 
    weights, cost = gd(n_attributes, y_norm, w,alpha, iterations)
    time = timeit.default_timer() - start
    print(time)

    '''
    #Temp Image
    # plt.title("output_naoh_1-6 SGD Cost Function")
    # plt.ylabel("Cost")
    # plt.xlabel("iterations\n")
    # plt.plot(cost)
    # # plt.show()
    # plt.savefig('output_naoh_1-6_GD_Cost_Function.png',bbox_inches='tight')
    # plt.close()
    '''

    w = np.delete(w, 0,0) # removes bias of GD weight

    #Stochastic Gradient Descent on standardized data 
    start = timeit.default_timer()
    w_ = sgd(sgd_n_attributes.to_numpy(), y_norm, w, alpha, iterations)
    time = timeit.default_timer() - start
    print(time)

    #Test and train data
    x_test = attributes.sample(n=20) #select 1000 random samples to test
    naoh_test = naoh_out.loc[x_test.index.tolist()] # selects the corresponding y_test values
    x_train = attributes.drop(x_test.index.tolist()) #drops x_test values and sets to x_train
    naoh_train = naoh_out.drop(naoh_test.index.tolist()) #drops naoh_test values and sets to naoh_train

    #Linear Regression
    model = linear_regression(x_train, naoh_train)
    naoh_pred = x_test.dot(model)

    ##Uncomment to run voltage analyis
    # for i in v_out:
    #     v_data = v_out[i]
    #     v_test   = v_data.loc[x_test.index.tolist()]
    #     v_train = v_data.drop(v_test.index.tolist())
        
    #     model = linear_regression(x_train, v_train)
    #     v_pred = x_test.dot(model)

    #     #saveimg(v_test, v_pred, i) #saves images
    