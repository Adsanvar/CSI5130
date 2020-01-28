import numpy as np
import pandas as pd

data = pd.read_csv('alkali_data.csv')
data = data.drop(columns=['time'], axis=1)
v_out = data.iloc[:,6:12]
naoh_out = data['output_naoh_1-6']
attributes = data.drop(columns=['output_naoh_1-6'], axis=1) #drops output_naoh_1-6 from our X
attributes = attributes.drop(attributes.columns[7:13], axis=1) #drops Voltages 1-6 from out X

attributes = (attributes - attributes.mean()) / attributes.std()
attributes = np.c_[np.ones(attributes.shape[0]), attributes]

#GRADIENT DESCENT

alpha = 0.01 #Step size
iterations = 1000 #No. of iterations
m = naoh_out.size #No. of data points
np.random.seed(123) #Set the seed
theta = np.random.rand(66) #Pick some random values to start with


#GRADIENT DESCENT
def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs

#Pass the relevant variables to the function and get the new values back...
past_thetas, past_costs = gradient_descent(attributes, naoh_out, theta, iterations, alpha)
print(past_thetas)
theta = past_thetas[-1]

#Print the results...
print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))