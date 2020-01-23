####################################################################################
#
#   CSI 5130 Programming Assignment 1  - linear regression
#   implement gradient descent algorithm and stochastic gradient descent algorithm
#
####################################################################################
import numpy as np

points = [(1, 2), (2, 4)]

############################################################
# Modeling: 

def L(w):
# TODO: implement the loss
    return sum((x * w - y)**2 for x, y in points)

def dL(w):
# TODO: implement the gradient
    return sum(2 * (x * w - y) * x for x, y in points)

# TODO: the example code illustrates how to calculate loss for model y = w * x, 
# you need to implement a generalized sum of squares for d features model
# y = w1 * x1 + w2 * x2 + ... + wd * xd + w0
# in that case, the w is a vector of dimension d+1


def sL(w, i):
    return 1
# TODO: implement the loss at data point i

def sdL(w, i):
    return 1
# TODO: implement the stochastic gradient at data point i


    

############################################################
# Algorithms

def gradientDescent(L, dL, d):
    return 1
# TODO: Implement the gradient descent algorithm  
    
def stochasticGradientDescent(sL, sdL, d, n):
    return 1
# TODO: Implement the gradient descent algorithm  
# n = number of points    


if __name__ == '__main__':
############################################################
# Gradient descent algorithm example for model y = wx 

    w = 0
    eta = 0.01  # step size
    for t in range(50):
        value = L(w)
        gradient = dL(w)
        w = w - eta * gradient  # KEY: take a step
        print 'step {}, w = {}, F(w) = {}, F\'(w) = {}'.format(t, w, value, gradient)
    
    # TODO: load file to get the training data set
    
    # TODO: Set hyperparameters
    # hyperparameters = {
    #                'learning_rate': [...],
                    #'weight_regularization': [...], placeholder for regularization
    #                'num_iterations': [...],
                    # related to prior weight
    #                'weight_decay': [...] 
    #                }

    # TODO average over multiple runs
    
    # TODO training the model and testing it

    # TODO generate plots
    # [...]
    
    # TODO analysis
    