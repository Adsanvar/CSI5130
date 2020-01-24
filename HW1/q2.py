import numpy as np

s = np.array([(1,1), (2,2), (3,3)])

weights = [0, 0]
step = 0.1
epoch = 3
N = 3

def gd(v, w, epoch, step):
    weights = w
    cost = 0

    for i in range(epoch):
        y_pred = np.dot(v[i][0], weights)
        print(y_pred)






gd(s, w0,w1,epoch,step)