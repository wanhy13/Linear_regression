import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):


    # loss = np.sum(np.log(1+np.exp(-yTr*(w.T.dot(xTr)))))
    # gradient = np.sum((-xTr.T * yTr.T)/(1+np.exp(yTr*(w.T.dot(xTr)))).T, axis=0).T




    loss = np.sum(np.log(1+np.exp(-yTr.T*(xTr.T.dot(w)))))
    gradient =np.sum((-yTr.T*xTr.T)/(1+np.exp(yTr.T*(xTr.T.dot(w)))),axis = 0)
    gradient = gradient.reshape(gradient.shape[0],1)



    return loss,gradient
