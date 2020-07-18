from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regression constant
# w weight vector (default w=0)
# w: dx1
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
    loss = np.sum(np.maximum(0, np.add(1, -np.multiply(yTr, w.T.dot(xTr))))) + lambdaa * w.T.dot(w)
    # a indicate which point is valid
    a = 1-yTr*w.T.dot(xTr)
    a = (abs(a)+a)*0.5
    a = np.sign(a)
    gradient = 2 * lambdaa * w -(yTr*xTr).dot(a.T)
    # gradient = gradient.reshape(gradient.shape[0], 1)
    
    return loss,gradient
