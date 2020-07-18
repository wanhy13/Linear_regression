
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io

def trainspamfilter(xTr,yTr):

    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Feel free to change this code any way you want
    
    # f = lambda w : hinge(w,xTr,yTr,.01)
    f = lambda w: ridge(w, xTr, yTr,1)
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),7e-6,1000)
   # w_trained = grdescent(f, np.random.normal(0,1,(xTr.shape[0],1)), 7e-6, 2000)
    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained
