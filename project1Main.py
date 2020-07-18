from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from grdscent import grdescent
from valsplit import valsplit

from ridge import ridge
from scipy import io
import numpy as np

from checkgradLogistic import checkgradLogistic
from checkgradHingeAndRidge import checkgradHingeAndRidge

from ridge import ridge
from hinge import hinge
from logistic import logistic

# load the data:
data = io.loadmat('data/data_train_default.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)
# f = lambda w : ridge(w,xTr,yTr,0.1)
# w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),7e-4,1000)
small_step = 1e-5
feature_vector = np.zeros((xTr.shape[0],1))
# feature_vector =
lambdaa = 10

# ridge_error = checkgradHingeAndRidge(ridge, feature_vector, small_step, xTr, yTr, lambdaa)
# print("Ridge error is", ridge_error)
#
# hinge_error = checkgradHingeAndRidge(hinge, feature_vector, small_step, xTr, yTr, lambdaa)
# print("Hinge error is", hinge_error)
#
# logistic_error = checkgradLogistic(logistic, feature_vector, small_step, xTr, yTr)
# print("Logistic error is", logistic_error)

# train spam filter with parameters in trainspamfilter.py
        
w_trained = trainspamfilter(xTr,yTr)
spamfilter(xTv,yTv,w_trained)












