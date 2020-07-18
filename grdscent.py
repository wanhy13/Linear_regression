
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-15):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    w = w0
    for iter in range(1,maxiter):
        oloss = func(w)[0]
        wnext = w-stepsize*func(w)[1]
        loss, grad = func(wnext)
        if(oloss>loss):
            stepsize = stepsize*1.01
        if(oloss<loss):
            stepsize = stepsize * 0.5

        w = wnext

        # print(loss, np.linalg.norm(grad), stepsize)

        if np.linalg.norm(stepsize*func(w)[1])< tolerance:
            break

    eps = 2.2204e-14 #minimum step size for gradient descent

    
    return w
