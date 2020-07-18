from scipy import io
from trainspamfilter import trainspamfilter
from vis_spam import vis_spam


data = io.loadmat('data/data_train_default.mat')
X = data['X']
Y = data['Y']
X = X.toarray()
w = trainspamfilter(X,Y)
vis_spam(w)