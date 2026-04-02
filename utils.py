import numpy as np

def rmse(y, y_pred):
    return np.sqrt(((y - y_pred)**2/2).sum())
def grad_mse(y, y_pred):
    return -(y - y_pred)
def hess_mse(y, y_pred):
    return np.ones(y.shape[0])

