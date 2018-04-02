import numpy as np
from scipy.optimize import minimize


def linear_model(x,y,a,b,weights):
    return (a*x - y + b)**2*weights

def fit_model(model,init_cond,x_vec,y_vec,weights):
    J = lambda param: np.sum(model(x_vec,y_vec,param[0],param[1],weights))
    sol = minimize(J,init_cond)
    return sol
