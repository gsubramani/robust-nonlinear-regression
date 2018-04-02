import numpy as np

def robust_lsq(model_error_func, model_fit_func, X,
               iterations=1000, fit_samples=2, fit_with_best_n=None, priors=None):


    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors

    probabilities = probabilities / np.sum(probabilities)
    indices = np.array(range(len(X)))

    for iter in range(iterations):
        sampled_indices = np.random.choice(range(len(indices)),
                                           p=probabilities, size=fit_samples, replace=False)

        # X_subset = X[sampled_indices, :]
        X_subset = X[sampled_indices]
        params = model_fit_func(X_subset)
        errors = model_error_func(X, params)

        current_prob = 1 / np.arctan(1 + errors)
        probabilities = probabilities * current_prob
        probabilities = probabilities / np.sum(probabilities)

    if fit_with_best_n == None:
        return probabilities
    else:
        robust_X = X[np.argsort(probabilities)[-fit_with_best_n:]]
        robust_params = model_fit_func(robust_X)
        return probabilities, robust_params, model_error_func(robust_X, robust_params)

def robust_lsq_m_estimates_(model_error_func, model_fit_func, X,
                           iterations=1000, priors=None):
    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors
    probabilities = probabilities / np.sum(probabilities)
    for iter in range(iterations):
        params = model_fit_func(X, probabilities)
        errors = model_error_func(X, params, probabilities)
        current_prob = 1 / (1 + errors **0.1)
        # current_prob = 1/np.arctan(1+errors)
        probabilities = probabilities * current_prob
        probabilities = probabilities / np.sum(probabilities)
    params = model_fit_func(X, probabilities)
    errors = model_error_func(X, params, probabilities)
    return probabilities, params, errors


def robust_lsq_m_estimates(model_error_func, model_fit_func, X,
                           iterations=1000, priors=None):
    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors

    probabilities = probabilities / np.sum(probabilities)
    best_errors = 1e100
    for iter in range(iterations):
        params = model_fit_func(X, probabilities)
        errors = model_error_func(X, params, probabilities)
        if np.sum(errors) < best_errors:
            best_param = params
            best_errors = np.sum(errors)
        current_prob = 1 / (1 + errors ** 0.1)
        # current_prob = 1/np.arctan(1+errors)
        probabilities = probabilities * current_prob
        probabilities = probabilities / np.sum(probabilities)
    params = model_fit_func(X, probabilities)
    errors = model_error_func(X, params, probabilities)
    if np.sum(errors) < best_errors:
        best_param = params
    return probabilities, best_param, errors


## test out robust fitting

import matplotlib.pyplot as plt
from scipy.optimize import minimize

def model(x,y,a,b,weights):
    return (a*x - y + b)**2*weights

def fit_model(model,init_cond,x_vec,y_vec,weights):
    J = lambda param: np.sum(model(x_vec,y_vec,param[0],param[1],weights))
    sol = minimize(J,init_cond)
    return sol



if __name__ == "__main__":

    x_actual = np.random.rand(100)
    y_actual = x_actual * (-1) + 3
    y_measured = y_actual + np.random.rand(len(x_actual)) * 0.1 - 0.05
    x_measured = x_actual + np.random.rand(len(x_actual)) * 0.1 - 0.05
    x_measured = np.append(x_measured, np.array(range(50)) / 100.0)
    y_measured = np.append(y_measured, np.zeros(50) + np.random.rand(50) * 10)

    X = np.transpose([x_measured, y_measured])
    weights = np.ones(len(X))
    model_fit_func = lambda X: fit_model(model,[1,1],X[:,0],X[:,1],np.ones(len(X))).x
    model_error_func = lambda X,params: model(X[:,0],X[:,1],params[0],params[1],np.ones(len(X)))

    sol_lsq = fit_model(model, [1, 1], x_measured, y_measured,weights)

    probs,params,errors = robust_lsq(model_error_func,model_fit_func,X,
                                 iterations = 100,fit_samples = 10,fit_with_best_n = 100)

    model_fit_weights_func = lambda X, weights: fit_model(model, [1, 1], X[:, 0], X[:, 1], weights).x
    model_error_weights_func = lambda X, params, weights: model(X[:, 0], X[:, 1], params[0], params[1], weights)

    X = np.transpose([x_measured, y_measured])

    probsMest, paramsMest, errorsMest = robust_lsq_m_estimates(model_error_weights_func, model_fit_weights_func, X,
                                                   iterations=40, priors=None)


    robust_indices = np.where(probs > 0.0002)
    outlier_indices = np.where(probs < 0.0002)

    robust_fit_x = x_measured[robust_indices]
    robust_fit_y = y_measured[robust_indices]

    outlier_x = x_measured[outlier_indices]
    outlier_y = y_measured[outlier_indices]


    plt.plot(x_actual, y_actual, '.b')
    plt.plot(x_measured, y_measured, '.g')
    plt.plot(x_measured, x_measured * sol_lsq.x[0] + sol_lsq.x[1], 'r')
    plt.plot(x_measured, x_measured * params[0] + params[1], '.k')
    plt.plot(x_measured, x_measured * paramsMest[0] + paramsMest[1], '--k')
    plt.plot(robust_fit_x, robust_fit_y, 'ok')
    plt.plot(outlier_x, outlier_y, 'xk')
    plt.show()

