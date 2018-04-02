#####################################
# Author:   Guru Subramani
# Date:     April 2nd 2018
# MIT License
#####################################

import numpy as np

def robust_lsq_ransac(model_error_func, model_fit_func, X,
               iterations=1000, fit_samples=2, fit_with_best_n=None, priors=None,norm_func = np.arctan):

    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors

    probabilities /= np.sum(probabilities)
    indices = np.array(range(len(X)))
    current_prob = probabilities.copy()
    for iter in range(iterations):
        sampled_indices = np.random.choice(range(len(indices)),
                                           p=probabilities, size=fit_samples, replace=False)

        X_subset = X[sampled_indices]
        params = model_fit_func(X_subset)
        errors = model_error_func(X, params)

        current_prob[:] = 1 / norm_func(1 + errors[:])
        probabilities *= current_prob
        probabilities /= np.sum(probabilities)

    if fit_with_best_n == None:
        return probabilities
    else:
        robust_X = X[np.argsort(probabilities)[-fit_with_best_n:]]
        robust_params = model_fit_func(robust_X)
        return probabilities, robust_params, model_error_func(robust_X, robust_params)


def robust_lsq_m_estimates(model_error_func, model_fit_func, X,
                           iterations=1000, priors=None,norm_func = lambda x: 1/(1 + x**0.1)):
    if priors == None:
        probabilities = np.ones(len(X))
    else:
        probabilities = priors

    probabilities /= np.sum(probabilities)
    best_errors = 1e100
    best_param = None
    current_prob = probabilities.copy()
    for iter in range(iterations):
        params = model_fit_func(X, probabilities)
        errors = model_error_func(X, params, probabilities)
        if np.sum(errors) < best_errors:
            best_param = params
            best_errors = np.sum(errors)

        current_prob[:] = norm_func(errors[:])
        probabilities *= current_prob
        probabilities /= np.sum(probabilities)
    params = model_fit_func(X, probabilities)
    errors = model_error_func(X, params, probabilities)
    if np.sum(errors) < best_errors:
        best_param = params
    return probabilities, best_param, errors


