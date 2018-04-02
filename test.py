## test out robust fitting

import matplotlib.pyplot as plt
from models import *
from robust_lsq import *

if __name__ == "__main__":

    x_actual = np.random.rand(100)
    y_actual = x_actual * (-1) + 3
    y_measured = y_actual + np.random.rand(len(x_actual)) * 0.1 - 0.05
    x_measured = x_actual + np.random.rand(len(x_actual)) * 0.1 - 0.05
    x_measured = np.append(x_measured, np.array(range(50)) / 100.0)
    y_measured = np.append(y_measured, np.zeros(50) + np.random.rand(50) * 10)

    X = np.transpose([x_measured, y_measured])
    weights = np.ones(len(X))
    model_fit_func = lambda X: fit_model(linear_model,[1,1],X[:,0],X[:,1],np.ones(len(X))).x
    model_error_func = lambda X,params: linear_model(X[:,0],X[:,1],params[0],params[1],np.ones(len(X)))

    sol_lsq = fit_model(linear_model, [1, 1], x_measured, y_measured,weights)

    probs,params,errors = robust_lsq_ransac(model_error_func,model_fit_func,X,
                                 iterations = 100,fit_samples = 10,fit_with_best_n = 100)

    model_fit_weights_func = lambda X, weights: fit_model(linear_model, [1, 1], X[:, 0], X[:, 1], weights).x
    model_error_weights_func = lambda X, params, weights: linear_model(X[:, 0], X[:, 1], params[0], params[1], weights)

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

