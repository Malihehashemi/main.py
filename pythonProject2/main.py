
import matplotlib
matplotlib.use('TkAgg')  # or any other backend you prefer
import numpy as np
from numpy import matlib, log, vander, eye, exp
from numpy.linalg import inv, det, cholesky, solve
import matplotlib.pyplot as plt

time = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
degradation_210 = np.array([0.0005760, 0.00056998, 0.0005646, 0.0005642, 0.0005640, 0.0005628, 0.0005610, 0.000562, 0.0005622])
degradation_210_New = np.array([0.0005760, 0.00056998, 0.0005646, 0.0005642, 0.0005640, 0.0005628, 0.0005610])

def LODE_kernel(x,y, theta):
    return exp(-0.5*np.sum(x+y)/theta**2)


def kernel_matrix(X, Y, kernel, kernel_standard_dev):
    n = X.shape[0]
    m = Y.shape[0]
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i][j] = LODE_kernel(X[i], Y[j], kernel_standard_dev)
    return K

def kernel_matrix_squared(X, kernel, kernel_standard_dev):
    n = X.shape[0]
    K_s = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            K_s[i][j] = LODE_kernel(X[i], X[j], kernel_standard_dev)
    K_s = K_s + K_s.transpose()
    return K_s

n_t = len(time)
def myGP(X_train, y_train, X_test, theta, tau, kernel):
    #est_meas_noise_stdDev = ((np.max(y_train)-np.min(y_train))/100.0) * tau
    est_meas_noise_stdDev=0.1
    i_times_tau = np.identity(n_t) * (est_meas_noise_stdDev ** 2)*1000

    # kernel matrix for training points
    K_a = kernel_matrix_squared(X_train, kernel, theta)
    # kernel matrix of prediction points
    K_ss_a = kernel_matrix_squared(X_test, kernel, theta)
    # kernel matrix of mixed training and prediction points
    K_s_a = kernel_matrix(X_train, X_test, kernel, theta)

    # chol of inverse of kernel matrix with added noise covariance
    L = cholesky(K_a + i_times_tau)

    # calculate GP weights (using chol)
    alpha = solve(L.T, solve(L, y_train))

    # calculate Postirior mean
    mu = K_s_a.T @ alpha
    plt.plot(X_test, mu, label="Mean prediction", linestyle="dotted", color=(0.9, 0.3, 0))
    # caculating Posterior standard deviation
    v = solve(L, K_s_a)
    cov = K_ss_a - v.T @ v
    # cov     = 1-np.sum((kappa @ K_Xx) * K_Xx, 0)
    stdDev = np.sqrt(np.diag(np.clip(cov, a_min=0, a_max=None)))
    return mu, stdDev
theta = 90.85
sigma = 0.1
tau = 1723*765.8
new_time = np.arange(0, 151, 1)
kernel= LODE_kernel(time, degradation_210, theta)
myGP(time, degradation_210, new_time, theta, tau, kernel)
# Show the plot
plt.legend()
plt.xlabel('Time')
plt.ylabel('Degradation')
plt.title('Mean Prediction')
plt.show()