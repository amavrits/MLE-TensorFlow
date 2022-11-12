from tensorflow_probability import distributions as tfd
import tensorflow as tf
import time as tm
import matplotlib.pyplot as plt
import numpy as np
from MLE_TF import MLE

N = 100
x = np.random.uniform(0, 10, N)
y = 2 + 5 * x + np.random.randn(N) * 0.5
x = np.c_[np.ones_like(x), x]

def linear_regression(theta, x, y):
    dist = tfd.Normal(theta[0] + theta[1] * x[:, -1], scale=theta[2])
    def true_fn(): return tf.reduce_sum(dist.log_prob(y), axis=0)
    def false_fn(): return -np.inf
    return tf.cond(theta[2] > 0, true_fn, false_fn)

loglike = lambda theta, y: linear_regression(theta, x, y)

mle = MLE(loglike)

start = tm.time()
mle.fit(y, n_theta=x.shape[1] + 1)
end = tm.time()
print(f"tensorflow took: {end - start:.2f} seconds")


# mle.nonparameteric_bootstrapping(n_bootstrap=3)

