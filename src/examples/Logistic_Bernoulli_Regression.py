from tensorflow_probability import distributions as tfd
import tensorflow as tf
import time as tm
import matplotlib.pyplot as plt
import numpy as np
from MLE_TF import MLE

N = 100_000
P = 250

# 10_000--> bias and var increases / #2_000_000 --> they both decrease
# instead of MLE, work with empirical risk!!!

def bernoulli_logit_regression(theta, x, y):
    theta = tf.expand_dims(theta, 1)
    dist = tfd.Bernoulli(x @ theta)
    return tf.reduce_sum(dist.log_prob(y))

tf.random.set_seed(123)
alpha_true = tfd.Normal(0.666, 1.0).sample()
beta_true = tfd.Normal(0.0, 3.14).sample([P, 1])
theta_true = np.append(alpha_true, beta_true)

x = tfd.Normal(0.0, 1.0).sample([N, P])
x = tf.concat([tf.reshape(tf.ones(x.shape[0]), [-1, 1]), x], axis=1)
y = tfd.Bernoulli(x @ theta_true.reshape(-1, 1)).sample()
y = tf.cast(y, tf.float32)

loglike = lambda theta, y: bernoulli_logit_regression(theta, x, y)

mle = MLE(loglike)

start = tm.time()
mle.fit(y, n_theta=x.shape[1])
end = tm.time()
print(f"tensorflow took: {end - start:.2f} seconds")

n_plot = 20
theta_opt = mle.theta
theta_true_plot = theta_true[:n_plot]
theta_opt_plot = theta_opt[:n_plot]

fig = plt.figure()
plt.scatter(np.arange(1, theta_true_plot.shape[0] + 1), theta_true_plot, color='b')
plt.scatter(np.arange(1, theta_opt_plot.shape[0] + 1), theta_opt_plot, color='r')
for i, s in enumerate(np.sqrt(mle.theta_var[:n_plot])):
    plt.plot([i+1, i+1], [theta_opt_plot[i] - 1.96*s, theta_opt_plot[i] + 1.96*s], color='k')
