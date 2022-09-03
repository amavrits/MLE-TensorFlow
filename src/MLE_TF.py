import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import time as tm
import matplotlib.pyplot as plt
import numpy as np

class TF_MLE():
    def __init__(self, loglikehood):
        self.loglikehood = loglikehood

    @tf.function
    def loss_and_gradient(self, theta):
        return tfp.math.value_and_gradient(lambda theta: -self.loglikehood(theta, self.x), theta)

    @tf.function
    def eval_observed_fisher(self, theta):
        observed_fisher = tf.hessians(
            ys=self.loglikehood(theta, self.x),
            xs=theta,
            gate_gradients=False,
            aggregation_method=None,
            name='hessians'
        )
        return -observed_fisher[0]

    def fit(self, x, n_theta=None):
        if n_theta is None:
            self.n_theta = x.shape[1]
        else:
            self.n_theta = n_theta
        self.x = x
        init = tf.zeros(self.n_theta)
        opt = tfp.optimizer.lbfgs_minimize(
            lambda theta: self.loss_and_gradient(theta), init, max_iterations=1_000
        )
        self.opt = opt
        self.get_estimator()

    def get_estimator(self):
        self.theta = self.opt.position
        observed_fisher = self.eval_observed_fisher(self.theta)
        self.theta_cov = np.linalg.inv(observed_fisher)
        self.theta_var = np.diag(self.theta_cov)
        self.max_loglike = self.loglikehood(self.theta, self.x)

    def wald_test(self, theta_Ho):
        pass

if __name__ == '__main__':

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

    loglike = lambda theta, y: bernoulli_logit_regression(theta, x, y)
    mle = TF_MLE(loglike)

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
