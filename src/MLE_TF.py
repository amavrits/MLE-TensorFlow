import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import numpy as np

class Bootstrapping:

    def nonparameteric_bootstrapping(self, n_bootstrap=10_000, seed=None):

        # if seed is not None:
        #     np.random.seed(seed)
        #
        # idx_bootstrap = np.random.choice(np.arange(self.x.shape[0]), replace=True, size=n_bootstrap)
        # bootstrap_samples = self.x[idx_bootstrap]
        #
        # self.theta_sample = np.zeros((n_bootstrap, self.n_theta))
        # for i, bootstrap_sample in enumerate(bootstrap_samples):
        #     self.fit(bootstrap_sample, n_theta=self.n_theta)
        #     self.theta_sample[i] = self.theta.numpy()

        pass

    def parameteric_bootstrapping(self, predictive_fn, n_bootstrap=10_000, seed=None):
        pass

    def hypothesis_testing(self):
        pass

    def confidence_interval(self, confidence_lvl=0.95):
        pass

class MLE(Bootstrapping):
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

    def fit(self, x, n_theta=None, init=None, max_iterations=1_000, tol=1e-8, get_asymptotics=True):

        if n_theta is None:
            self.n_theta = x.shape[1]
        else:
            self.n_theta = n_theta
        self.x = x

        if init is None:
            init = tf.ones(self.n_theta)

        self.opt = tfp.optimizer.lbfgs_minimize(
            lambda theta: self.loss_and_gradient(theta),
            initial_position=init,
            max_iterations=max_iterations,
            stopping_condition=tfp.optimizer.converged_all,
            tolerance=tol
        )
        self.get_estimator(get_asymptotics=get_asymptotics)

    def get_estimator(self, get_asymptotics=True):
        self.theta = self.opt.position
        self.max_loglike = self.loglikehood(self.theta, self.x)
        if get_asymptotics:
            observed_fisher = self.eval_observed_fisher(self.theta)
            self.theta_cov = np.linalg.inv(observed_fisher)
            self.theta_var = np.diag(self.theta_cov)

    def wald_test(self, theta_Ho):
        pass
