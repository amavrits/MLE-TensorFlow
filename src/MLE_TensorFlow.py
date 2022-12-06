import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import numpy as np

class Bootstrapping:

    def set_non_parameteric_bootstrapp_sample(self, n_bootstrap=10_000, seed=None):

        if seed is not None:
            np.random.seed(seed)

        idx_bootstrap_samples = np.random.choice(np.arange(self.y.shape[0]),
                                                 replace=True, size=(n_bootstrap, self.y.shape[0]))
        idx_bootstrap_samples = tf.cast(idx_bootstrap_samples, tf.int32)
        bootstrap_samples = tf.gather(self.y, idx_bootstrap_samples).numpy()

        return bootstrap_samples

    def bootstrapping(self, bootstrap_samples=None):

        self.n_bootstrap = bootstrap_samples.shape[0]
        self.bootstrap_samples = bootstrap_samples
        bootstrap_samples = tf.cast(bootstrap_samples, tf.float32)

        if not self.MLE_fitted:
            self.MLE_fit()

        self.theta_boot = np.zeros((self.n_bootstrap, self.n_theta))
        self.loglike_boot = np.zeros(self.n_bootstrap)
        for i, bootstrap_sample in enumerate(bootstrap_samples):
            self.fit(y=bootstrap_sample, init=tf.cast(self.theta_hat, tf.float32), max_iterations=100_000, tol=1e-10)
            self.theta_boot[i] = self.opt.position.numpy()
            self.loglike_boot[i] = self.loglikehood(self.opt.position, bootstrap_sample)

    def hypothesis_testing(self):
        pass

    def confidence_interval(self, confidence_lvl=0.95):
        pass


class MLE_TF(Bootstrapping):

    def __init__(self):
        self.MLE_fitted = False

    @tf.function
    def loss_and_gradient(self, theta, y):
        return tfp.math.value_and_gradient(lambda theta: -self.loglikehood(theta, y), theta)

    @tf.function
    def eval_observed_fisher(self, theta):
        hessian = tf.hessians(
            ys=self.loglikehood(theta, self.y),
            xs=theta,
            gate_gradients=False,
            aggregation_method=None,
            name='hessians'
        )
        return -hessian[0]

    def fit(self, y, init=None, max_iterations=10_000, tol=1e-8):

        if init is None:
            init = tf.ones(self.n_theta)

        self.opt = tfp.optimizer.lbfgs_minimize(
            lambda theta: self.loss_and_gradient(theta, y),
            initial_position=init,
            max_iterations=max_iterations,
            stopping_condition=tfp.optimizer.converged_all,
            tolerance=tol
        )

    def get_MLE_estimator(self, get_asymptotics=True):

        self.theta_hat = self.opt.position.numpy()

        self.MLE_loglike = self.loglikehood(self.opt.position, self.y).numpy()

        if get_asymptotics:
            observed_fisher = self.eval_observed_fisher(self.theta_hat)
            self.theta_cov = np.linalg.inv(observed_fisher)
            self.theta_var = np.diag(self.theta_cov)

    def MLE_fit(self, init=None, max_iterations=10_000, tol=1e-8, get_asymptotics=False):

        self.fit(y=self.y, init=init, max_iterations=max_iterations, tol=tol)

        self.get_MLE_estimator(get_asymptotics=get_asymptotics)

        self.MLE_prediction_model = self.set_prediction_model(theta=self.theta_hat)

        self.MLE_fitted = True

    def wald_test(self):
        pass

    def likelihood_ratio_test(self):
        pass

class MLE_Keras(Bootstrapping):

    def __init__(self):
        self.MLE_fitted = False

    def MLE_fit(self, get_asymptotics=False, epochs=1_000, verbose=False, learning_rate=0.01):

        negloglik = lambda y, rv: -rv.log_prob(y)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss=negloglik)

        self.fit_model(epochs=epochs, verbose=verbose)

        # self.get_MLE_estimator(get_asymptotics=get_asymptotics)

        # self.MLE_prediction_model = self.set_prediction_model(theta=self.theta_hat)

        self.MLE_fitted = True

    def get_MLE_estimator(self, get_asymptotics=True):

        self.theta_hat = self.opt.position.numpy()

        self.MLE_loglike = self.loglikehood(self.opt.position, self.y).numpy()

        if get_asymptotics:
            observed_fisher = self.eval_observed_fisher(self.theta_hat)
            self.theta_cov = np.linalg.inv(observed_fisher)
            self.theta_var = np.diag(self.theta_cov)

    def wald_test(self):
        pass

    def likelihood_ratio_test(self):
        pass