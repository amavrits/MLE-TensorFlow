from tensorflow_probability import distributions as tfd
import tensorflow as tf
import time as tm
import matplotlib.pyplot as plt
import numpy as np
from MLE_TensorFlow import MLE_TF

class Linear_Regression(MLE_TF):

    def __init__(self, y):
        self.y = y
        self.n_theta = y.shape[1] + 1

    def loglikehood(self, theta, y):
        x_data, y_data = y[:, 0], y[:, 1]
        beta = theta[:-1]
        sigma = theta[-1]
        y_hat = beta[0] + beta[1] * x_data
        dist = tfd.Normal(loc=y_hat, scale=1e-5 + sigma)
        loglike = tf.reduce_sum(dist.log_prob(y_data), axis=0)
        def true_fn(): return loglike
        def false_fn(): return -np.inf
        return tf.cond(tf.math.reduce_all(sigma > 0), true_fn, false_fn)


    def set_prediction_model(self, theta):
        def prediction_model(x):
            y_hat = theta[0] + theta[1] * x
            pred_model = tfd.Normal(loc=y_hat, scale=theta[2])
            return pred_model
        return prediction_model



try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

N = 100
x = np.random.uniform(0, 10, N)
y = 2 + 5 * x + np.random.randn(N) * 0.5
data = np.c_[x, y]
data = tf.cast(data, tf.float32)

mle = Linear_Regression(data)
start = tm.time()
mle.MLE_fit()
end = tm.time()
print(f"tensorflow took: {end - start:.2f} seconds")

x_new = np.linspace(x.min(), x.max(), 1_000)
y_model_new = 2 + 5 * x_new
predictive_model = mle.MLE_prediction_model(x=tf.cast(x_new, tf.float32))
MLE_y_hat = predictive_model.loc.numpy()
MLE_predictive_samples = predictive_model.sample(1).numpy()

start = tm.time()
bootstrap_samples = mle.set_non_parameteric_bootstrapp_sample(n_bootstrap=10)
mle.bootstrapping(bootstrap_samples=bootstrap_samples)
end = tm.time()
print(f"tensorflow bootstrapping took: {end - start:.2f} seconds")


fig = plt.figure()
plt.scatter(x_new, MLE_predictive_samples, c='b', alpha=0.2, label='Predictions')
plt.scatter(x, y, c='r', alpha=0.8, label='Data')
plt.plot(x_new, MLE_y_hat, c='k', linewidth=2, label='MLE model')
plt.plot(x_new, y_model_new, c='r', label='True model')
plt.xlabel('Dependent variable', fontsize=14)
plt.ylabel('Independent variable', fontsize=14)
plt.legend(fontsize=12)

fig, axs = plt.subplots(1, mle.n_theta+1)
for i, ax in enumerate(axs[:3]):
    ax.hist(mle.theta_boot[:, i], color='b', alpha=0.8, density=True)
    ax.axvline(mle.theta_hat[i], color='r')
    ax.set_xlabel('Estimator '+str(i+1), fontsize=14)
    if i == 0:
        ax.set_ylabel('Density [-]', fontsize=14)
    else:
        ax.yaxis.set_visible(False)
axs[3].hist(mle.loglike_boot, color='b', alpha=0.8, density=True)
axs[3].axvline(mle.MLE_loglike, color='r')
axs[3].set_xlabel('Log-likelihood', fontsize=14)
