import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import time as tm
import matplotlib.pyplot as plt
import numpy as np
from MLE_TensorFlow import MLE_Keras

class Linear_Regression(MLE_Keras):

    def __init__(self, y):
        self.y = y
        self.set_model()

    def set_model(self):
        model = tf.keras.Sequential([
            # tf.keras.layers.Dense(1),
            tf.keras.layers.Dense(1 + 1),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(tfd.Normal(loc=t[..., :1], scale=tf.nn.softplus(t[..., 1:]))))
        ])

        self.model = model

    def fit_model(self, epochs, verbose):
        x_data, y_data = self.y[:, 0, tf.newaxis], self.y[:, 1, tf.newaxis]
        self.model.fit(x_data, y_data, epochs=epochs, verbose=verbose)




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
x = np.random.uniform(0, 10, size=(N))
y = 2 + 5 * x + np.random.randn(N) * 0.5
data = np.stack((x, y), axis=1)
data = tf.cast(data, tf.float32)

mle = Linear_Regression(data)
start = tm.time()
mle.MLE_fit()
end = tm.time()
print(f"tensorflow took: {end - start:.2f} seconds")

x_new = np.linspace(0, x.max(), 1_000)
y_model_new = 2 + 5 * x_new
MLE_model = mle.model(tf.cast(x_new, tf.float32)[:, tf.newaxis])
MLE_y_hat = MLE_model.mean().numpy().squeeze()
MLE_st_error = MLE_model.stddev().numpy().squeeze()
MLE_prediction = MLE_model.sample().numpy().squeeze()

beta = mle.model.get_weights()

fig = plt.figure()
plt.scatter(x_new, MLE_prediction, c='b', alpha=0.2, label='Predictions')
plt.scatter(x, y, c='r', alpha=0.8, label='Data')
plt.plot(x_new, y_model_new, c='r', label='True model')
plt.plot(x_new, MLE_y_hat, c='k', linewidth=2, label='MLE model')
plt.xlabel('Dependent variable', fontsize=14)
plt.ylabel('Independent variable', fontsize=14)
plt.legend(fontsize=12)


# fig, axs = plt.subplots(1, mle.n_theta+1)
# for i, ax in enumerate(axs[:3]):
#     ax.hist(mle.theta_boot[:, i], color='b', alpha=0.8, density=True)
#     ax.axvline(mle.theta_hat[i], color='r')
#     ax.set_xlabel('Estimator '+str(i+1), fontsize=14)
#     if i == 0:
#         ax.set_ylabel('Density [-]', fontsize=14)
#     else:
#         ax.yaxis.set_visible(False)
# axs[3].hist(mle.loglike_boot, color='b', alpha=0.8, density=True)
# axs[3].axvline(mle.MLE_loglike, color='r')
# axs[3].set_xlabel('Log-likelihood', fontsize=14)
