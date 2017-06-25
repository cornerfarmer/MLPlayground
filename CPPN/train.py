import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio


class CPPN:
    def __init__(self):
        self.ax = plt.gca()
        self.fig = plt.gcf()
        self.gif_saved = False

        self.fig.canvas.mpl_connect('button_release_event', self.show_next)
        plt.axis('off')

        self.estimator = learn.SKCompat(learn.Estimator(model_fn=self.cnn_model_fn))

        data = np.zeros((1, 11), dtype=np.float32)
        labels = np.asarray([[[0.0]]], dtype=np.float32)
        self.estimator.fit(data, labels, 1, 0)

        self.size = 1024
        self.start = -10.0
        self.end = 10.0

        self.x = np.arange(self.start, self.end, (self.end - self.start) / self.size)
        self.y = np.arange(self.start, self.end, (self.end - self.start) / self.size)
        self.z_start = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(self.y) * len(self.x), 0)
        self.z_end = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(self.y) * len(self.x), 0)
        self.steps = 100.0
        self.step = 0
        self.image_data = dict()

        timer = self.fig.canvas.new_timer(interval=1)
        timer.add_callback(self.show_next, self.ax)
        timer.start()

        plt.show()

    def cnn_model_fn(self, features, labels, mode):
        with tf.name_scope("network"):
            layer_input = features
            for i in range(4):
                layer_input = tf.layers.dense(layer_input, 32, tf.nn.tanh, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

            y = tf.layers.dense(layer_input, 1, tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

        loss = tf.losses.mean_squared_error(labels, y)

        train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), 0.1, "Adam")

        predictions = y

        return model_fn_lib.ModelFnOps(mode=mode, loss=loss, predictions=predictions, train_op=train_op)


    def show_next(self, event):
        print("step %i" % self.step)

        if not self.step in self.image_data:
            z = self.z_start + (self.z_end - self.z_start) * self.step / self.steps

            input = np.transpose([np.tile(self.x, len(self.y)), np.repeat(self.y, len(self.x))])

            i_t = input.transpose()
            r = np.sqrt(i_t[0] * i_t[0] + i_t[1] * i_t[1]).reshape((len(i_t[0]), 1))

            input = np.concatenate((input, r, z), axis=1)

            self.image_data[self.step] = self.estimator.predict(input.astype(np.float32)).reshape((self.size, self.size))

        self.ax.imshow(self.image_data[self.step], cmap='gray', vmin=0, vmax=1, interpolation='nearest')

        self.ax.figure.canvas.draw()

        self.step += 1
        self.step %= self.steps

        if self.step == 0 and not self.gif_saved:
            images = []

            imageio.mimsave('output/' + time.strftime("%Y%m%d-%H%M%S") + '.gif', self.image_data.values())
            print("gif saved")
            self.gif_saved = True

CPPN()
