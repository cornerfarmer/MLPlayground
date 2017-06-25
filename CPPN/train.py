import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio

class CPPN:
    def __init__(self):

        self.x = tf.placeholder(tf.float32, (None, 11))

        layer_input = self.x
        for i in range(4):
            layer_input = tf.layers.dense(layer_input, 32, tf.nn.tanh, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

        self.y = tf.layers.dense(layer_input, 1, tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, x):
        return self.sess.run(self.y, {self.x: x})


class Main:
    def __init__(self):
        self.ax = plt.gca()
        self.fig = plt.gcf()
        self.gif_saved = False

        self.fig.canvas.mpl_connect('button_release_event', self.show_next)
        plt.axis('off')

        self.size = 128
        self.start = -10.0
        self.end = 10.0

        self.model = CPPN()

        self.x = np.arange(self.start, self.end, (self.end - self.start) / self.size)
        self.y = np.arange(self.start, self.end, (self.end - self.start) / self.size)
        self.z_start = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(self.y) * len(self.x), 0)
        self.z_end = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(self.y) * len(self.x), 0)
        self.steps = 10.0
        self.step = 0
        self.image_data = dict()

        timer = self.fig.canvas.new_timer(interval=1)
        timer.add_callback(self.show_next, self.ax)
        timer.start()

        plt.show()

    def show_next(self, event):
        print("step %i" % self.step)

        if not self.step in self.image_data:
            z = self.z_start + (self.z_end - self.z_start) * self.step / self.steps

            input = np.transpose([np.tile(self.x, len(self.y)), np.repeat(self.y, len(self.x))])

            i_t = input.transpose()
            r = np.sqrt(i_t[0] * i_t[0] + i_t[1] * i_t[1]).reshape((len(i_t[0]), 1))

            input = np.concatenate((input, r, z), axis=1)

            self.image_data[self.step] = self.model.predict(input.astype(np.float32)).reshape((self.size, self.size))

        self.ax.imshow(self.image_data[self.step], cmap='gray', vmin=0, vmax=1, interpolation='nearest')

        self.ax.figure.canvas.draw()

        self.step += 1
        self.step %= self.steps

        if self.step == 0 and not self.gif_saved:
            images = []

            imageio.mimsave('output/' + time.strftime("%Y%m%d-%H%M%S") + '.gif', self.image_data.values())
            print("gif saved")
            self.gif_saved = True

Main()
