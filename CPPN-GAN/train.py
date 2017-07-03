import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class CPPN:
    def __init__(self, sess):
        with tf.variable_scope("generator"):
            self.x = tf.placeholder(tf.float32, (None, 784, 11))

            layer_input = self.x
            for i in range(4):
                layer_input = tf.layers.dense(layer_input, 32, tf.nn.tanh)

            self.y = tf.layers.dense(layer_input, 1, tf.nn.sigmoid)

            self.sess = sess

    def predict(self, x):
        return self.sess.run(self.y, {self.x: x})


class Discriminator:
    def __init__(self, sess, x, reuse):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            conv1 = tf.layers.conv2d(x, 32, (5, 5), padding="same", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)
            conv2 = tf.layers.conv2d(pool1, 64, [5, 5], padding="same", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            self.y = tf.layers.dense(pool2_flat, 1, None)

            self.sess = sess

class Models:
    def __init__(self):
        self.sess = tf.Session()

        self.true_labels = tf.ones(dtype=tf.float32, shape=[1, 1])
        self.false_labels = tf.zeros(dtype=tf.float32, shape=[1, 1])

        self._build_generator()
        self._build_discriminator()

        self.sess.run(tf.global_variables_initializer())

    def _build_generator(self):
        self.generator = CPPN(self.sess)
        x = tf.reshape(self.generator.y, [-1, 28, 28, 1])
        tf.summary.image("generator", x, 1)
        self.discriminator_fake = Discriminator(self.sess, x, False)

        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels, logits=self.discriminator_fake.y))
        self.discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.false_labels, logits=self.discriminator_fake.y))

        vars = [var for var in tf.trainable_variables() if "generator/" in var.name]
        self.train_step_gen = tf.train.AdamOptimizer(1e-4, name="AdamGenerator").minimize(self.generator_loss, var_list=vars)

    def _build_discriminator(self):
        self.d_input = tf.placeholder(tf.float32, (None, 28 * 28))
        x = tf.reshape(self.d_input, [-1, 28, 28, 1])
        self.discriminator_orig = Discriminator(self.sess, x, True)

        self.discriminator_loss = self.discriminator_loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels, logits=self.discriminator_orig.y))

        vars = [var for var in tf.trainable_variables() if "discriminator/" in var.name]
        self.train_step_dis = tf.train.AdamOptimizer(1e-4, name="AdamDiscriminator").minimize(self.discriminator_loss, var_list=vars)

    def _create_generator_input(self, x_vec, y_vec):
        z = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(y_vec) * len(x_vec), 0)

        input = np.transpose([np.tile(x_vec, len(y_vec)), np.repeat(y_vec, len(x_vec))])

        i_t = input.transpose()
        r = np.sqrt(i_t[0] * i_t[0] + i_t[1] * i_t[1]).reshape((len(i_t[0]), 1))

        return np.concatenate((input, r, z), axis=1)

    def _create_generator_batch_input(self, x_vec, y_vec, batch_size):
        batch = np.ndarray((batch_size, len(y_vec) * len(x_vec), 11))
        for i in range(batch_size):
            batch[i] = self._create_generator_input(x_vec, y_vec)
        return batch


    def train(self, x_vec, y_vec):
        train_writer = tf.summary.FileWriter('tensorboard/' + time.strftime("%Y%m%d-%H%M%S"), self.sess.graph)
        tf.summary.scalar("discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("generator_loss", self.generator_loss)

        merged_summary_op = tf.summary.merge_all()

        for i in range(100):
            batch = mnist.train.next_batch(500)
            _, _, summary = self.sess.run([self.train_step_gen, self.train_step_dis, merged_summary_op], feed_dict={self.d_input: batch[0], self.generator.x: self._create_generator_batch_input(x_vec, y_vec, 500)})

            train_writer.add_summary(summary, i)

            print("Finished step " + str(i))

    def generate_image(self, x_vec, y_vec, size):
        return self.generator.predict(self._create_generator_batch_input(x_vec, y_vec, 1).astype(np.float32)).reshape((size, size))

class Main:
    def __init__(self):
        self.ax = plt.gca()
        self.fig = plt.gcf()
        self.gif_saved = False

        self.fig.canvas.mpl_connect('button_release_event', self.show_next)
        plt.axis('off')

        self.size = 28
        self.start = -10.0
        self.end = 10.0

        self.x = np.arange(self.start, self.end, (self.end - self.start) / self.size)
        self.y = np.arange(self.start, self.end, (self.end - self.start) / self.size)
        self.z_start = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(self.y) * len(self.x), 0)
        self.z_end = np.random.uniform(-1.0, 1.0, size=(1, 8)).repeat(len(self.y) * len(self.x), 0)
        self.steps = 10.0
        self.step = 0
        self.image_data = dict()

        self.models = Models()
        self.models.train(self.x, self.y)

        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.show_next, self.ax)
        timer.start()

        plt.show()

    def show_next(self, event):
        print("step %i" % self.step)

        if not self.step in self.image_data:
            self.image_data[self.step] = self.models.generate_image(self.x, self.y, self.size)

        self.ax.imshow(self.image_data[self.step], cmap='gray', vmin=0, vmax=1, interpolation='nearest')

        self.ax.figure.canvas.draw()

        self.step += 1
        self.step %= self.steps

        if False and self.step == 0 and not self.gif_saved:
            images = []

            imageio.mimsave('output/' + time.strftime("%Y%m%d-%H%M%S") + '.gif', self.image_data.values())
            print("gif saved")
            self.gif_saved = True

Main()
