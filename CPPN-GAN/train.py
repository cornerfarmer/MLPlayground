import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


rand_input_size = 1
generator_steps_per_iteration = 5
batch_size = 500
iterations = 100000

class CPPN:
    def __init__(self, sess):
        with tf.variable_scope("generator"):
            self.x = tf.placeholder(tf.float32, (None, 784, rand_input_size + 3))

            layer_input = self.x
            for i in range(4):
                layer_input = tf.layers.dense(layer_input, 128, tf.nn.tanh, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

            self.y = tf.layers.dense(layer_input, 1, tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

            #self.y = tf.slice(self.x, [0, 0, 1], [-1, -1, 1]) / 10

            self.sess = sess

    def predict(self, x):
        return self.sess.run(self.y, {self.x: x})

    def create_input(self, x_vec, y_vec):
        z = np.random.uniform(-1.0, 1.0, size=(1, rand_input_size)).repeat(len(y_vec) * len(x_vec), 0)

        input = np.transpose([np.tile(x_vec, len(y_vec)), np.repeat(y_vec, len(x_vec))])

        i_t = input.transpose()
        r = np.sqrt(i_t[0] * i_t[0] + i_t[1] * i_t[1]).reshape((len(i_t[0]), 1))

        return np.concatenate((input, r, z), axis=1)

    def create_batch_input(self, x_vec, y_vec, batch_size):
        batch = np.ndarray((batch_size, len(y_vec) * len(x_vec), rand_input_size + 3))
        for i in range(batch_size):
            batch[i] = self.create_input(x_vec, y_vec)
        return batch

class MLPGenerator:
    def __init__(self, sess):
        with tf.variable_scope("generator"):
            self.x = tf.placeholder(tf.float32, (None, rand_input_size))

            layer_input = self.x
            layer_input = tf.layers.dense(layer_input, 128, tf.nn.relu)

            self.y = tf.layers.dense(layer_input, 784, tf.nn.sigmoid)

            self.sess = sess

    def predict(self, x):
        return self.sess.run(self.y, {self.x: x})

    def create_batch_input(self, x_vec, y_vec, batch_size):
       return np.random.uniform(-1.0, 1.0, size=(batch_size, rand_input_size))

class MLPDiscriminator:
    def __init__(self, sess, x, reuse):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            layer_input = tf.reshape(x, (-1, 28 * 28))
            layer_input = tf.layers.dense(layer_input, 128, tf.nn.relu)

            self.y = tf.layers.dense(layer_input, 1)

            self.sess = sess


class Discriminator:
    def __init__(self, sess, x, reuse):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            conv1 = tf.layers.conv2d(x, 24, (5, 5), padding="same", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)
            conv2 = tf.layers.conv2d(pool1, 24 * 2, [5, 5], padding="same", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)
            conv3 = tf.layers.conv2d(pool2, 24 * 4, [5, 5], padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, [2, 2], 2)

            pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 24 * 4])
            self.y = tf.layers.dense(pool3_flat, 1)

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
        self.discriminator_fake = MLPDiscriminator(self.sess, x, False)

        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels, logits=self.discriminator_fake.y))
        self.discriminator_loss_false = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.false_labels, logits=self.discriminator_fake.y))

        vars = [var for var in tf.trainable_variables() if "generator/" in var.name]
        self.gen_optimizer = tf.train.AdamOptimizer(0.005, name="AdamGenerator")
        self.train_step_gen = self.gen_optimizer.minimize(self.generator_loss, var_list=vars)

    def _build_discriminator(self):
        self.d_input = tf.placeholder(tf.float32, (None, 28 * 28))
        x = tf.reshape(self.d_input, [-1, 28, 28, 1])
        tf.summary.image("discriminator", x, 1)
        self.discriminator_orig = MLPDiscriminator(self.sess, x, True)

        self.discriminator_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_labels, logits=self.discriminator_orig.y))
        self.discriminator_loss = self.discriminator_loss_false + self.discriminator_loss_true
        self.discriminator_loss /= 2

        vars = [var for var in tf.trainable_variables() if "discriminator/" in var.name]
        self.train_step_dis = tf.cond(tf.logical_and(self.generator_loss < 0.8, self.discriminator_loss > 0.45), lambda: tf.train.AdamOptimizer(0.001, name="AdamDiscriminator").minimize(self.discriminator_loss, var_list=vars), lambda: tf.constant(False))
        #self.train_step_dis = tf.train.AdamOptimizer(0.001, name="AdamDiscriminator").minimize(self.discriminator_loss, var_list=vars)



    def train(self, x_vec, y_vec):
        train_writer = tf.summary.FileWriter('tensorboard/' + time.strftime("%Y%m%d-%H%M%S"), self.sess.graph)
        tf.summary.scalar("discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("discriminator_loss_true", self.discriminator_loss_true)
        tf.summary.scalar("discriminator_loss_false", self.discriminator_loss_false)
        tf.summary.scalar("generator_loss", self.generator_loss)
        tf.summary.scalar("generator_learning_rate", self.gen_optimizer._lr_t)

        merged_summary_op = tf.summary.merge_all()

        images = np.ndarray((1, 784))
        labels = np.ndarray((1, 10))
        new_i = 0
        for i in range(len(mnist.train.labels)):
            if mnist.train.labels[i][0] > 0:
                images[new_i] = mnist.train.images[i] * 255
                labels[new_i] = mnist.train.labels[i]
                new_i += 1
                break

        mnist_ones = DataSet(images, labels, reshape=False)
       # mnist.train.images = images
       # mnist.train.labels = labels

        for i in range(iterations * generator_steps_per_iteration):
            if i % generator_steps_per_iteration == 0:
                batch = mnist_ones.next_batch(batch_size)
                _, _, summary = self.sess.run([self.train_step_gen, self.train_step_dis, merged_summary_op], feed_dict={self.d_input: batch[0], self.generator.x: self.generator.create_batch_input(x_vec, y_vec, batch_size)})
                train_writer.add_summary(summary, i / generator_steps_per_iteration)
                print("Finished step " + str(i / generator_steps_per_iteration))
            else:
                _ = self.sess.run([self.train_step_gen], feed_dict={self.generator.x: self.generator.create_batch_input(x_vec, y_vec, batch_size)})


    def generate_image(self, x_vec, y_vec, size):
        return self.generator.predict(self.generator.create_batch_input(x_vec, y_vec, 1).astype(np.float32)).reshape((size, size))

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
