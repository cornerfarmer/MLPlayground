import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import numpy as np

def cnn_model_fn(features, labels, mode):
    with tf.name_scope("network"):
        h = tf.layers.dense(features, 3, tf.nn.sigmoid)
        y = tf.layers.dense(h, 1, tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels, y)

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), 0.1, "Adam")

    predictions = y

    return model_fn_lib.ModelFnOps(mode=mode, loss=loss, predictions=predictions, train_op=train_op)


estimator = learn.SKCompat(learn.Estimator(model_fn=cnn_model_fn, model_dir="test"))

data = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
labels = np.asarray([[[0.0]], [[1.0]], [[1.0]], [[0.0]]], dtype=np.float32)

tf.logging.set_verbosity(tf.logging.INFO)
estimator.fit(x=data, y=labels, steps=2000)

eval_results = estimator.score(x=data, y=labels)
print(eval_results)

x = np.asarray([[0.0, 1.0]], dtype=np.float32)
print(estimator.predict(x))

