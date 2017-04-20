"""Trains and Evaluates the facial keypoints detection network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

import tensorflow as tf

NUM_LABELS = 30
IMAGE_SIZE = 96
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def fully_connected(input, size):
    """
    Create a fully connected TensorFlow layer.
    :param input: Input tensor for calculating layer shape.
    :param size: Layer size, e.g. number of units
    :return: A graph variable calculating single fully connected layer.
    """
    weights = tf.get_variable('weights',
                              shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('bias',
                           shape=[size],
                           initializer=tf.constant_initializer(0.0)
                           )
    return tf.matmul(input, weights) + bias


def inference(input, output_size=30):
    """
    Performs a whole model pass.
    :param input: Input tensor to be passed through the model.
    :return: Model prediction.  
    """
    with tf.variable_scope('hidden'):
        hidden = fully_connected(input, 100)
    relu_hidden = tf.nn.relu(hidden)
    with tf.variable_scope('out'):
        logits = fully_connected(relu_hidden, size=output_size)
    return logits


def loss(predictions, labels):
    """
    Calculates loss with Numpy.
    :param predictions: ndarray. Predictions.
    :param labels: ndarrya. Actual values.
    :return: Squared mean error for given predictions.
    """
    return tf.reduce_mean(tf.squared_difference(predictions, labels))


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op















