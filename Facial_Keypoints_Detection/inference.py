# network model for facial keypoints detection
# ==================================================================

"""
Builds the Facial keypoints detection network.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

import tensorflow as tf


def inference_one_hidden_layer(images, hidden_units, output_size):
    """
    Build the one hidden layer model up to where it may be used for inference. 
    :param images: Image placeholder, from inputs().
    :param hidden_units: Size of the hidden layer.(only one hidden layer here)
    :param output_size: Size of the output layer.
    :return: Output tensor.
    """
    # Hidden
    with tf.name_scope('hidden'):
        weights = tf.get_variable('weights',
                                  shape=[input.get_shape()[1], hidden_units],
                                  initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias',
                               shape=[hidden_units],
                               initializer=tf.constant_initializer(0.0)
                               )
        hidden = tf.nn.relu(tf.matmul(images, weights) + bias)

    # Linear
    with tf.name_scope('linear'):
        weights = tf.get_variable('weights',
                                  shape=[hidden_units, output_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias',
                               shape=[output_size],
                               initializer=tf.constant_initializer(0.0)
                               )
        output = tf.matmul(hidden, weights) + bias
    return output


def loss(predictions, labels):
    """
    Calculates loss with Numpy.
    :param predictions: ndarray. Predictions.
    :param labels: ndarrya. Actual values.
    :return: Squared mean error for given predictions.
    """
    return np.mean(np.square(predictions-labels))


















