"""Neural network and loss primitives for the spatially-varying-diffusion PINN.

Ported from PINNs_SVD.ipynb without behavioral changes.
"""
from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


class NeuralNet(object):
    def __init__(self, *inputs, layers):
        self.layers = layers
        self.num_layers = len(self.layers)

        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)

        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True, name=str(random.randint(1, 10000))))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))

    def get_trainable_variables(self):
        return self.weights + self.biases + self.gammas

    def __call__(self, *inputs):
        H = (tf.concat(inputs, 1) - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W / tf.norm(W, axis=0, keepdims=True)
            H = tf.matmul(H, V)
            H = g * H + b
            if l < self.num_layers - 2:
                H = H * tf.sigmoid(H)

        return tf.split(H, num_or_size_splits=H.shape[1], axis=1)


@tf.custom_gradient
def fwd_gradients(dy, dx):
    def grad(dy_dx):
        with tf.GradientTape() as tape:
            tape.watch(dx)
            G = tf.gradients(dy, dx, grad_ys=dy_dx)[0]
        return G

    return grad


def relative_error(pred, exact):
    if isinstance(pred, np.ndarray):
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact)) / tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))


def mean_squared_error(pred, exact):
    if isinstance(pred, np.ndarray):
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))
