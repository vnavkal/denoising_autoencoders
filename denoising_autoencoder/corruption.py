import abc

import tensorflow as tf


class Corrupter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def corrupt(self, X):
        ...


class GaussianCorrupter(Corrupter):
    def __init__(self, stddev):
        self._stddev = stddev

    def corrupt(self, X):
        return X + tf.random_normal(shape=tf.shape(X), stddev=self._stddev)


class MaskingCorrupter(Corrupter):
    def __init__(self, corruption_prob):
        self._corruption_prob = corruption_prob

    def corrupt(self, X):
        distribution = tf.distributions.Bernoulli(dtype=tf.float32, probs=self._corruption_prob)
        return X * (tf.ones_like(X) - distribution.sample(sample_shape=tf.shape(X)))
