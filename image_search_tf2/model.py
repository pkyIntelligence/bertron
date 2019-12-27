import tensorflow as tf

from utils.model import *


class ImageSearchModel(tf.keras.Model):

    def __init__(self, dropout_rate, image_size, number_of_classes=10):
        '''
        Defines CNN model.

        :param dropout_rate: dropout_rate
        :param learning_rate: learning_rate
        :param image_size: tuple, (height, width) of an image
        :param number_of_classes: integer, number of classes in a dataset.
        '''
        super(ImageSearchModel, self).__init__()

        self._bn_layer = tf.keras.layers.BatchNormalization()

        self._conv_block_1 = ConvBlock(number_of_filters=64,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._conv_block_2 = ConvBlock(number_of_filters=128,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._conv_block_3 = ConvBlock(number_of_filters=256,
                                       kernel_size=(5, 5),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._conv_block_4 = ConvBlock(number_of_filters=512,
                                       kernel_size=(5, 5),
                                       strides=(1, 1),
                                       padding='SAME',
                                       activation=tf.nn.relu,
                                       max_pool=True,
                                       batch_norm=True)

        self._flatten_layer = tf.keras.layers.Flatten()

        self._dense_block_1 = DenseBlock(units=128,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._dense_block_2 = DenseBlock(units=256,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._dense_block_3 = DenseBlock(units=512,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._dense_block_4 = DenseBlock(units=1024,
                                         activation=tf.nn.relu,
                                         dropout_rate=dropout_rate,
                                         batch_norm=True)

        self._final_dense = tf.keras.layers.Dense(units=number_of_classes,
                                                  activation=None)

        self._final_softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training):
        x = self._bn_layer(inputs, training)
        x, conv_1_features = self._conv_block_1(x, training)
        x, conv_2_features = self._conv_block_2(x, training)
        x, conv_3_features = self._conv_block_3(x, training)
        x, conv_4_features = self._conv_block_4(x, training)
        x = self._flatten_layer(x)
        x, dense_1_features = self._dense_block_1(x, training)
        x, dense_2_features = self._dense_block_2(x, training)
        x, dense_3_features = self._dense_block_3(x, training)
        x, dense_4_features = self._dense_block_4(x, training)
        x = self._final_dense(x)
        x = self._final_softmax(x)
        return x, dense_2_features, dense_4_features
