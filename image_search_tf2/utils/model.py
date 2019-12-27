import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, number_of_filters, kernel_size, strides=(1, 1),
                 padding='SAME', activation=tf.nn.relu,
                 max_pool=True, batch_norm=True):
        '''
        Defines convolutional block layer.

        :param number_of_filters: integer, number of conv filters
        :param kernel_size: tuple, size of conv layer kernel
        :param padding: string, type of padding technique: SAME or VALID
        :param activation: tf.object, activation function used on the layer
        :param max_pool: boolean, if true the conv block will use max_pool
        :param batch_norm: boolean, if true the conv block will use batch normalization
        '''
        super(ConvBlock, self).__init__()

        self._conv_layer = tf.keras.layers.Conv2D(filters=number_of_filters,
                                                  kernel_size=kernel_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  activation=activation)

        self._max_pool = max_pool
        if max_pool:
            self._mp_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                       strides=(2, 2),
                                                       padding='SAME')

        self._batch_norm = batch_norm
        if batch_norm:
            self._bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):

        conv_features = x = self._conv_layer(inputs)
        if self._max_pool:
            x = self._mp_layer(x)
        if self._batch_norm:
            x = self._bn_layer(x, training)
        return x, conv_features


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, units, activation=tf.nn.relu,
                 dropout_rate=None, batch_norm=True):
        '''
        Defines dense block layer.

        :param units: integer, number of neurons/units for a dense layer
        :param activation: tf.object, activation function used on the layer
        :param dropout_rate: dropout rate used in this dense block
        :param batch_norm: boolean, if true the conv block will use batch normalization
        '''
        super(DenseBlock, self).__init__()

        self._dense_layer = tf.keras.layers.Dense(units, activation=activation)

        self._dropout_rate = dropout_rate
        if dropout_rate is not None:
            self._dr_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self._batch_norm = batch_norm
        if batch_norm:
            self._bn_layer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):

        dense_features = x = self._dense_layer(inputs)
        if self._dropout_rate is not None:
            x = self._dr_layer(x, training)
        if self._batch_norm:
            x = self._bn_layer(x, training)
        return x, dense_features
