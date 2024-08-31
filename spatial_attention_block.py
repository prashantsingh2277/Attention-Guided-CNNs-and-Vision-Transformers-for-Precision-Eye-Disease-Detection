
import tensorflow as tf
from tensorflow.keras import layers

class SpatialAttentionBlock2D(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttentionBlock2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, (self.kernel_size, self.kernel_size), padding='same', activation='sigmoid')

    def call(self, inputs):
        attention = self.conv(inputs)
        return inputs * attention

    def get_config(self):
        config = super(SpatialAttentionBlock2D, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
