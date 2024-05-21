import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.layers_list = []

        for i in range(num_layers):
            self.layers_list.append(self._make_layer(growth_rate, kernel_size))

    def _make_layer(self, growth_rate, kernel_size):
        """Construct a conv-batchnorm-relu block."""
        layers = tf.keras.Sequential([
            BatchNormalization(),
            ReLU(),
            Conv1D(growth_rate, kernel_size, padding='same', use_bias=False)
        ])
        return layers

    def call(self, x):
        for layer in self.layers_list:
            out = layer(x)
            x = tf.concat([x, out], axis=-1)  # Concatenate output from previous layers
        return x
    
class DenseTransition(tf.keras.layers.Layer):
    def __init__(self, compression_factor, pool_size=3):
        super(DenseTransition, self).__init__()
        self.compression_factor = compression_factor
        self.pool_size = pool_size

    def call(self, x):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        out_filters = int(tf.keras.backend.int_shape(x)[-1] * self.compression_factor)
        x = Conv1D(out_filters, 1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.AvgPool1D(pool_size=self.pool_size, strides=2)(x)
        return x