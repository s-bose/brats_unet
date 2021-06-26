import tensorflow as tf
from keras.layers import Input, Activation, Add
from keras.models import Model

from .conv import conv_single


class residual(tf.keras.Model):
    def __init__(self, out_channels):
        super(residual, self).__init__()
        self.conv1 = conv_single(out_channels)
        self.conv2 = conv_single(out_channels)
        self.relu = Activation("relu")

    def call(self, x):
        pre = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = Add()([pre, x])
        x = self.relu(x)
        return x

    def model(self):
        x = Input((128, 128, 128, 32))
        return Model(inputs=[x], outputs=[self.call(x)])
