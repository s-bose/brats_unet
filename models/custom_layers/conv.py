import tensorflow as tf
from keras.layers import Input, Conv3D, BatchNormalization, Activation

from keras.models import Model


class conv_single(tf.keras.Model):
    def __init__(
        self,
        out_channels,
        kernel_size=(3, 3, 3),
        padding="same",
        kernel_initializer="he_normal",
        **kwargs
    ):
        super(conv_single, self).__init__(**kwargs)
        self.conv1 = Conv3D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_initializer,
        )
        self.relu1 = Activation("relu")
        self.bn1 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        return x

    def model(self):
        x = Input((64, 64, 64, 4))
        return Model(inputs=[x], outputs=self.call(x))
