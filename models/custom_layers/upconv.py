import tensorflow as tf
from keras.layers import Input, Conv3D, UpSampling3D

from keras.models import Model


class up_conv(tf.keras.Model):
    def __init__(self, out_channel, kernel_size=2, ups_size=(2, 2, 2)):
        super(up_conv, self).__init__()
        self.outc = out_channel
        self.ks = kernel_size
        self.usize = ups_size

        self.upsample3D = UpSampling3D(size=self.usize)
        self.conv3D = Conv3D(
            filters=self.outc, kernel_size=self.ks, padding="same", activation="relu"
        )

    def call(self, x):
        x = self.upsample3D(x)
        x = self.conv3D(x)
        return x

    def model(self):
        x = Input((16, 16, 16, 256))
        return Model(inputs=[x], outputs=self.call(x))
