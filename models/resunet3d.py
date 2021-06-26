from keras.layers import Input, Dropout, concatenate, MaxPool3D, Conv3D
from keras.models import Model


from custom_layers import conv_single, residual, up_conv
from config import dropout


class resunet3d(tf.keras.Model):
    def __init__(self, pool_size=(2, 2, 2), outc_init=32):
        super(resunet3d, self).__init__()
        outc = outc_init
        self.pool_size = pool_size

        self.conv1a = conv_single(out_channels=outc)
        self.res1a = residual(out_channels=outc)
        self.maxp1 = MaxPool3D(pool_size=pool_size)
        self.dp1a = Dropout(dropout)

        self.conv2a = conv_single(out_channels=outc * 2)
        self.res2a = residual(out_channels=outc * 2)
        self.maxp2 = MaxPool3D(pool_size=pool_size)
        self.dp2a = Dropout(dropout)

        self.conv3a = conv_single(out_channels=outc * 4)
        self.res3a = residual(out_channels=outc * 4)
        self.maxp3 = MaxPool3D(pool_size=pool_size)
        self.dp3a = Dropout(dropout)

        self.conv4a = conv_single(out_channels=outc * 8)
        self.res4a = residual(out_channels=outc * 8)

        self.upconv1 = up_conv(out_channel=outc * 4)
        self.dp1b = Dropout(dropout)
        self.conv1b = conv_single(out_channels=outc * 4)
        self.res1b = residual(out_channels=outc * 4)

        self.upconv2 = up_conv(out_channel=outc * 2)
        self.dp2b = Dropout(dropout)
        self.conv2b = conv_single(out_channels=outc * 2)
        self.res2b = residual(out_channels=outc * 2)

        self.upconv3 = up_conv(out_channel=outc)
        self.dp3b = Dropout(dropout)
        self.conv3b = conv_single(out_channels=outc)
        self.res3b = residual(out_channels=outc)

        self.softmax = Conv3D(
            filters=4, kernel_size=(1, 1, 1), padding="same", activation="softmax"
        )

    def call(self, x):

        enc1 = self.conv1a(x)  # 64x64x64x32
        enc1 = self.res1a(enc1)  # 64x64x64x32

        mp1 = self.maxp1(enc1)  # 32x32x32x32
        dp1a = self.dp1a(mp1)  # 32x32x32x32

        enc2 = self.conv2a(dp1a)  # 32x32x32x64
        enc2 = self.res2a(enc2)  # 32x32x32x64

        mp2 = self.maxp2(enc2)  # 16x16x16x64
        dp2a = self.dp2a(mp2)  # 16x16x16x64

        enc3 = self.conv3a(dp2a)  # 16x16x16x128
        enc3 = self.res3a(enc3)  # 16x16x16x128

        mp3 = self.maxp3(enc3)  # 8x8x8x128
        dp3a = self.dp3a(mp3)  # 8x8x8x128

        enc4 = self.conv4a(dp3a)  # 8x8x8x256
        enc4 = self.res4a(enc4)  # 8x8x8x256

        up1 = self.upconv1(enc4)  # 16x16x16x128
        merge1 = concatenate([up1, enc3], axis=4)
        # [16x16x16x128 + 16x16x16x128]
        # 16x16x16x256

        dp1b = self.dp1b(merge1)  # 16x16x16x256

        dec1 = self.conv1b(dp1b)  # 16x16x16x128
        dec1 = self.res1b(dec1)  # 16x16x16x128

        up2 = self.upconv2(dec1)  # 32x32x32x64
        merge2 = concatenate([up2, enc2], axis=4)
        # 32x32x32x128

        dp2b = self.dp2b(merge2)  # 32x32x32x128

        dec2 = self.conv2b(dp2b)  # 32x32x32x64
        dec2 = self.res2b(dec2)  # 32x32x32x64

        up3 = self.upconv3(dec2)
        merge3 = concatenate([up3, enc1], axis=4)

        dp3b = self.dp3b(merge3)

        dec3 = self.conv3b(dp3b)
        dec3 = self.res3b(dec3)

        final_softmax = self.softmax(dec3)
        return final_softmax

    def model(self):
        x = Input((128, 128, 128, 4))
        return Model(inputs=[x], outputs=self.call(x))
