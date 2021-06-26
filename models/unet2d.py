import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Dropout,
)

from keras.models import Model

dropout = 0.2
hn = "he_normal"


def unet():

    inputs = Input((128, 128, 2))

    conv1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=hn)(
        inputs
    )
    conv1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv1
    )

    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=hn)(pool)
    conv = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=hn)(conv)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=hn)(
        pool1
    )
    conv2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv2
    )

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=hn)(
        pool2
    )
    conv3 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv3
    )

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=hn)(
        pool4
    )
    conv5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv5
    )
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv2D(256, 2, activation="relu", padding="same", kernel_initializer=hn)(
        UpSampling2D(size=(2, 2))(drop5)
    )
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=hn)(
        merge7
    )
    conv7 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv7
    )

    up8 = Conv2D(128, 2, activation="relu", padding="same", kernel_initializer=hn)(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=hn)(
        merge8
    )
    conv8 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv8
    )

    up9 = Conv2D(64, 2, activation="relu", padding="same", kernel_initializer=hn)(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = concatenate([conv, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=hn)(
        merge9
    )
    conv9 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=hn)(
        conv9
    )

    up = Conv2D(32, 2, activation="relu", padding="same", kernel_initializer=hn)(
        UpSampling2D(size=(2, 2))(conv9)
    )
    merge = concatenate([conv1, up], axis=3)
    conv = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=hn)(
        merge
    )
    conv = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=hn)(conv)

    conv10 = Conv2D(4, (1, 1), activation="softmax")(conv)

    model = Model(inputs=inputs, outputs=conv10)

    return model
