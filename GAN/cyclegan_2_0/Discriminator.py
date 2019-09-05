import tensorflow as tf
from utils import downsample


class Discriminator_(tf.keras.Model):
    def __init__(self,):
        super(Discriminator_, self).__init__()
        """Define the layers used in the network."""
        initializer = tf.random_normal_initializer(0., 0.02)
        self.down1 = downsample(64, 4, False)
        self.down2 = downsample(128, 4)
        self.down3 = downsample(256, 4)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                           kernel_initializer=initializer)

    def call(self, inputs, target=None, training=False):
        """Calculate the forward pass through the network."""
        # print(inputs.shape, target.shape)
        if target is not None:
            x = tf.keras.layers.concatenate([inputs, target])
        else:
            x = inputs
        # print('discrim', x.shape)
        x = self.down1(x)
        # print('discrim', x.shape)
        x = self.down2(x)
        # print('discrim', x.shape)
        x = self.down3(x)
        # print('discrim', x.shape)
        x = self.zero_pad1(x)
        # print('discrim', x.shape)
        x = self.conv(x)
        # print('discrim', x.shape)
        x = self.batchnorm1(x)
        # print('discrim', x.shape)
        x = self.leaky_relu(x)
        # print('discrim', x.shape)
        x = self.zero_pad2(x)
        # print('discrim', x.shape)
        return self.last(x)



# -----------------------------
