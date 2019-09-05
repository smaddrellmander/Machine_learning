import tensorflow as tf
from utils import upsample, downsample

"""Sub class the Generator as well."""


class Generator_(tf.keras.Model):
    """docstring for Generator_."""
    def __init__(self,):
        super(Generator_, self).__init__()
        self.OUTPUT_CHANNELS = 3
        self.downsample = [  downsample(64, 4, apply_batchnorm=False),
                           downsample(128, 4),  # (bs, 64, 64, 128)
                           downsample(256, 4),  # (bs, 32, 32, 256)
                           downsample(512, 4),  # (bs, 16, 16, 512)
                           downsample(512, 4),  # (bs, 8, 8, 512)
                           downsample(512, 4),  # (bs, 4, 4, 512)
                           downsample(512, 4),  # (bs, 2, 2, 512)
                           downsample(512, 4)]  # (bs, 1, 1, 512)

        self.upsample = [upsample(512, 4, apply_dropout=True),  # (bs,2,2,1024)
                         upsample(512, 4, apply_dropout=True),  # (bs,4,4,1024)
                         upsample(512, 4, apply_dropout=True),  # (bs,8,8,1024)
                         upsample(512, 4),  # (bs, 16, 16, 1024)
                         upsample(256, 4),  # (bs, 32, 32, 512)
                         upsample(128, 4),  # (bs, 64, 64, 256)
                         upsample(64, 4)    # (bs, 128, 128, 128)
                         ]

        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS, 4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')  # (bs, 256, 256, 3)
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=False):
        """Define the forward pass."""
        x = inputs
        # print(x.shape)
        # Downsampling through the model
        skips = []
        for down in self.downsample:
            x = down(x)
            skips.append(x)
            # print(x.shape)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.upsample, skips):
            x = up(x)
            x = self.concat([x, skip])
            # print(x.shape)
        return self.last(x)
