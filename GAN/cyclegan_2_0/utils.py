import imageio
import os

import tensorflow as tf

# Util network functions

def downsample(filters, size, apply_batchnorm=True):
	"""Scale input image down by a factor of 2."""
	initializer = tf.random_normal_initializer(0., 0.02)
	result = tf.keras.Sequential()
	result.add(
		tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
							   kernel_initializer=initializer, use_bias=False))
	if apply_batchnorm:
	    result.add(tf.keras.layers.BatchNormalization())
	result.add(tf.keras.layers.LeakyReLU())
	return result

# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)

def upsample(filters, size, apply_dropout=False):
	"""Scale input image up by a factor of 2."""
	initializer = tf.random_normal_initializer(0., 0.02)
	result = tf.keras.Sequential()
	result.add(
    	tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    	padding='same',
                                    	kernel_initializer=initializer,
                                    	use_bias=False))
	result.add(tf.keras.layers.BatchNormalization())
	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))
	result.add(tf.keras.layers.ReLU())
	return result

#
#
# # Util processing / images functions.
# def create_gif(filenames, outname='default', duration=0.2):
# 	images = []
# 	for filename in filenames:
# 		images.append(imageio.imread(filename))
# 	output_file = 'results/{}.gif'.format(outname)
# 	imageio.mimsave(output_file, images, duration=duration)
