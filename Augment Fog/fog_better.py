# -*- coding: utf-8 -*-
import sys
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from dataset import DatasetInitializer
lib_path = '/home/n2202864a/Downloads/Foggy-CycleGAN-master/lib'
os.environ['DRIVE_PROJECT'] = lib_path


datasetInit = DatasetInitializer(256, 256)


def plot_clear2fog_intensity(model_clear2fog, image_clear, intensity = 0.8,
                             normalized_input=True, close_fig=False):
    import matplotlib.pyplot as plt
    import tensorflow as tf

    original_intensity = intensity
    if normalized_input:
        intensity = intensity * 2 - 1
    intensity = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(intensity), 0), 0)
    prediction_clear2fog = model_clear2fog((tf.expand_dims(image_clear, 0), intensity))

    fig = plt.figure(figsize=(12, 6))

    display_list = [prediction_clear2fog[0]]
    title = ['To Fog {:0.2}'.format(original_intensity)]

    for i in range(1):
        #plt.subplot(1, 1, i + 1)
        #plt.title(title[i])
        to_display = display_list[i]
        if normalized_input:
            to_display = to_display * 0.5 + 0.5
        plt.imshow(to_display)
        plt.axis('off')

    if close_fig:
        plt.close(fig)
    return fig


def gauss_blur_model(input_shape, kernel_size=19, sigma=5, **kwargs):
    import tensorflow as tf
    import numpy as np
    def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        #https://stackoverflow.com/questions/55643675/how-do-i-implement-gaussian-blurring-layer-in-keras
        #https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python/17201686#17201686
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    class SymmetricPadding2D(tf.keras.layers.Layer):
        # Source: https://stackoverflow.com/a/55210905/11394663
        def __init__(self, output_dim, padding=(1, 1),
                     data_format="channels_last", **kwargs):
            self.output_dim = output_dim
            self.data_format = data_format
            self.padding = padding
            super(SymmetricPadding2D, self).__init__(**kwargs)

        def build(self, input_shape):
            super(SymmetricPadding2D, self).build(input_shape)

        def call(self, inputs, **kwargs):
            if self.data_format is "channels_last":
                # (batch, depth, rows, cols, channels)
                pad = [[0, 0]] + [[i, i] for i in self.padding] + [[0, 0]]
            # elif self.data_format is "channels_first":
            else:
                # (batch, channels, depth, rows, cols)
                pad = [[0, 0], [0, 0]] + [[i, i] for i in self.padding]
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "REFLECT")
            return out

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.output_dim

    if kernel_size % 2 == 0:
        raise Exception("kernel size should be an odd number")
    gauss_inputs = tf.keras.layers.Input(shape=input_shape)

    kernel_weights = matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=sigma)
    in_channels = input_shape[-1]
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)  # apply the same filter on all the input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons
    gauss_layer = tf.keras.layers.DepthwiseConv2D(kernel_size, use_bias=False, padding='valid')
    p = (kernel_size - 1) // 2
    # noinspection PyCallingNonCallable
    x = SymmetricPadding2D(0, padding=[p, p])(gauss_inputs)
    x = gauss_layer(x)
    ########################
    gauss_layer.set_weights([kernel_weights])
    gauss_layer.trainable = False
    return tf.keras.Model(inputs=gauss_inputs, outputs=x, **kwargs)

  
def create_dir(path):
    """
    Creates a path recursively if it doesn't exist
    :param path: The specified path
    :return: None
    """
    if path is None or path == '':
        return
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise Exception("Not a valid path: {}".format(path))

        

class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

        
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
      
      
    def call(self, x, **kwargs):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset



class ModelsBuilder:
    def __init__(self, output_channels=3, image_height=256, image_width=256, normalized_input=True):
        self.output_channels = output_channels
        self.image_height = image_height
        self.image_width = image_width
        self.normalized_input = normalized_input

        
    def downsample(self, filters, size, norm_type='instancenorm', apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(tf.keras.layers.BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

      
    def upsample(self, filters, size, norm_type='instancenorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

      
    def resize_conv(self, filters, kernel_size, strides=1, norm_type='instancenorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.UpSampling2D(2, interpolation='bilinear'))
        result.add(
            tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.LeakyReLU())

        return result

      
    def concatenate_image_and_intensity(self, image_input, intensity_input):
        intensity = tf.keras.layers.RepeatVector(self.image_height * self.image_height)(intensity_input)
        intensity = tf.keras.layers.Reshape((self.image_height, self.image_height, 1))(intensity)
        return tf.keras.layers.Concatenate(axis=-1)([image_input, intensity])

      
    def build_generator(self, use_transmission_map=False, use_gauss_filter=True, norm_type='instancenorm',
                        use_intensity=True, kernel_size=4,
                        use_resize_conv=False):
        image_input = tf.keras.layers.Input(shape=[self.image_height, self.image_height, self.output_channels])
        inputs = image_input
        x = image_input
        if use_intensity:
            intensity_input = tf.keras.layers.Input(shape=(1,))
            inputs = [image_input, intensity_input]
            x = self.concatenate_image_and_intensity(x, intensity_input)

        down_stack = [
            self.downsample(64, kernel_size, norm_type=norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            self.downsample(128, kernel_size, norm_type=norm_type),  # (bs, 64, 64, 128)
            self.downsample(256, kernel_size, norm_type=norm_type),  # (bs, 32, 32, 256)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 16, 16, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 8, 8, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 4, 4, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 2, 2, 512)
            self.downsample(512, kernel_size, norm_type=norm_type),  # (bs, 1, 1, 512)
        ]

        if use_resize_conv:
            up_stack = [
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
                self.resize_conv(512, kernel_size, norm_type=norm_type),  # (bs, 16, 16, 1024)
                self.resize_conv(256, kernel_size, norm_type=norm_type),  # (bs, 32, 32, 512)
                self.resize_conv(128, kernel_size, norm_type=norm_type),  # (bs, 64, 64, 256)
                self.resize_conv(64, kernel_size, norm_type=norm_type),  # (bs, 128, 128, 128)
            ]
        else:
            up_stack = [
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
                self.upsample(512, kernel_size, norm_type=norm_type),  # (bs, 16, 16, 1024)
                self.upsample(256, kernel_size, norm_type=norm_type),  # (bs, 32, 32, 512)
                self.upsample(128, kernel_size, norm_type=norm_type),  # (bs, 64, 64, 256)
                self.upsample(64, kernel_size, norm_type=norm_type),  # (bs, 128, 128, 128)
            ]

        initializer = tf.random_normal_initializer(0., 0.02)

        last = tf.keras.layers.Conv2DTranspose(1 if use_transmission_map else self.output_channels,
                                               kernel_size,
                                               strides=2,
                                               padding='same',
                                               name='transmission_layer' if use_transmission_map else 'output_layer',
                                               kernel_initializer=initializer,
                                               activation='tanh' if self.normalized_input else 'sigmoid')  # (bs, 256, 256, 1)
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        if use_transmission_map:
            transmission = x
            if self.normalized_input:
                transmission = tf.keras.layers.Lambda(lambda t: t * 0.5 + 0.5, name='fix_transmission_range')(
                    transmission)
            if use_gauss_filter:
                #gauss_ = 
                transmission = gauss_blur_model([self.image_height, self.image_width, 1], name="gauss_blur")(
                    transmission)

            x = tf.keras.layers.multiply([image_input, transmission])
            one_minus_t = tf.keras.layers.Lambda(lambda t: 1 - t, name='transmission_invert')(transmission)
            x = tf.keras.layers.add([x, one_minus_t])

        return tf.keras.Model(inputs=inputs, outputs=x)

      
    def build_discriminator(self, norm_type='instancenorm', use_intensity=True, kernel_size=4):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        inputs = inp
        x = inp
        if use_intensity:
            intensity_input = tf.keras.layers.Input(shape=(1,))
            inputs = [inp, intensity_input]
            x = self.concatenate_image_and_intensity(x, intensity_input)

        down1 = self.downsample(64, kernel_size, norm_type=norm_type, apply_norm=False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, kernel_size, norm_type=norm_type)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, kernel_size, norm_type=norm_type)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, kernel_size, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
        if norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)
        else:
            norm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, kernel_size, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
        return tf.keras.Model(inputs=inputs, outputs=last)

use_transmission_map = True #@param{type: "boolean"}
use_gauss_filter = True #@param{type: "boolean"}
use_resize_conv = True #@param{type: "boolean"}

modelsBuilder = ModelsBuilder()
generator_clear2fog = modelsBuilder.build_generator(use_transmission_map=use_transmission_map,
                                                     use_gauss_filter=use_gauss_filter,
                                                     use_resize_conv=use_resize_conv)
generator_fog2clear = modelsBuilder.build_generator(use_transmission_map=False)

intensity_path = '/home/n2202864a/Downloads/Foggy-CycleGAN-master/output'
create_dir(intensity_path)
path='/home/n2202864a/Downloads/Foggy-CycleGAN-master/input'
for image in os.listdir(path):
    file_path = f'/home/n2202864a/Downloads/Foggy-CycleGAN-master/input/{image}'

    image_clear = tf.io.decode_png(tf.io.read_file(file_path), channels=3)
    image_clear, _ = datasetInit.preprocess_image_test(image_clear, 0)
    #plt.showfig()
    step = 0.05
    for (ind, i) in enumerate(tf.range(0,1+step, step)):
        fig = plot_clear2fog_intensity(generator_clear2fog,image_clear)
        #fig.savefig("/content/gdrive/MyDrive/out2")
        fig.savefig(os.path.join(intensity_path
                             , f"{image}"), bbox_inches='tight', pad_inches=0)
        os.chdir('/home/n2202864a/Downloads/Foggy-CycleGAN-master/output')
        plt.savefig(f'{image}',bbox_inches='tight',pad_inches=0)


