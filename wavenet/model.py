import tensorflow as tf

print("TensorFlow version:", tf.__version__)


class MyConv1D(tf.keras.layers.Conv1D):

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding='causal',
        dilation_rate=1,
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
        )
        self.cache_queue = None

    def call(self, inputs, is_generate=False):
        if (self.kernel_size[0] != 2 or not is_generate):
            return super().call(inputs)

        return self.generate(inputs)

    def generate(self, inputs):
        state = self.cache_queue.dequeue()
        self.cache_queue.enqueue(inputs)
        inputs = tf.concat([state, inputs], axis=1)
        return tf.nn.convolution(inputs, self.kernel) + self.bias

    def build(self, input_shape):
        super().build(input_shape)

        self.init_queue(input_shape[0], input_shape[2])

    def init_queue(self, batch_size, input_channels):
        self.cache_queue = tf.queue.FIFOQueue(capacity=self.dilation_rate[0],
                                              dtypes=tf.float32)
        self.cache_queue.enqueue_many(
            tf.zeros(shape=(self.dilation_rate[0], batch_size, 1,
                            input_channels),
                     dtype=tf.float32))


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        kernel_size,
        residual_channels,
        dilation_channels,
        skip_channels,
        dilation_rate=1,
    ):
        super(ResidualBlock, self).__init__()

        self.tanh_conv1D = MyConv1D(
            dilation_channels,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
        )
        self.sig_conv1D = MyConv1D(dilation_channels,
                                   kernel_size,
                                   dilation_rate=dilation_rate,
                                   padding="causal")
        self.res_output = MyConv1D(residual_channels, 1, padding="same")
        self.skip_conv = MyConv1D(skip_channels, 1, padding="same")

    def call(self, inputs, is_generate=False):
        res = inputs
        tanh_out = self.tanh_conv1D(inputs, is_generate)
        sig_out = self.sig_conv1D(inputs, is_generate)
        tanh_out = tf.nn.tanh(tanh_out)
        sig_out = tf.nn.sigmoid(sig_out)
        merged = tf.multiply(tanh_out, sig_out)
        res_out = self.res_output(merged, is_generate)
        skip_out = self.skip_conv(merged, is_generate)
        res_out = tf.add(res_out, res)

        return res_out, skip_out


class WaveNet(tf.keras.Model):

    def __init__(self, dilations, kernel_size, residual_channels,
                 dilation_channels, skip_channels):
        super(WaveNet, self).__init__()

        self.kernel_size = kernel_size
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.caucal_conv = tf.keras.layers.Conv1D(residual_channels,
                                                  kernel_size=1,
                                                  padding='causal')
        self.residual_blocks = []
        for d in dilations:
            self.residual_blocks.append(
                ResidualBlock(kernel_size,
                              residual_channels,
                              dilation_channels,
                              skip_channels,
                              dilation_rate=d))

        self.conv1 = MyConv1D(128, kernel_size=1, padding="causal")
        self.conv2 = MyConv1D(256, kernel_size=1, padding="causal")

    def call(self, inputs, is_generate=False):
        x = self.caucal_conv(inputs)
        skip = None
        for residual_block in self.residual_blocks:
            x, h = residual_block(x, is_generate)
            if skip is None:
                skip = h
            else:
                skip = skip + h
        x = skip
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if not is_generate:
            x = tf.nn.softmax(x)

        return x
