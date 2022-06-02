import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Layer
from tensorflow.keras import Model

print("TensorFlow version:", tf.__version__)


class ResidualBlock(Layer):
    def __init__(
        self,
        kernel_size,
        residual_channels,
        dilation_channels,
        skip_channels,
        dilation_rate=1,
    ):
        super(ResidualBlock, self).__init__()

        self.tanh_conv1D = Conv1D(
            dilation_channels,
            kernel_size,
            dilation_rate=dilation_rate,
            activation="tanh",
            padding="causal",
        )
        self.sig_conv1D = Conv1D(
            dilation_channels,
            kernel_size,
            dilation_rate=dilation_rate,
            activation="sigmoid",
            padding="causal"
        )
        self.res_output = Conv1D(residual_channels, 1, padding="same")
        self.multiply = tf.keras.layers.Multiply()
        self.SkipConv = tf.keras.layers.Conv1D(
            skip_channels, 1, padding="same")
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        res = inputs
        tanh_out = self.tanh_conv1D(inputs)
        sig_out = self.sig_conv1D(inputs)
        merged = self.multiply((tanh_out, sig_out))
        res_out = self.res_output(merged)
        skip_out = self.SkipConv(merged)
        res_out = self.add((res_out, res))

        return res_out, skip_out


class WaveNet(Model):
    def __init__(
        self, dilations, kernel_size,
        residual_channels,
        dilation_channels,
        skip_channels
        # output_channels
    ):
        super(WaveNet, self).__init__()

        self.kernel_size = kernel_size
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        # self.output_channels = output_channels

        self.caucal_conv = tf.keras.layers.Conv1D(
            residual_channels, kernel_size=1, padding='causal')
        self.residual_blocks = []
        for d in dilations:
            self.residual_blocks.append(ResidualBlock(kernel_size,
                                                      residual_channels,
                                                      dilation_channels,
                                                      skip_channels,
                                                      dilation_rate=d))

        self.relu1 = tf.keras.layers.ReLU()
        self.conv1 = tf.keras.layers.Conv1D(128, kernel_size=1, padding="same")
        self.relu2 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv1D(
            256, kernel_size=1, padding="same", activation='softmax')

    def call(self, inputs):
        x = self.caucal_conv(inputs)
        skip = None
        for residual_block in self.residual_blocks:
            x, h = residual_block(x)
            if skip is None:
                skip = h
            else:
                skip = skip + h
        x = skip
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x
