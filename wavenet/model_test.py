import tensorflow as tf
from model import MyConv1D, ResidualBlock, WaveNet


class Conv1DTest(tf.test.TestCase):

    def setUp(self):
        super(Conv1DTest, self).setUp()

    def testConv1DOutputWithDilationOne(self):
        conv1d = MyConv1D(filters=3, kernel_size=2)
        x = tf.random.normal((2, 2, 4))
        y = conv1d(x)

        y0 = conv1d(tf.expand_dims(x[:, 0, :], axis=1), is_generate=True)
        y1 = conv1d(tf.expand_dims(x[:, 1, :], axis=1), is_generate=True)

        self.assertEqual(conv1d.cache_queue.size(), 1)
        self.assertAllClose(tf.expand_dims(y[:, 0, :], axis=1), y0)
        self.assertAllClose(tf.expand_dims(y[:, 1, :], axis=1), y1)
        self.assertAllClose(y, tf.concat([y0, y1], 1))

    def testConv1DOutputWithDilationFour(self):
        conv1d = MyConv1D(filters=3, kernel_size=2, dilation_rate=4)
        x = tf.random.normal((2, 8, 4))
        y = conv1d(x)
        outputs = []
        for i in range(8):
            outputs.append(
                conv1d(tf.expand_dims(x[:, i, :], axis=1), is_generate=True))

        self.assertEqual(conv1d.cache_queue.size(), 4)
        self.assertAllClose(y, tf.concat(outputs, axis=1))


class ResidualBlockTest(tf.test.TestCase):

    def setUp(self):
        super(ResidualBlockTest, self).setUp()

    def testResidualBlockGenerate(self):
        kernel_size = 2
        residual_channels = 12
        dilation_channels = 12
        skip_channels = 20
        time_len = 30
        residual_block = ResidualBlock(kernel_size,
                                       residual_channels,
                                       dilation_channels,
                                       skip_channels,
                                       dilation_rate=2)
        x = tf.random.normal((1, time_len, residual_channels)) * 10
        y_res, y_skip = residual_block(x)

        self.assertEqual(y_res.shape, (1, time_len, residual_channels))
        self.assertEqual(y_skip.shape, (1, time_len, skip_channels))

        for i in range(time_len):
            input = tf.expand_dims(x[:, i, :], axis=1)
            yi_res, yi_skip = residual_block(input, is_generate=True)
            self.assertAllClose(y_res[:, i, :], yi_res[:, 0, :])
            self.assertAllClose(y_skip[:, i, :], yi_skip[:, 0, :])


class WaveNetTest(tf.test.TestCase):

    def setUp(self):
        super(WaveNetTest, self).setUp()
        tf.random.set_seed(1)

    def testWaveNetFastGenerate(self):
        # Test that fast generation yields the same result as normal call to
        # `WaveNet`.
        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        kernel_size = 2
        residual_channels = 12
        dilation_channels = 12
        skip_channels = 24
        time_len = 500

        input = tf.one_hot(indices=127, depth=256, dtype=tf.float32)
        input = tf.reshape(input, [1, 1, 256])

        wavenet = WaveNet(dilations, kernel_size, residual_channels,
                          dilation_channels, skip_channels)

        # Generate output with normal call to the model.
        output_slow = []
        inputs = input
        for _ in range(time_len):
            x = wavenet(inputs)[:, -1, :]
            index = tf.argmax(x, axis=-1)
            x = tf.one_hot(indices=index, depth=256)
            x = tf.reshape(x, [1, 1, 256])
            output_slow.append(index.numpy().item())
            inputs = tf.concat((inputs, x), axis=1)

            # `sum(dilations) + 1` is the length of the receptive field of
            # `wavenet`.
            if inputs.shape[1] > sum(dilations) + 1:
                inputs = inputs[:, 1:, :]

        # Generate output using `is_generate=True`.
        output_fast = []
        x = input
        for _ in range(time_len):
            x = wavenet(x, is_generate=True)
            index = tf.argmax(x, axis=-1)
            x = tf.one_hot(indices=index, depth=256)
            x = tf.reshape(x, [1, 1, 256])
            output_fast.append(index.numpy().item())

        self.assertAllEqual(output_slow, output_fast)


tf.test.main()
