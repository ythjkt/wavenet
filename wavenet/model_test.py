import tensorflow as tf
import numpy as np
from model import MyConv1D


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


class WaveNetTest(tf.test.TestCase):
    pass


tf.test.main()