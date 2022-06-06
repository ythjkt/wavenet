from wavenet.model import WaveNet
import numpy as np
import tensorflow as tf
from preprocess import mulaw_quantize
import time
import json
import os
from scipy.io import wavfile
import params
from util import *

SAMPLING_RATE = 22050
GENERATE_LEN = 2050
CHECKPOINTS_DIR = './results/ckpts' # TODO: Put this to params.py



def print_progress_bar(iteration, total, frequency=10, prefix='',
                       fill='#', length=50, print_end='\r'):
    if iteration % frequency != 0:
        return
    percent = 100 * iteration / float(total)
    fill_len = int(length * iteration // total)
    bar = fill * fill_len + '-' * (length - fill_len)
    print(f'\r{prefix} {bar} : {percent: .2f}', end=print_end)
    if iteration == total:
        print()


def main():
    os.makedirs('./generated/', exist_ok=True)

    # Initialize WaveNet.
    wavenet = WaveNet(
        params.dilations,
        params.filter_width,
        params.residual_channels,
        params.dilation_channels,
        params.skip_channels)
    latest = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
    wavenet.load_weights(latest)
    initial_value = mulaw_quantize(0)
    inputs = tf.one_hot(indices=initial_value, depth=256, dtype=tf.float32)
    inputs = tf.reshape(inputs, [1, 1, 256])
    # debug
    inputs = tf.one_hot(indices=10, depth=256, dtype=tf.float32)
    inputs = tf.reshape(inputs, [1, 1, 256])
    inputs = tf.repeat(inputs, 100, axis=1)
    print(inputs.shape)
    print(inputs)
    # debug
    outputs = []
    for i in range(GENERATE_LEN):
        # o = wavenet.predict(inputs, verbose=0)
        # print(o.shape)
        x = wavenet.predict(inputs, verbose=0)[:, -1, :]
        print('argmax!', tf.argmax(inputs, axis=-1))
        # print('-1', x.shape)
        x = tf.expand_dims(x, axis=1)
        # print('expand_dims', x.shape)
        x = tf.argmax(x, axis=-1)
        print('argmax', x)
        # print('argmax', x.shape)
        x = tf.one_hot(indices=x, depth=256)
        # print('one-Hot', x.shape)
        x = tf.reshape(x, [1, 1, 256])
        print(tf.math.argmax(x[0, 0]), f'step {i}')
        outputs.append(tf.argmax(x, axis=-1).numpy().item())

        inputs = tf.concat((inputs, x), axis=1)

        if inputs.shape[1] > 1024:
            inputs = inputs[:, 1:, :]

        assert inputs.shape[1] < 1025

        print_progress_bar(i, GENERATE_LEN)

    outputs = np.array(outputs)
    print(outputs)
    outputs = inv_mulaw_quantize(outputs)

    print(inv_mulaw_quantize(np.array(125)))

    print(outputs)

    save_wav(outputs, f'./generated/audio-{time.strftime("%Y%m%d-%H%M%S")}.wav', SAMPLING_RATE)


if __name__ == '__main__':
    main()
    print('genedated')
