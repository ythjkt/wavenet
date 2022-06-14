from wavenet.model import WaveNet
import time
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
GENERATE_LEN = SAMPLING_RATE // 10
CHECKPOINTS_DIR = './results/ckpts'  # TODO: Put this to params.py


def print_progress_bar(iteration,
                       total,
                       frequency=10,
                       prefix='',
                       fill='#',
                       length=50,
                       print_end='\r'):
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
    wavenet = WaveNet(params.dilations, params.filter_width,
                      params.residual_channels, params.dilation_channels,
                      params.skip_channels)
    latest = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
    start_time = time.time()
    with tf.device('/cpu:0'):
        wavenet.load_weights(latest)
        outputs = wavenet.generate(GENERATE_LEN,
                                   progress_callback=print_progress_bar)
        end_time = time.time()
        print(
            f"Generating {GENERATE_LEN} took {end_time - start_time:.0f} seconds."
        )

    with tf.device('/gpu:0'):
        wavenet.load_weights(latest)
        start_time = time.time()
        initial_value = mulaw_quantize(10)
        inputs = tf.one_hot(indices=initial_value, depth=256, dtype=tf.float32)
        inputs = tf.reshape(inputs, [1, 1, 256])
        outputs = []
        for i in range(GENERATE_LEN):
            x = wavenet(inputs)[:, -1, :]
            x = tf.expand_dims(x, axis=1)
            x = tf.argmax(x, axis=-1)
            x = tf.one_hot(indices=x, depth=256)
            x = tf.reshape(x, [1, 1, 256])
            outputs.append(tf.argmax(x, axis=-1).numpy().item())

            inputs = tf.concat((inputs, x), axis=1)

            if inputs.shape[1] > 1025:
                inputs = inputs[:, 1:, :]
            assert inputs.shape[1] <= 1025

            print_progress_bar(i, GENERATE_LEN)
            # print(outputs)
        end_time = time.time()
    print(
        f"Generating {GENERATE_LEN} took {end_time - start_time:.0f} seconds.")

    outputs = tf.argmax(outputs, axis=-1).numpy()
    print(outputs.shape)
    outputs = inv_mulaw_quantize(outputs)

    save_wav(outputs, f'./generated/audio-{time.strftime("%Y%m%d-%H%M%S")}.wav',
             SAMPLING_RATE)


if __name__ == '__main__':
    main()
    print('genedated')
