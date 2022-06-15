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
GENERATE_LEN = SAMPLING_RATE * 2
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
    # latest = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
    start_time = time.time()
    wavenet.load_weights('results/weights/wavenet_01500')

    initial_value = mulaw_quantize(200)
    input = tf.one_hot(indices=initial_value, depth=256, dtype=tf.float32)
    input = tf.reshape(input, [1, 1, 256])
    outputs = []
    with tf.device('/cpu:0'):
        for i in range(GENERATE_LEN):
            input = wavenet.call(input, is_generate=True)
            outputs.append(tf.squeeze(tf.argmax(input, axis=-1)))
            print_progress_bar(i, GENERATE_LEN)

    outputs = tf.concat(outputs, axis=0)
    end_time = time.time()
    print(
        f"Generating {GENERATE_LEN} took {end_time - start_time:.0f} seconds.")

    outputs = outputs.numpy()
    outputs = inv_mulaw_quantize(outputs)
    save_wav(outputs, f'./generated/audio-{time.strftime("%Y%m%d-%H%M%S")}.wav',
             SAMPLING_RATE)


if __name__ == '__main__':
    main()
    print('genedated')
