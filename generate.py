from wavenet.model import WaveNet
import numpy as np
from black import out
import tensorflow as tf
from preprocess import mulaw_quantize
import json
import os
from scipy.io import wavfile

SAMPLING_RATE = 22050
GENERATE_LEN = 3 * SAMPLING_RATE


def generate():
    pass


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


def inv_mulaw(x, mu=255):
    return np.sign(x) * (1.0 / mu) * ((1.0 + mu) ** np.abs(x) - 1.0)


def inv_mulaw_quantize(x, mu=255):
    x = 2 * x.astype(np.float32) / mu - 1

    return inv_mulaw(x, mu)


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.0001, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def main():
    os.makedirs('./generated/', exist_ok=True)
    with open('wavenet_params.json', 'r') as f:
        wavenet_params = json.load(f)

    # Initialize WaveNet.
    wavenet = WaveNet(
        wavenet_params['dilations'],
        wavenet_params['filter_width'],
        wavenet_params['residual_channels'],
        wavenet_params['dilation_channels'],
        wavenet_params['skip_channels'], )
    weight_path = './results/weights/wavenet_0100'
    wavenet.load_weights(weight_path)
    initial_value = mulaw_quantize(0)
    inputs = tf.one_hot(indices=initial_value, depth=256, dtype=tf.float32)
    inputs = tf.reshape(inputs, [1, 1, 256])

    outputs = []
    for i in range(10):
        x = wavenet.predict(inputs)[:, -1, :]
        x = tf.expand_dims(x, axis=1)
        x = tf.argmax(x, axis=-1)
        x = tf.one_hot(indices=x, depth=256)
        x = tf.reshape(x, [1, 1, 256])
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

    save_wav(outputs, './generated/audio.wav', SAMPLING_RATE)


if __name__ == '__main__':
    main()
    print('genedated')
