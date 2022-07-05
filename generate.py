import os
import time
import params
import argparse
from util import *
from wavenet.model import WaveNet
import tensorflow as tf


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet train script')
    parser.add_argument('weights_path',
                        type=str,
                        help="path to weights of the model")
    parser.add_argument('output_dir',
                        type=str,
                        help="path to directory for generated audio file")
    parser.add_argument('-l',
                        '--length',
                        type=int,
                        default=5,
                        help='length of the generated audio in seconds')
    parser.add_argument(
        '-r',
        '--sampling_rate',
        type=int,
        default=params.sampling_rate,
        help="should match the sampling rate used with preproces.py")
    args = parser.parse_args()

    if args.length < 0:
        raise ValueError('length needs to be positive')

    if args.sampling_rate < 0:
        raise ValueError('sampling_rate needs to be positive')

    return args


def print_progress_bar(iteration,
                       total,
                       frequency=10,
                       prefix='',
                       fill='#',
                       length=50,
                       print_end='\r'):
    if iteration % frequency != 0 and iteration != total:
        return
    percent = 100 * iteration / float(total)
    fill_len = int(length * iteration // total)
    bar = fill * fill_len + '-' * (length - fill_len)
    print(f'\r{prefix} {bar} : {percent: .2f}', end=print_end)
    if iteration == total:
        print()


def main():
    args = get_arguments()

    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)

    # Initialize WaveNet.
    wavenet = WaveNet(params.dilations, params.filter_width,
                      params.residual_channels, params.dilation_channels,
                      params.skip_channels)

    start_time = time.time()
    wavenet.load_weights(args.weights_path)

    initial_value = mulaw_quantize(0)
    input = tf.one_hot(indices=initial_value, depth=256, dtype=tf.float32)
    input = tf.reshape(input, [1, 1, 256])
    outputs = []
    generate_len = args.length * args.sampling_rate
    with tf.device('/cpu:0'):
        for i in range(generate_len):
            input = wavenet.call(input, is_generate=True)
            index = tf.argmax(input, axis=-1)
            input = tf.reshape(tf.one_hot(indices=index, depth=256),
                               [1, 1, 256])
            outputs.append(tf.squeeze(tf.argmax(input, axis=-1)))
            print_progress_bar(i + 1, generate_len)

    outputs = tf.concat(outputs, axis=0)
    end_time = time.time()
    print(
        f"Generating {generate_len} took {end_time - start_time:.0f} seconds.")

    outputs = outputs.numpy()
    outputs = inv_mulaw_quantize(outputs)
    output_path = os.path.join(args.output_dir,
                               f'audio-{time.strftime("%Y%m%d-%H%M%S")}.wav')
    save_wav(outputs, output_path, args.sampling_rate)


if __name__ == '__main__':
    main()