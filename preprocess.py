"""Generates tfrecord from audio files in data directory."""

import os
import librosa
from util import *
import params
import random
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet preprocess script')
    parser.add_argument('input_dir',
                        type=str,
                        help="path to directory containing *.wav files")
    parser.add_argument(
        'output_dir',
        type=str,
        help="path to directory where tfrecord files is written to")
    parser.add_argument(
        '-s',
        "--split",
        type=float,
        default=0.9,
        help="split ratio between train and test data between 0 and 1")
    default_time_len = (sum(params.dilations) + 1) * 2
    parser.add_argument('--time_length',
                        type=int,
                        default=default_time_len,
                        help="time length of each training data")
    parser.add_argument('--hop_length',
                        type=int,
                        default=default_time_len // 2,
                        help='hop size when spliting the audio')
    parser.add_argument('-r',
                        '--sampling_rate',
                        type=int,
                        default=params.sampling_rate,
                        help="sampling rate used to load audio file")
    parser.add_argument('-d',
                        '--duration',
                        type=float,
                        default=None,
                        help="duration to load from each audio file in seconds")

    args = parser.parse_args()

    if args.split >= 1 or args.split <= 0:
        raise ValueError('split argument should be 0 < split < 1')
    if args.time_length < sum(params.dilations) + 1:
        raise ValueError(
            'time_length should be at least as long as the size of the'
            f'receptive field: {sum(params.dilations) + 1}')

    return args


def split_wav(wav: np.ndarray, time_length: int,
              hop_length: int) -> list[np.ndarray]:
    splits = []
    for i in range(0, wav.shape[0] - time_length + 1, hop_length):
        splits.append(wav[i:i + time_length])

    return splits


def make_example(wav: np.ndarray) -> tf.train.Example:
    return tf.train.Example(features=tf.train.Features(
        feature={
            'wav': tf.train.Feature(int64_list=tf.train.Int64List(value=wav))
        }))


def preprocess_audio(file_path: str, time_length: int, hop_length: int,
                     sampling_rate: int,
                     duration: float | None) -> list[tf.train.Example]:
    """Load audio file and split into training examples of time_length."""

    wav, _ = librosa.load(file_path,
                          sr=sampling_rate,
                          duration=duration,
                          mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=12)
    wav = librosa.util.normalize(wav) * 0.95
    wav = mulaw_quantize(wav)
    wavs = split_wav(wav, time_length, hop_length)

    for idx, wav in enumerate(wavs):
        if wav.shape[0] != time_length:
            raise Exception(
                f"Wrong shape at idx {idx} with shape {wav.shape[0]}"
                f"and wavs.len {len(wavs)}")

    return [make_example(wav) for wav in wavs]


def main():
    args = get_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data_filepath = os.path.join(args.output_dir, 'train_data.tfrecord')
    test_data_filepath = os.path.join(args.output_dir, 'test_data.tfrecord')
    with tf.io.TFRecordWriter(
            train_data_filepath) as train_writer, tf.io.TFRecordWriter(
                test_data_filepath) as test_writer:
        for p in os.listdir(args.input_dir):
            filepath = os.path.join(args.input_dir, p)
            if not os.path.isfile(filepath) or not p.endswith(('.mp3', '.wav')):
                continue
            records = preprocess_audio(filepath, args.time_length,
                                       args.hop_length, args.sampling_rate,
                                       args.duration)
            random.shuffle(records)
            break_point = int(len(records) * args.split)
            for record in records[:break_point]:
                train_writer.write(record.SerializeToString())

            for record in records[break_point:]:
                test_writer.write(record.SerializeToString())

    print(f"Stored train data in {train_data_filepath}.")
    print(f"Stored test data in {test_data_filepath}.")


if __name__ == '__main__':
    main()
