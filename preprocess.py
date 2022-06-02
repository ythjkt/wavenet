"""Generates tfrecord from audio files in data directory."""

import os
import librosa
import tensorflow as tf
from util import *
import params

SAMPLING_RATE = 22050
# Length of a trainign data used in training.
TRAIN_LEN = SAMPLING_RATE
HOP_SIZE = SAMPLING_RATE // 2


def split_wav(wav):
    splits = []
    for i in range(0, wav.shape[0] - TRAIN_LEN + 1, HOP_SIZE):
        splits.append(wav[i:i+TRAIN_LEN])

    return splits


def make_example(wav):
    return tf.train.Example(features=tf.train.Features(feature={
        'wav': tf.train.Feature(int64_list=tf.train.Int64List(value=wav))
    }))


def preprocess_audio(file_path: str):
    wav, _ = librosa.load(file_path, sr=SAMPLING_RATE)
    wav = librosa.effects.trim(wav, top_db=40)[0]
    wav = librosa.util.normalize(wav) * 0.95
    wav = mulaw_quantize(wav)
    wavs = split_wav(wav)

    for idx, wav in enumerate(wavs):
        if wav.shape[0] != TRAIN_LEN:
            print(
                f"Wrong shape at idx {idx} with shape {wav.shape[0]} and"
                "wavs.len {len(wavs)}")

    return [make_example(wav) for wav in wavs]


def create_tfrecord(results_dir, data_dir):
    """

    """
    os.makedirs(results_dir, exist_ok=True)

    with tf.io.TFRecordWriter(os.path.join(results_dir, 'train_data.tfrecord')) as writer:
        for p in os.listdir(data_dir):
            file_path = os.path.join(data_dir, p)
            if not os.path.isfile(file_path) or not p.endswith(('.mp3', '.wav')):
                continue
            records = preprocess_audio(file_path)
            for record in records:
                writer.write(record.SerializeToString())


def main():
    create_tfrecord(params.train_data_dir, params.data_dir)


if __name__ == '__main__':
    main()
