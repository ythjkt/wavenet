"""Generates tfrecord from audio files in data directory."""

import os
import librosa
import tensorflow as tf
from util import *
import params
import random

# Make sure that TRAIN_LEN > length of receptive field which is
# sum(dilations) + 1.
TRAIN_LEN = (sum(params.dilations) + 1) * 2
HOP_SIZE = TRAIN_LEN // 2


def split_wav(wav):
    splits = []
    for i in range(0, wav.shape[0] - TRAIN_LEN + 1, HOP_SIZE):
        splits.append(wav[i:i + TRAIN_LEN])

    return splits


def make_example(wav):
    return tf.train.Example(features=tf.train.Features(
        feature={
            'wav': tf.train.Feature(int64_list=tf.train.Int64List(value=wav))
        }))


def preprocess_audio(file_path: str):
    wav, _ = librosa.load(file_path, sr=params.sampling_rate, duration=20)
    wav, _ = librosa.effects.trim(wav, top_db=12)
    wav = librosa.util.normalize(wav) * 0.95
    wav = mulaw_quantize(wav)
    wavs = split_wav(wav)

    for idx, wav in enumerate(wavs):
        if wav.shape[0] != TRAIN_LEN:
            raise Exception(
                f"Wrong shape at idx {idx} with shape {wav.shape[0]}"
                "and wavs.len {len(wavs)}")

    return [make_example(wav) for wav in wavs]


def create_tfrecord(train_data_dir, raw_data_dir):
    os.makedirs(train_data_dir, exist_ok=True)
    train_data_filepath = os.path.join(train_data_dir, 'train_data.tfrecord')
    test_data_filepath = os.path.join(train_data_dir, 'test_data.tfrecord')
    with tf.io.TFRecordWriter(
            train_data_filepath) as train_writer, tf.io.TFRecordWriter(
                test_data_filepath) as test_writer:
        for p in os.listdir(raw_data_dir):
            filepath = os.path.join(raw_data_dir, p)
            if not os.path.isfile(filepath) or not p.endswith(('.mp3', '.wav')):
                continue
            records = preprocess_audio(filepath)

            random.shuffle(records)
            break_point = int(len(records) * params.train_test_split)
            for record in records[:break_point]:
                train_writer.write(record.SerializeToString())

            for record in records[break_point:]:
                test_writer.write(record.SerializeToString())

    print(f"Stored train data in {train_data_filepath}.")
    print(f"Stored test data in {test_data_filepath}.")


def main():
    create_tfrecord(params.train_data_dir, params.data_dir)


if __name__ == '__main__':
    main()
