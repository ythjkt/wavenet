import os
from venv import create
import librosa
import numpy as np
import tensorflow as tf

RESULT_DIR = 'results'
SAMPLING_RATE = 22050
# Length of a trainign data used in training.
TRAIN_LEN = 5 * SAMPLING_RATE
HOP_SIZE = 2 * SAMPLING_RATE


def sliding_window(wav):
    wavs = []
    if wav.shape[0] < TRAIN_LEN:
        pass


def mulaw_quantize(x, mu=255):
    x = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)
    x = (x + 1.0) * mu / 2.0
    return x.astype(int)


def pad(wav):
    assert wav.shape[0] <= TRAIN_LEN

    # Pad with 0.
    return np.pad(wav, (0, TRAIN_LEN - wav.shape[0]), mode='constant')


def split_wav(wav):
    splits = []
    for i in range(0, wav.shape[0] - TRAIN_LEN + 1, HOP_SIZE):
        splits.append(wav[i:i+TRAIN_LEN])

    # splits[-1] = pad(splits[-1])
    return splits


def make_example(wav):
    return tf.train.Example(features=tf.train.Features(feature={
        'wav': tf.train.Feature(int64_list=tf.train.Int64List(value=wav))
    }))


def preprocess_audio(file_path):
    wav, _ = librosa.load(file_path, sr=SAMPLING_RATE, duration=10)
    wav = librosa.effects.trim(wav, top_db=40)[0]
    wav = librosa.util.normalize(wav) * 0.95
    wav = mulaw_quantize(wav)
    wavs = split_wav(wav)

    for idx, wav in enumerate(wavs):
        if wav.shape[0] != TRAIN_LEN:
            print(
                f"Wrong shape at idx {idx} with shape {wav.shape[0]} and wavs.len {len(wavs)}")

    return [make_example(wav) for wav in wavs]


def create_tfrecord():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with tf.io.TFRecordWriter(os.path.join(RESULT_DIR, 'train_data.tfrecord')) as writer:
        records = preprocess_audio('data/Vocoder.mp3')
        for record in records:
            writer.write(record.SerializeToString())


if __name__ == '__main__':
    create_tfrecord()
