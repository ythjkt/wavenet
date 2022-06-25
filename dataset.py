import os
import params
import tensorflow as tf


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes, {
            "wav":
                tf.io.FixedLenSequenceFeature(
                    [], dtype=tf.int64, allow_missing=True)
        })['wav']


def get_xy(wav):
    wav = tf.one_hot(wav, 256, axis=-1, dtype=tf.float32)
    x = wav[:-1, :]
    y = wav[1:, :]
    return x, y


def get_data(data_dir: str):
    """
    Loads train and test data stored in data_dir and returns as TFRecordDataSet.
    """
    train_data_path = os.path.join(data_dir, params.train_data_filename)
    test_data_path = os.path.join(data_dir, params.test_data_filename)

    train_data = tf.data.TFRecordDataset(train_data_path).shuffle(300).map(
        decode_fn).map(get_xy).batch(1).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_data = tf.data.TFRecordDataset(test_data_path).shuffle(300).map(
        decode_fn).map(get_xy).batch(1).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, test_data