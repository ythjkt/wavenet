import tensorflow as tf


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        {"wav": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.int64, allow_missing=True)}
    )['wav']


def get_xy(wav):
    wav = tf.one_hot(wav, 256, axis=-1, dtype=tf.float32)
    x = wav[:-1, :]
    y = wav[1:, :]
    return x, y


def get_train_data():
    train_data = tf.data.TFRecordDataset(
        './train_data/train_data.tfrecord').map(decode_fn)\
        .map(get_xy).batch(3)\
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data
