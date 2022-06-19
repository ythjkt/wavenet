from __future__ import annotations

import os
import re
import tensorflow as tf
import numpy as np
from wavenet.model import WaveNet
from dataset import get_train_data, get_test_data
import params

SAVE_INTERVAL = 20
RESULTS_DIR = './results/'
WEIGHTS_DIR = './results/weights/'
CHECKPOINTS_DIR = './results/ckpts'
LOAD_CHECKPOINT = True
CHECKPOINT_PATH = r'wavenet_([0-9]+)\.ckpt'


@tf.function
def train_step(model: tf.keras.Model, x, y, loss_object, optimizer, loss_metric,
               train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric.update_state(loss)
    train_accuracy.update_state(y, predictions)


@tf.function
def validate_step(model: tf.keras.Model, x, y, loss_object, loss_metric,
                  accuracy_metric):
    predictions = model(x)
    loss = loss_object(y, predictions)
    loss_metric.update_state(loss)
    accuracy_metric.update_state(y, predictions)


def main():
    print("Num GPUs available: ", len(tf.config.list_physical_devices('GPU')))
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Initialize WaveNet.
    model = WaveNet(params.dilations, params.filter_width,
                    params.residual_channels, params.dilation_channels,
                    params.skip_channels)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    current_epoch = 0
    if LOAD_CHECKPOINT:
        latest = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
        try:
            print(f'Loading weights at {latest}')
            model.load_weights(latest)
            current_epoch = int(re.compile(CHECKPOINT_PATH).findall(latest)[0])
        except:
            print(f'Failed to load weights at {latest}')

    summary_writer = tf.summary.create_file_writer(RESULTS_DIR)
    step = 0
    print("Start training WaveNet.")
    train_data = get_train_data()
    test_data = get_test_data()
    for epoch in range(current_epoch, params.epoch):
        train_loss.reset_state()
        test_loss.reset_state()
        train_accuracy.reset_state()
        test_accuracy.reset_state()
        for x, y in train_data:
            train_step(model, x, y, loss_object, optimizer, train_loss,
                       train_accuracy)

            with summary_writer.as_default():
                tf.summary.scalar('train/loss', train_loss.result(), step=step)
                tf.summary.scalar('train/accuracy',
                                  train_accuracy.result(),
                                  step=step)
            step += 1

        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR,
                                           f'wavenet_{epoch:05d}.ckpt')
            model.save_weights(checkpoint_path)

        for x, y in test_data:
            validate_step(model, x, y, loss_object, test_loss, test_accuracy)

        print(f'Epoch {epoch + 1} '
              f'Loss {train_loss.result():20.19f} '
              f'Test Loss {test_loss.result():20.19f} '
              f'Accuracy {train_accuracy.result():20.19f} '
              f'Test Accuracy {test_accuracy.result():20.19f}')

    print(f"Done training for {epoch} epoch.")
    model.save_weights(os.path.join(WEIGHTS_DIR, f'wavenet_{epoch+1:05d}'))


if __name__ == '__main__':
    main()
