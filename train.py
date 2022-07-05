from __future__ import annotations

import os
import re
import argparse
from wavenet.model import WaveNet
from dataset import get_data
import params
import tensorflow as tf

CHECKPOINT_PATH = r'wavenet_([0-9]+)\.ckpt'
SUMMARY_DIR = 'summary'


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet train script')
    parser.add_argument(
        'input_dir',
        type=str,
        help=
        "path to directory containing train and test dataset in tfrecord format"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="path to directory for storing checkpoints and weights")
    parser.add_argument('-e',
                        '--epoch',
                        type=int,
                        default=1000,
                        help='number of epochs to train the model for')
    parser.add_argument(
        '-l',
        '--load_checkpoint',
        action='store_true',
        help='continue training from the latest checkpoint in input_dir if any')
    parser.add_argument('-i',
                        '--save_interval',
                        type=int,
                        default=20,
                        help='interval at which checkpoint should be saved')
    parser.add_argument('-r',
                        '--learning_rate',
                        type=float,
                        default=1e-3,
                        help='learning rate used with Adam optimizer')
    args = parser.parse_args()

    if args.learning_rate < 0:
        raise ValueError('learning rate needs to be positive')

    return args


@tf.function
def train_step(model, x, y, loss_object, optimizer, loss_metric,
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
    args = get_arguments()

    weights_dir = os.path.join(args.output_dir, params.weights_dir)
    checkpoint_dir = os.path.join(args.output_dir, params.checkpoint_dir)
    summary_dir = os.path.join(args.output_dir, SUMMARY_DIR)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # Initialize WaveNet.
    model = WaveNet(params.dilations, params.filter_width,
                    params.residual_channels, params.dilation_channels,
                    params.skip_channels)

    # Initialize loss function and optimizer function.
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Initialize metrics.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # Load weights from the latest checkpoint if it exists.
    current_epoch = 0
    if args.load_checkpoint:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        try:
            print(f'Loading weights at {latest}')
            model.load_weights(latest)
            current_epoch = int(re.compile(CHECKPOINT_PATH).findall(latest)[0])
        except:
            print(f'Failed to load weights at {latest}')

    summary_writer = tf.summary.create_file_writer(summary_dir)

    print("Start training WaveNet.")
    step = 0
    train_data, test_data = get_data(args.input_dir)
    for epoch in range(current_epoch, args.epoch):
        # Reset all metrics every epoch.
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

        for x, y in test_data:
            validate_step(model, x, y, loss_object, test_loss, test_accuracy)

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir,
                                           f'wavenet_{epoch:05d}.ckpt')
            model.save_weights(checkpoint_path)

        print(f'Epoch {epoch + 1} '
              f'Loss {train_loss.result():20.19f} '
              f'Test Loss {test_loss.result():20.19f} '
              f'Accuracy {train_accuracy.result():20.19f} '
              f'Test Accuracy {test_accuracy.result():20.19f}')

    print(f"Done training for {epoch+1} epoch.")
    weights_path = os.path.join(weights_dir, f'wavenet_{epoch+1:05d}')
    print(f"Saving weights to {weights_path}")
    model.save_weights(weights_path)


if __name__ == '__main__':
    main()
