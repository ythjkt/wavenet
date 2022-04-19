import os
import tensorflow as tf
import numpy as np
from wavenet.model import WaveNet
import json
from dataset import get_train_data

EPOCH = 2
NUM_CHANNEL = 128
SAVE_INTERVAL = 1


@tf.function
def train_step(model, x, y, loss_object, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

#   "filter_width": 2,
#   "sample_rate": 16000,
#   "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
#                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
#                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
#                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
#                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
#   "residual_channels": 32,
#   "dilation_channels": 32,
#   "quantization_channels": 256,
#   "skip_channels": 512
# }


def train():
    os.makedirs("./results/" + "weights/", exist_ok=True)
    with open('wavenet_params.json', 'r') as f:
        wavenet_params = json.load(f)

    # Initialize WaveNet.
    wavenet = WaveNet(
        wavenet_params['dilations'],
        wavenet_params['filter_width'],
        wavenet_params['residual_channels'],
        wavenet_params['dilation_channels'],
        wavenet_params['skip_channels'], )

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    summary_writer = tf.summary.create_file_writer('./results')
    step = 0
    for epoch in range(EPOCH):
        train_data = get_train_data()
        for x, y in train_data:
            loss = train_step(wavenet, x, y, loss_object, optimizer)

            with summary_writer.as_default():
                tf.summary.scalar('train/loss', loss, step=step)

            if step % 5 == 0:
                print(f"Step {step}")

            step += 1

        if epoch % SAVE_INTERVAL == 0:
            print(f"Step {step}, loss {loss}")
            np.save('./results/' + f"weights/step.npy", np.array(step))
            wavenet.save_weights('./results/' + f"weights/wavenet_{epoch:04}")

    print("training done")
    np.save('./results/' + f"weights/step.npy", np.array(step))
    wavenet.save_weights('./results/' + f"weights/wavenet_{epoch:04}")


if __name__ == '__main__':
    train()
