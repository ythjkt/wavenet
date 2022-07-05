# WaveNet paramenters.
filter_width = 2
sampling_rate = 16000
dilations = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
]
residual_channels = 32
dilation_channels = 32
quantization_channels = 256
skip_channels = 512

# Other parameters.
train_data_filename = "train_data.tfrecord"
test_data_filename = "test_data.tfrecord"
weights_dir = 'weights'
checkpoint_dir = 'ckpts'
epoch = 10000
mu = 256