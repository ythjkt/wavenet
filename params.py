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
result_dir = "results"
# TODO: Remove these two paths once train.py is also updated.
train_data_dir = "train_data"
data_dir = "data"

epoch = 10000
mu = 256