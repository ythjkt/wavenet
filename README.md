# WaveNet for Music Generation.

## Preprocesing data.
preprocess.py generates tfrecord from audio files.

First create a directory and put audio files to generate training data from.
Then pass that directory as the first argument to the script. Supported audio
formats are "wav" and "mp3".

```
# Run python3 preprocess.py --help to see all options.
# This creates tfrecord files in <output_dir>.
python3 preprocess.py [options] <input_dir> <output_dir>
```

## Training the model.
train.py trains WaveNet using tfrecord data created by preprocess.py and saves
the trained weights to disk.

```
# Run python3 train.py --help to see all the options.
python3 train.py <data dir> <result dir>
```