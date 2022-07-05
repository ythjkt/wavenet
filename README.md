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

## Generating audio.
generate.py generates audio iteratively by making the model predict the value of
the next timestamp. It starts with an arbitrary timestamp 1 and make the model
predict the next timestamp. Then it appends the output of the model to the input
makind the model generate one timestamp at a time.

By default the script generates an audio clip of 5 seconds.

```
# Run python3 train.py --help to see all the options.
python3 generate.py <weights_path> <output_dir>
```

## Running unit tests.
Make sure these tests pass after modifying model.py or util.py.
```
pytest test_util.py
python3 wavenet/model_test.py
```