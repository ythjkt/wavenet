import librosa
from util import *

SAMPLING_RATE = 22050


def main():
    # Test inv_mulaw_quantize(mulaw_quantize()).
    file_path = 'data/feather.mp3'
    wav, sr = librosa.load(file_path, duration=60)
    # wav = librosa.effects.trim(wav, top_db=40)[0]
    # wav = librosa.util.normalize(wav) * 0.95
    # wav = mulaw_quantize(wav, mu = 10e5)
    # wav = inv_mulaw_quantize(wav, mu = 10e5)
    output_path = 'generated/feather.wav'

    save_wav(wav, output_path, sr)


if __name__ == '__main__':
    main()
