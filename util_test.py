import librosa
from util import *

SAMPLING_RATE = 22050


def main():
    # Test inv_mulaw_quantize(mulaw_quantize()).
    file_path = 'data/Vocoder.mp3'
    wav, _ = librosa.load(file_path, sr=SAMPLING_RATE, duration=20)
    wav = librosa.effects.trim(wav, top_db=40)[0]
    wav = librosa.util.normalize(wav) * 0.95
    wav = mulaw_quantize(wav)
    wav = inv_mulaw_quantize(wav)
    output_path = 'generated/vocoder.wav'
    save_wav(wav, output_path, SAMPLING_RATE)


if __name__ == '__main__':
    main()
