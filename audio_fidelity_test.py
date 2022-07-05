import librosa
from util import *
import params

SAMPLING_RATE = 22050


def main():
    # Test inv_mulaw_quantize(mulaw_quantize()).
    sr = params.sampling_rate
    file_path = 'data/feather_dot.wav'
    wav, sr = librosa.load(file_path, duration=2, sr=sr)
    wav = librosa.effects.trim(wav, top_db=40)[0]
    wav = librosa.util.normalize(wav) * 0.95
    wav = mulaw_quantize(wav)
    wav = inv_mulaw_quantize(wav)
    output_path = 'generated/feather_dot.wav'

    save_wav(wav, output_path, sr)


if __name__ == '__main__':
    main()
