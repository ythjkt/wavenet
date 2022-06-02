import numpy as np
from scipy.io import wavfile
import json

MU_VALUE = 255

def mulaw(x: np.ndarray, mu = MU_VALUE) -> np.ndarray:
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

def mulaw_quantize(x: np.ndarray, mu = MU_VALUE) -> np.ndarray:
    x = mulaw(x)
    x = (x + 1.0) * mu / 2.0
    return x.astype(int)

def inv_mulaw(x: np.ndarray, mu=MU_VALUE):
    return np.sign(x) * (1.0 / mu) * ((1.0 + mu) ** np.abs(x) - 1.0)


def inv_mulaw_quantize(x: np.ndarray, mu=MU_VALUE):
    x = 2 * x.astype(np.float32) / mu - 1
    return inv_mulaw(x, mu)


def save_wav(wav: np.ndarray, path: str, sr: int):
    wav *= 32767 / max(0.0001, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))
