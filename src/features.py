"""Sound level (dB(A)) and log-mel spectrogram feature extraction."""
import numpy as np
import librosa
from scipy.signal import bilinear, lfilter
import config

P_REF = 20e-6


def _a_weighting_coeffs(fs):
    """Digital A-weighting filter (IEC 61672) via bilinear transform."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(nums, dens, fs)


_B, _A = _a_weighting_coeffs(config.SAMPLE_RATE)


def spl_dba(x):
    """A-weighted sound pressure level of a window (Pascals) in dB(A)."""
    y = lfilter(_B, _A, x)
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    return float(20 * np.log10(rms / P_REF))


def spl_db(x):
    """Un-weighted (linear) sound pressure level in dB."""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    return float(20 * np.log10(rms / P_REF))


def log_mel(x):
    """Log-mel spectrogram, standardised per-window -> shape (N_MELS, frames, 1)."""
    S = librosa.feature.melspectrogram(
        y=x.astype(np.float32), sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, fmax=config.FMAX)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.mean()) / (S.std() + 1e-8)
    return S[..., np.newaxis].astype(np.float32)
