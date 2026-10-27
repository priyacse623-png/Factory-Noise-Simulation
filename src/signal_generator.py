"""Synthetic factory sound generator (values are in Pascals).

Each machine condition gets a different spectral/temporal signature:
  Normal        - low-frequency motor hum + mild broadband noise      (~68-74 dB)
  Bearing fault - hum + high-frequency resonance, amplitude-modulated  (~76-84 dB)
  Air leak      - strong broadband high-frequency hiss                 (~80-88 dB)
  Impact        - periodic sharp clicks / knocks                       (~86-95 dB)
  Overload      - loud low-frequency hum with many harmonics           (~90-98 dB)
"""
import numpy as np
from scipy.signal import butter, lfilter
import config

P_REF = 20e-6  # 20 micro-Pascal reference pressure


def _bandpass(x, lo, hi, fs=config.SAMPLE_RATE, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return lfilter(b, a, x)


def _highpass(x, fc, fs=config.SAMPLE_RATE, order=4):
    b, a = butter(order, fc / (fs / 2), btype="high")
    return lfilter(b, a, x)


def _scale_to_db(x, target_db):
    """Scale signal so its RMS corresponds to target_db SPL."""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    target_rms = P_REF * 10 ** (target_db / 20)
    return x * (target_rms / rms)


def generate(label, rng=None, n=config.N_SAMPLES, fs=config.SAMPLE_RATE):
    """Return one window (numpy array, Pascals) for the given class label."""
    rng = rng or np.random.default_rng()
    t = np.arange(n) / fs
    noise = rng.standard_normal(n)

    if label == "Normal":
        f0 = rng.uniform(90, 130)
        x = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
             + 0.6 * _bandpass(noise, 80, 1500))
        level = rng.uniform(68, 74)

    elif label == "Bearing fault":
        f0 = rng.uniform(90, 130)
        res = rng.uniform(2500, 4000)                  # bearing resonance
        am = 1 + 0.9 * np.sin(2 * np.pi * rng.uniform(25, 45) * t)
        x = (0.6 * np.sin(2 * np.pi * f0 * t)
             + 1.0 * am * _bandpass(noise, res - 400, res + 400)
             + 0.3 * _bandpass(rng.standard_normal(n), 80, 1500))
        level = rng.uniform(76, 84)

    elif label == "Air leak":
        x = _highpass(noise, rng.uniform(3000, 4500)) + 0.2 * _bandpass(noise, 100, 800)
        level = rng.uniform(80, 88)

    elif label == "Impact":
        x = 0.1 * _bandpass(noise, 80, 1500)
        period = rng.uniform(0.18, 0.35)               # seconds between knocks
        start = rng.uniform(0, period)
        while start < 1.0:
            i = int(start * fs)
            L = min(int(0.03 * fs), n - i)
            if L > 0:
                decay = np.exp(-np.arange(L) / (0.006 * fs))
                x[i:i + L] += 6 * decay * _bandpass(rng.standard_normal(L), 300, 6000)
            start += period
        level = rng.uniform(86, 95)

    elif label == "Overload":
        f0 = rng.uniform(45, 70)
        x = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 12))
        x = x + 0.4 * _bandpass(noise, 60, 1000)
        level = rng.uniform(90, 98)

    else:
        raise ValueError(f"Unknown class: {label}")

    # ambient factory background (so classes are not trivially clean)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    x = x + rng.uniform(0.05, 0.25) * _bandpass(rng.standard_normal(n), 50, 4000)
    return _scale_to_db(x, level)
