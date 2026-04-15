import numpy as np


def window_features(x, fs, win_s=1.0, step_s=0.5):
    """
    Convert a 1D signal into sliding-window features.

    Returns an array of shape (n_windows, 2):
      [window_mean, window_std]
    """
    x = np.asarray(x).ravel()

    win = int(round(win_s * fs))
    step = int(round(step_s * fs))

    if win <= 0 or step <= 0:
        raise ValueError("win_s and step_s must be > 0.")
    if win > x.size:
        raise ValueError("Window size is larger than the signal length.")

    feats = []
    for start in range(0, x.size - win + 1, step):
        seg = x[start:start + win]
        feats.append([float(seg.mean()), float(seg.std(ddof=1))])

    return np.array(feats, dtype=float)

    # src/features.py

import numpy as np


def window_features(x, fs, win_s=1.0, step_s=0.5):
    """
    Convert a 1D signal into sliding-window features.

    Returns an array of shape (n_windows, 2):
      [window_mean, window_std]
    """
    x = np.asarray(x).ravel()

    win = int(round(win_s * fs))
    step = int(round(step_s * fs))

    if win <= 0 or step <= 0:
        raise ValueError("win_s and step_s must be > 0.")
    if win > x.size:
        raise ValueError("Window size is larger than the signal length.")

    feats = []
    for start in range(0, x.size - win + 1, step):
        seg = x[start:start + win]
        feats.append([float(seg.mean()), float(seg.std(ddof=1))])

    return np.array(feats, dtype=float)