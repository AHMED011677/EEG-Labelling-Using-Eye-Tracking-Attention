import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def lowpass_50hz(x, fs, cutoff=50, order=4):
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, x)

def notch_50hz(x, fs, f0=50, Q=30):

    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    return filtfilt(b, a, x)