import numpy as np
from cachetools.keys import hashkey
from cachetools import LRUCache, cached
from scipy import signal, fftpack
from datavis.common import strided_array


def speckey(sig, *args, **kwargs):
    key = hashkey(*args, **kwargs)
    return key


@cached(LRUCache(maxsize=10), key=speckey)
def spectrogram(sig, fs, win_len=512, hop=256, win_type='hanning', filename=''):
    W = signal.get_window(win_type, win_len, fftbins=False)
    sig_strided = strided_array(sig, win_len, hop)
    sig_windowed = np.multiply(sig_strided, W)
    Sxx = np.abs(np.fft.rfft(sig_windowed, win_len))[:, :win_len // 2]
    Sxx = np.transpose(Sxx)
    freq = np.arange(0, fs / 2, fs / win_len)
    return Sxx, freq


def envelope(sig: np.ndarray):
    env = np.abs(signal.hilbert(sig, fftpack.helper.next_fast_len(len(sig))))
    return env


def segmented_spectogram(y: np.ndarray, fs: int, fs_step: float, fs_max: float, db_threshold: float) -> np.ndarray:
    fs_win = fs_max / fs_step
    win_len = int(fs / fs_win)
    spec, freq = spectrogram(y, fs, win_len=win_len, hop=win_len)

    bands_Hz = np.arange(fs_step, fs_max, fs_step)
    bands_bin = (bands_Hz / fs_win).astype(int)
    spec_db = 20 * np.log10(spec / np.max(spec))
    spec_bands = np.split(spec_db, bands_bin)
    spec_segmented_and_thresholded = np.array([np.sum(arr > db_threshold) / arr.size for arr in spec_bands])

    return spec_segmented_and_thresholded


# def spectrogram(sig, fs, filename='', win_len=512, hop=256, win_type='hanning'):
#     W = signal.get_window(win_type, win_len, fftbins=False)
#     nyquist = fs // 2
#
#     t = np.arange(0, len(sig) - win_len + 1, hop)
#     frames = [sig[i:i + win_len] * W for i in t]
#     Sxx = [np.abs(np.fft.rfft(frame, win_len))[:win_len // 2] for frame in frames]
#     Sxx = np.transpose(Sxx)
#     frequencies = [e * nyquist / float(win_len / 2) for e in np.arange(win_len // 2)]
#     return Sxx, frequencies