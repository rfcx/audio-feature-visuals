import yaml
import librosa
import numpy as np
from functools import wraps
from scipy.stats import entropy
from datavis import spectral


def toggle(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        if args[2]['use']:
            return f(*args, **kwds)
        else:
            return None
    return wrapper


def compute_acoustic_complexity_index(y, fs, config):
    aci_params = config['Features']['Acoustic_Complexity_Index']['params']
    spec_params = config['Features']['Acoustic_Complexity_Index']['spectrogram']
    spec, freq = spectral.spectrogram(sig=y, fs=fs, **spec_params)
    j_bin = int(aci_params['bin'] * fs / spec_params['hop'])
    full_bins = spec.shape[1] // j_bin

    spec = spec[:, :j_bin * full_bins].reshape(-1, spec.shape[0], j_bin)
    spec_diff = np.sum(np.abs(np.diff(spec)), axis=2)
    aci = np.sum(spec_diff / np.sum(spec, axis=2), axis=1)
    return aci.sum(), aci


def compute_acoustic_diversity_index(y, fs, config):
    adi_params = config['Features']['Acoustic_Diversity_Index']['params']
    fs_max = min(adi_params['fs_max'], fs / 2)
    fs_step = adi_params['fs_step']
    db_threshold = adi_params['db_threshold']

    spec_segmented = spectral.segmented_spectogram(y=y, fs=fs, fs_step=fs_step, fs_max=fs_max, db_threshold=db_threshold)
    ADI = entropy(spec_segmented)
    return ADI


def compute_bioacoustic_index(y: np.ndarray, fs: int, config: dict):
    bi_params = config['Features']['Bioacoustic_Index']['params']
    spec_params = config['Features']['Bioacoustic_Index']['spectrogram']
    fs_min = bi_params['fs_min']
    fs_max = min(bi_params['fs_max'], fs / 2)
    spec, freq = spectral.spectrogram(sig=y, fs=fs, **spec_params)

    min_freq_idx = (np.abs(freq - fs_min)).argmin() - 1
    max_freq_idx = (np.abs(freq - fs_max)).argmin()

    spec = spec[min_freq_idx: max_freq_idx]

    spec_BI = 20 * np.log10(spec / np.max(spec))
    spec_BI_mean = np.mean(10 ** (spec_BI / 10), axis=1)
    spec_BI_mean = 10 * np.log10(spec_BI_mean)
    spectre_BI_mean_normalized = spec_BI_mean - spec_BI_mean.min()
    BI = (spectre_BI_mean_normalized / (freq[1] - freq[0])).sum()

    return BI


def compute_spectral_entropy(y: np.ndarray, fs: int, config: dict) -> float:
    spec_params = config['Features']['Spectral_entropy']['spectrogram']
    spec, freq = spectral.spectrogram(sig=y, fs=fs, **spec_params)
    N = spec.shape[0]
    spec_sum = np.sum(spec, axis=1) / np.sum(spec)
    spectral_entropy = entropy(spec_sum) / np.log(N)

    return spectral_entropy


def compute_temporal_entropy(y: np.ndarray, fs: int, config: dict) -> float:
    envelope = spectral.envelope(sig=y)
    envelope /= np.sum(envelope)
    N = len(envelope)
    temporal_entropy = entropy(envelope) / np.log(N)
    return temporal_entropy


def compute_acoustic_evenness_index(y: np.ndarray, fs: int, config: dict) -> float:
    aei_params = config['Features']['Acoustic_Evenness_Index']['params']
    fs_max = min(aei_params['fs_max'], fs / 2)
    fs_step = aei_params['fs_step']
    db_threshold = aei_params['db_threshold']
    spec_segmented = spectral.segmented_spectogram(y=y, fs=fs, fs_step=fs_step, fs_max=fs_max, db_threshold=db_threshold)
    aei = _gini(spec_segmented)
    return aei


def _gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad / np.mean(x)
    G = 0.5 * rmad
    return G


if __name__ == '__main__':
    yml_file = 'config.yaml'
    with open(yml_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sample_path = '/media/tracek/linux-data/client/rfcx/test/2020-02-25T05-45-06.587Z.wav'
    # sample_path2 = '/media/tracek/linux-data/client/rfcx/test/2020-02-25T06-07-42.674Z.wav'
    sample, fs = librosa.load(sample_path, sr=None)
    ACI, temp = compute_acoustic_complexity_index(y=sample, fs=fs, config=config)
    ADI = compute_acoustic_diversity_index(y=sample, fs=fs, config=config)
    BI = compute_bioacoustic_index(y=sample, fs=fs, config=config)
    SE = compute_spectral_entropy(y=sample, fs=fs, config=config)
    TE = compute_temporal_entropy(y=sample, fs=fs, config=config)
    AEI = compute_acoustic_evenness_index(y=sample, fs=fs, config=config)
    print(f'Acoustic Diversity Index:: {ADI}')
    print(f'Acoustic Complexity Index: {ACI}')
    print(f'Bioacoustic Index: {BI}')
    print(f'Spectral entropy: {SE}')
    print(f'Temporal entropy: {TE}')
    print(f'Acoustic evenness index: {AEI}')




