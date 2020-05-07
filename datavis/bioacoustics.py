import librosa
import numpy as np
from functools import wraps
from scipy.stats import entropy
from datavis import spectral
from datavis.common import gini, strided_array, moving_average


def toggle(f):
    """
    Turn on / off execution of the function
    :param f: function
    :return: function or None
    """
    @wraps(f)
    def wrapper(*args, **kwds):
        if kwds['config']['use']:
            return f(*args, **kwds)
        else:
            return None
    return wrapper


@toggle
def get_acoustic_complexity_index(y: np.ndarray, fs: int, config: dict) -> float:
    """
    The ACI is based on the "observation that many biotic sounds, such as bird songs, are characterized by an intrinsic
    variability of intensities, while some types of human generated noise (such as car passing or airplane transit)
    present very constant intensity values"

    The results given are accumulative. Very long samples will return very large values for ACI.

    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian
    community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: Acoustic Complexity Index (ACI)
    """
    aci_params = config['params']
    spec_params = config['spectrogram']
    spec, freq = spectral.spectrogram(sig=y, fs=fs, **spec_params)
    j_bin = int(aci_params['bin'] * fs / spec_params['hop'])
    full_bins = spec.shape[1] // j_bin

    spec = spec[:, :j_bin * full_bins].reshape(-1, spec.shape[0], j_bin)
    spec_diff = np.sum(np.abs(np.diff(spec)), axis=2)
    aci = np.sum(spec_diff / np.sum(spec, axis=2), axis=1)
    return aci.sum()


@toggle
def get_acoustic_diversity_index(y: np.ndarray, fs: int, config: dict) -> float:
    """
    The ADI is calculated by dividing the spectrogram into bins (default 10) and taking the proportion of the signals in
    each bin above a threshold (default -50 dBFS). The ADI is the result of the Shannon index applied to these bins.

    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: Diversity Index (DI)
    """
    adi_params = config['params']
    fs_max = min(adi_params['fs_max'], fs / 2)
    fs_step = adi_params['fs_step']
    db_threshold = adi_params['db_threshold']

    spec_segmented = spectral.segmented_spectogram(y=y, fs=fs, fs_step=fs_step, fs_max=fs_max, db_threshold=db_threshold)
    ADI = entropy(spec_segmented)
    return ADI


@toggle
def get_bioacoustic_index(y: np.ndarray, fs: int, config: dict):
    """
    The Bioacoustic Index is calculated as the "area under each curve included all frequency bands associated with the
    dB value that was greater than the minimum dB value for each curve. The area values are thus a function of both the
    sound level and the number of frequency bands used by the avifauna"

    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii:
    bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.


    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: Bioacoustic Index (BI)
    """
    bi_params = config['params']
    spec_params = config['spectrogram']
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


@toggle
def get_spectral_entropy(y: np.ndarray, fs: int, config: dict) -> float:
    """
    Spectral Entropy (Shannon definition) of audio signal.
    The Shannon spectral entropy of a noisy signal will tend towards 1 whereas the Shannon spectral entropy of a pure
    tone signal will tend towards 0.

    Reference: Han, NC, Muniandy SV, Dayou J (2011) Acoustic classification of Australian anurans based on hybrid
    spectral-entropy approach. Applied Acoustics.

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: spectral entropy (SE)
    """
    spec_params = config['spectrogram']
    spec, freq = spectral.spectrogram(sig=y, fs=fs, **spec_params)
    N = spec.shape[0]
    spec_sum = np.sum(spec, axis=1) / np.sum(spec)
    spectral_entropy = entropy(spec_sum) / np.log(N)

    return spectral_entropy


def get_temporal_entropy(y: np.ndarray, fs: int, config: dict) -> float:
    """
    Temporal entropy is a measure of the temporal dispersal of acoustic energy within a recording,
    has been shown to reflect the number of avian calls in a recording (Sueur, Pavoine et al. 2008).

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: temporal entropy (TE)
    """
    envelope = spectral.envelope(sig=y)
    envelope /= np.sum(envelope)
    N = len(envelope)
    temporal_entropy = entropy(envelope) / np.log(N)
    return temporal_entropy


@toggle
def get_acoustic_evenness_index(y: np.ndarray, fs: int, config: dict) -> float:
    """
    The AEI is calculated by dividing the spectrogram into bins (default 10) and taking the proportion of the signals
    in each bin above a threshold (default -50 dBFS). The AEI is the result of the Gini index applied to these bins.

    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic
    analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: Acoustic Evenness Index (AEI)
    """
    aei_params = config['params']
    fs_max = min(aei_params['fs_max'], fs / 2)
    fs_step = aei_params['fs_step']
    db_threshold = aei_params['db_threshold']
    spec_segmented = spectral.segmented_spectogram(y=y, fs=fs, fs_step=fs_step, fs_max=fs_max, db_threshold=db_threshold)
    aei = gini(spec_segmented)
    return aei


@toggle
def get_spectral_centroid(y: np.ndarray, fs: int, config: dict):
    """
    Compute the spectral centroid of an audio signal


    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: spectral centroid (SC)
    """
    spec_params = config['spectrogram']
    spec, freq = spectral.spectrogram(sig=y, fs=fs, **spec_params)
    centroid = freq.dot(spec) / spec.sum(axis=0)
    return centroid


@toggle
def get_acoustic_activity(y: np.ndarray, fs: int, config: dict) -> dict:
    """

    Compute the following:

    Signal-to-noise ratio (SNR): the decibel difference between the maximum envelope amplitude in any minute segment
    and the background noise.

    Acoustic activity: the fraction of frames within a one minute segment where the signal envelope is more than 3 dB
    above the level of background noise

    Count of acoustic events: the number of times that the signal envelope crosses the 3 dB threshold

    Average duration of acoustic events: an acoustic event is a portion of recordingwhich startswhen the signal envelope
    crosses above the 3 dB threshold and ends when it crosses belowthe 3 dB threshold.

    Reference: Towsey, Michael W. (2013) Noise removal from wave-forms and spectro- grams derived from natural recordings of
    the environment. Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived from Natural
    Recordings of the Environment. Queensland University of Technology, Brisbane

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: dictionary with SNR, Acoustic_activity, Count_acoustic_events and Average_duration
    """
    params = config['params']

    duration_s = len(y) / fs
    wave_env = 20 * np.log10(np.max(np.abs(strided_array(y, params['frame_len'], params['frame_len'])), axis=1))
    minimum = np.max((np.min(wave_env), params['min_dB']))
    hist, bin_edges = np.histogram(wave_env, range=(minimum, minimum + params['dB_range']),
                                   bins=params['hist_number_bins'], density=False)
    hist_smooth = moving_average(hist, kernel=params['hist_smoothing_kernel'], border='same')
    modal_intensity = np.argmax(hist_smooth)

    if params['N'] > 0:
        count_thresh = 0.68 * np.sum(hist_smooth) # 2 standard deviations from the mean (under normal dist)
        count = hist_smooth[modal_intensity]
        index_bin = 1
        while count < count_thresh:
            if modal_intensity + index_bin <= len(hist_smooth):
                count = count + hist_smooth[modal_intensity + index_bin]
            if modal_intensity - index_bin >= 0:
                count = count + hist_smooth[modal_intensity - index_bin]
            index_bin += 1
        thresh = np.min((params['hist_number_bins'], modal_intensity + params['N'] * index_bin))
        background_noise = bin_edges[thresh]
    else:
        background_noise = bin_edges[modal_intensity]

    SNR = np.max(wave_env) - background_noise
    SN = wave_env - background_noise - params['activity_threshold_dB']
    acoustic_activity = (SN > 0).sum() / float(len(SN))

    start_event = [n[0] for n in np.argwhere((SN[:-1] < 0) & (SN[1:] > 0))]
    end_event = [n[0] for n in np.argwhere((SN[:-1] > 0) & (SN[1:] < 0))]
    if len(start_event) != 0 and len(end_event) != 0:
        if start_event[0] < end_event[0]:
            events = list(zip(start_event, end_event))
        else:
            events = list(zip(end_event, start_event))
        count_acoustic_events = len(events)
        average_duration_e = np.mean([end - begin for begin, end in events])
        average_duration_s = average_duration_e * duration_s / float(len(SN))
    else:
        count_acoustic_events = 0
        average_duration_s = 0

    d = {'SNR': SNR, 'Acoustic_activity': acoustic_activity, 'Count_acoustic_events': count_acoustic_events,
         'Average_duration': average_duration_s}
    return d


@toggle
def get_formant_frequencies(y: np.ndarray, fs: int, config: dict) -> dict:
    """
    Formants are frequency peaks in the spectrum which have a high degree of energy.
    See e.g. https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu/61528322#61528322
    for explanation how order is selected.

    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: dictionary with formants quartiles, IQR and number of formants
    """
    order = config['params']['order']
    if order is None:
        order = fs // 1000
    A = librosa.core.lpc(y, order)
    rts = np.roots(A)
    rts = rts[np.imag(rts) >= 0]
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = angz * fs / (2 * np.pi)
    frqs.sort()

    q25, q50, q75 = np.quantile(frqs, [0.25, 0.50, 0.75])
    d = {'formant_q25': q25,
         'formant_q50': q50,
         'formant_q75': q75,
         'formant_IQR': q75 - q25,
         'formant_len': len(frqs)}
    return d


def get_bioacoustic_features(y: np.ndarray, fs: int, config: dict) -> dict:
    """
    Compute all bioacustic features
    :param y: mono audio
    :param fs: sampling (in Hz)
    :param config: config dictionary
    :return: dictionary with all bioacustic features
    """
    AE = get_acoustic_activity(y=y, fs=fs, config=config['Acoustic_activity'])
    bioacoustic_features = {
        'Acoustic_Complexity_Index': get_acoustic_complexity_index(y=y, fs=fs, config=config['Acoustic_Complexity_Index']),
        'Acoustic_Diversity_Index': get_acoustic_diversity_index(y=y, fs=fs, config=config['Acoustic_Diversity_Index']),
        'Bioacoustic_Index': get_bioacoustic_index(y=y, fs=fs, config=config['Bioacoustic_Index']),
        'Spectral_entropy': get_spectral_entropy(y=y, fs=fs, config=config['Spectral_entropy']),
        'Temporal_entropy': get_temporal_entropy(y=y, fs=fs, config=config['Temporal_entropy']),
        'Acoustic_Evenness_Index': get_acoustic_evenness_index(y=y, fs=fs, config=config['Acoustic_Evenness_Index']),
        'SNR': AE['SNR'],
        'Acoustic_activity': AE['Acoustic_activity'],
        'Acoustic_events_count': AE['Count_acoustic_events'],
        'Event_average_duration': AE['Average_duration']
    }
    formants = get_formant_frequencies(y=y, fs=fs, config=config['Formants'])
    bioacoustic_features.update(formants)
    return bioacoustic_features





