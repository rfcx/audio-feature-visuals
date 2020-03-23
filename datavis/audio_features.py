import os
import configparser
import logging
import librosa
import pandas as pd
from joblib import Parallel, delayed
from datavis.yaafe_wrapper import YaafeWrapper, YAAFE_FEATURES
from datavis.audio_io import get_all_waves


def process_audio(path, block_size, selected_features):
    y, fs = librosa.load(path, sr=None)
    yaafe = YaafeWrapper(fs=fs, block_size=block_size, selected_features=selected_features)
    try:
        features = yaafe.get_features_as_df(y)
    except SystemError as e:
        logging.exception('Fatal error when loading data from ', path)
        return
    return features


def wav_dir_to_features(directory: str, config: str, n_jobs: int) -> pd.DataFrame:
    selected_features, block_size = _get_params(config)
    files = get_all_waves(directory=directory)

    if n_jobs == 1:
        features = [process_audio(path, block_size=block_size, selected_features=selected_features)
                    for path in files]
    else:
        features = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(process_audio)(path=path, block_size=block_size, selected_features=selected_features)
            for path in files)

    features = pd.concat(features, axis=0)
    return features


def _get_params(path: str):
    """
    Get selected audio features from string or ini file
    :param selection: string or path to ini file
    :return: comma-separated features
    """

    if os.path.isfile(path):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)
        block_size = config.getint('FFT', 'block_size')
        selected_features = [feature for feature, enabled in config.items('FEATURES') if enabled.lower() == 'yes']
    else:
        raise Exception('Config file not found.')

    selected_features_fullname = [YAAFE_FEATURES[feature] for feature in selected_features]
    selected_features_fullname = ','.join(selected_features_fullname)
    logging.info(f'Following features were selected: {selected_features_fullname}')

    return selected_features, block_size

