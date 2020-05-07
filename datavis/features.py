import os
import logging
import librosa
import yaml
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from datavis.yaafe_wrapper import YaafeWrapper
from datavis.bioacoustics import get_bioacoustic_features
from datavis.audio_io import get_all_waves_generator


def process_audio(path, config):
    try:
        y, fs = librosa.load(path, sr=None)
    except Exception as ex:
        logging.exception('Failed to load %s', path)
        return

    try:
        yaafe = YaafeWrapper(fs=fs, config=config['YAAFE_features'])
        yaafe_features = yaafe.compute_feature_stats(y)
        bioacoustic_features = get_bioacoustic_features(y=y, fs=fs, config=config['Bioacoustic_features'])
    except Exception as ex:
        logging.exception('Failed to process %s', path)
        return

    output_path = os.path.splitext(path)[0] + '.csv'
    pd.DataFrame(data={**bioacoustic_features, **yaafe_features}, index=[0]).to_csv(output_path, index=False)


def wav_dir_to_features(directory: str, config: str, n_jobs: int, resume: bool):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    files, total = get_all_waves_generator(directory=directory, resume=resume)

    if n_jobs == 1:
        for wav in tqdm(files):
            process_audio(path=wav, config=config)
    else:
        _ = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_audio)(path=path, config=config)
            for path in tqdm(files, total=total))


