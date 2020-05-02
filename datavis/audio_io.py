import os
import re
import glob
import logging
import pandas as pd
from io import StringIO
from typing import Generator, Tuple
from joblib import Parallel, delayed
from datetime import datetime
from pathlib import Path, PosixPath


class AudioIOException(Exception):
    pass


def get_all_waves_generator(directory: str, resume: bool = False):
    gen = Path(directory).rglob('*.wav')
    if resume:
        logging.info('Resuming processing')
        wav_paths = list(gen)
        all_paths = [os.path.splitext(path)[0] for path in wav_paths]
        all_paths = set(all_paths)
        csvs = list(Path(directory).rglob('*.csv'))
        csvs = [os.path.splitext(path)[0] for path in csvs]
        csvs = set(csvs)
        paths = all_paths - csvs
        paths = [path + '.wav' for path in paths]
        total = len(paths)
        logging.info('%d / %d completed. Remaining: %d', len(csvs), len(all_paths), total)
    else:
        total = sum(1 for _ in gen)
        paths = Path(directory).rglob('*.wav')
    return paths, total


def get_all_waves(directory: str) -> list:
    """
    Return all wave files (recursively) from the provided directory in sorted order
    :param directory: path to the directory
    :return: list of files (possibly empty)
    """
    files = glob.glob(directory + '/**/*.wav')
    if not files:
        logging.warning('No WAVE files found in ', directory)
    else:
        files.sort()
    return files


def extract_datetime_from_filename(filename: str) -> datetime:
    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', filename)
    if not match:
        raise AudioIOException(f'Infering datetime from filename {filename} failed. '
                               f'Following regular expression was expected '
                               f'"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}"')
    else:
        time_str = match.group(0)
        dt = datetime.strptime(time_str, "%Y-%m-%dT%H-%M-%S")
        return dt


def get_date_range_from_directory(directory: str, periods: int) -> pd.DatetimeIndex:
    waves = get_all_waves(directory=directory)
    first_file = os.path.basename(waves[0])
    last_file = os.path.basename(waves[-1])
    start = extract_datetime_from_filename(first_file)
    end = extract_datetime_from_filename(last_file)
    daterange = pd.date_range(start=start, end=end, periods=periods)
    return daterange


def read_result_csv(path):
    """
    Assumes the result file has a header and a single line with results
    :param path:
    :return:
    """
    with open(path) as fo:
        data = fo.readlines()[1]
    filename = os.path.basename(path)
    time = extract_datetime_from_filename(filename)
    data = str(time) + ',' + data
    return data


def get_result_header(path):
    with open(path) as fo:
        return fo.readline()


def read_results(directory: str) -> pd.DataFrame:
    """
    If reading this section makes you think "why not use pandas or dask read_csv?", answer is simple: processing
    with these takes prohibitively long time, especially concat of results. By using StringIO we reduce the load time
    over 100x for large datasets
    :param directory:
    :return:
    """
    csv_paths = list(Path(directory).rglob('*.csv'))
    header = get_result_header(csv_paths[0])
    data = Parallel(n_jobs=15, backend='loky')(delayed(read_result_csv)(path=path) for path in csv_paths)
    data = header + ''.join(data)
    data = StringIO(data)
    data = pd.read_csv(data)
    data.index = pd.to_datetime(data.index)
    return data.sort_index()
