import glob
import logging


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
