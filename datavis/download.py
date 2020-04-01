#!/usr/bin/env python3

import os
import sys
import click
import requests
import logging
import backoff
from tqdm import tqdm
from joblib import Parallel, delayed


@click.group()
@click.option('--verbose', default=False, is_flag=True, help='Run in a silent mode')
def cli(verbose):
    """
    Helper script to download audio files from Topher. Adjust "clean" method if format changes.
    Example of what the script expects:
    curl -o ./2019-09-26T12-56-23.230Z.wav https://assets.rfcx.org/audio/8719aa78-7562-4f8c-af6f-3a5d985de507.wav;
    """
    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def clean(download_str: str):
    download_split = download_str.split()
    output = download_split[2][2:]
    link = download_split[3][:-1]
    return link, output


def url_to_file(url, filename, log_failed=True):
    try:
        download(url, filename)
        logging.debug('%s downloaded successfully to %', url, filename)
    except Exception:
        logging.exception('Failed to download %s to %s', url, filename)
        if log_failed:
            with open('failed.log', 'a') as flog:
                flog.write('{},{}\n'.format(url, filename))


@backoff.on_exception(backoff.expo,
                      (requests.exceptions.Timeout,
                       requests.exceptions.ConnectionError),
                      max_time=60)
def download(url, filename):
    with requests.get(url=url, timeout=30, stream=True) as request:
        request.raise_for_status()
        with open(filename, 'wb') as fh:
            for chunk in request.iter_content(1024 * 1024):
                if chunk:
                    fh.write(chunk)


@cli.command('file', help='Helper CLI to download files. Currently works with a rather special structure defined by shell script provided by Topher')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a directory with audio in WAV format.")
@click.option("--output", "-out", type=click.Path(exists=False, file_okay=False, writable=True), required=True, help="Output directory.")
@click.option("--jobs", "-j", type=click.INT, default=8, help="Number of threads to run.", show_default=True)
def file_download(input, output, jobs):
    with open(input, 'r') as fin:
        files_to_download = fin.read().splitlines()

    if files_to_download[0].startswith('#'):
        files_to_download.pop(0)
    files_to_download = [clean(s) for s in files_to_download]

    _ = Parallel(n_jobs=jobs, backend='threading')(
        delayed(url_to_file)(url=url, filename=os.path.join(output, filename))
        for url, filename in tqdm(files_to_download))


@cli.command('resume', help='Resume failed downloads')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a directory with audio in WAV format.", default='failed.log')
def resume(input):
    logging.getLogger('backoff').addHandler(logging.StreamHandler())

    with open(input, 'r') as fin:
        files_to_download = fin.read().splitlines()

    files_to_download = [s.split(',') for s in files_to_download]

    for url, filepath in tqdm(files_to_download):
        url_to_file(url, filepath, log_failed=False)


if __name__ == '__main__':
    cli()