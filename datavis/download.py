#!/usr/bin/env python3

import os
import sys
import glob
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


def remove_files_already_downloaded(output_dir: str, output_path_with_link: list) -> list:
    files_to_download = [output_path for link, output_path in output_path_with_link]
    downloaded_files_path = glob.glob(os.path.join(output_dir, '*'))
    downloaded_files = [os.path.basename(s) for s in downloaded_files_path]
    remaining_files = set(files_to_download) - set(downloaded_files)
    output_path_with_link_d = {k: v for v, k in output_path_with_link}
    remaining_files_with_links = [(output_path_with_link_d[filename], filename) for filename in remaining_files]
    return list(remaining_files_with_links)


@cli.command('file', help='Helper CLI to download files. Currently works with a rather special structure defined by shell script provided by Topher')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a directory with audio in WAV format.")
@click.option("--output", "-out", type=click.Path(exists=False, file_okay=False, writable=True), required=True, help="Output directory.")
@click.option("--jobs", "-j", type=click.INT, default=8, help="Number of threads to run.", show_default=True)
@click.option('--fresh', default=False, is_flag=True, help='Ignore any files that might have been already downloaded and start fresh', show_default=True)
@click.option('--dry', default=False, is_flag=True, help='Dry run. Do not download anythyng, just show how much will be downloaded', show_default=True)
def file_download(input, output, jobs, fresh, dry):
    with open(input, 'r') as fin:
        output_path_with_link = fin.read().splitlines()

    if output_path_with_link[0].startswith('#'):
        output_path_with_link.pop(0)
    output_path_with_link = [clean(s) for s in output_path_with_link]

    total_files = len(output_path_with_link)
    remaining_files_no = total_files
    if not fresh:
        output_path_with_link = remove_files_already_downloaded(output_dir=output, output_path_with_link=output_path_with_link)
        remaining_files_no = len(output_path_with_link)

    if dry:
        print(f'Number of files to download: {remaining_files_no} out of total {total_files}')
    else:
        _ = Parallel(n_jobs=jobs, backend='threading')(
            delayed(url_to_file)(url=url, filename=os.path.join(output, filename))
            for url, filename in tqdm(output_path_with_link))


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