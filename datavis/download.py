#!/usr/bin/env python3

import os
import sys
import click
import requests
import logging
from tqdm import tqdm
from joblib import Parallel, delayed


logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


def clean(download_str: str):
    download_split = download_str.split()
    output = download_split[2][2:]
    link = download_split[3][:-1]
    return link, output


def download(url, filename):
    try:
        request = requests.get(url=url, timeout=10, stream=True)
        with open(filename, 'wb') as fh:
            for chunk in request.iter_content(1024 * 1024):
                fh.write(chunk)
    except Exception:
        logging.exception('Failed to download %s to %s', url, filename)
        with open('failed.log', 'a') as flog:
            flog.write('%s,%s'.format(url, filename))


@click.command(help='Helper CLI to download files. Currently works with a rather special structure defined by shell script provided by Topher')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a directory with audio in WAV format.")
@click.option("--output", "-out", type=click.Path(exists=False, file_okay=False, writable=True), required=True, help="Output directory.")
@click.option("--jobs", "-j", type=click.INT, default=8, help="Number of threads to run.", show_default=True)
def cli(input, output, jobs):
    with open(input, 'r') as fin:
        files_to_download = fin.read().splitlines()

    if files_to_download[0].startswith('#'):
        files_to_download.pop(0)
    files_to_download = [clean(s) for s in files_to_download]

    _ = Parallel(n_jobs=jobs, backend='threading')(
        delayed(download)(url=url, filename=os.path.join(output, filename))
        for url, filename in tqdm(files_to_download))


if __name__ == '__main__':
    cli()