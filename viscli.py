#!/usr/bin/env python3

import sys
import time
import click
import logging
from datavis.audio_features import wav_dir_to_features


@click.group()
@click.option('--quiet', default=False, is_flag=True, help='Run in a silent mode')
def cli(quiet):
    """
    viscli is a command line program that extracts audio features
    """
    if quiet:
        logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@cli.command('a2f', help='Audio to HDF5 features')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a directory with audio in WAV format.")
@click.option("--output", "-out", type=click.STRING, default='.', help="Output file or directory.")
@click.option("--jobs", "-j", type=click.INT, default=-1, help="Number of jobs to run. Defaults to all cores",
              show_default=True)
@click.option("--config", "-c", type=click.Path(exists=True), default='datavis/config.ini',
              help="File with configuration parameters for the algorithm.")
def process(input, output, jobs, config):
    start_time = time.time()
    df = wav_dir_to_features(directory=input, config=config, n_jobs=jobs)
    df.to_hdf(path_or_buf=output, key='data')
    logging.info(f'Total time: {time.time() - start_time:.2f}s, output shape: {df.shape}')


if __name__ == '__main__':
    cli()