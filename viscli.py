#!/usr/bin/env python3

import sys
import time
import click
import logging
import pandas as pd
from datavis.audio_features import wav_dir_to_features
from datavis.audio_io import get_date_range_from_directory
from datavis.audio_vis import save_heatmap_with_datetime, SUPPORTED_FORMATS

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
@click.option("--output", "-out", type=click.STRING, required=True, help="Output file.")
@click.option("--jobs", "-j", type=click.INT, default=-1, help="Number of jobs to run. Defaults to all cores",
              show_default=True)
@click.option("--config", "-c", type=click.Path(exists=True), default='datavis/config.ini',
              help="File with configuration parameters for the algorithm.")
def process(input, output, jobs, config):
    start_time = time.time()
    df = wav_dir_to_features(directory=input, config=config, n_jobs=jobs)
    df.to_hdf(path_or_buf=output, key='data')
    logging.info(f'Total time: {time.time() - start_time:.2f}s, output shape: {df.shape}')


@cli.command('f2i', help='HDF5 features to image')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a file with HDF5 features.")
@click.option("--directory", "-d", type=click.Path(exists=True), required=True, help="Path to the directory with "
              "audio files that were used to produce features. Datatime will be inferred from timestamps.")
@click.option("--output", "-out", type=click.STRING, required=True, help="Output file.")
@click.option("--format", "-f", type=click.Choice(SUPPORTED_FORMATS), default="html",
              show_default=True)
def process(input, directory, output, format):
    df = pd.read_hdf(input, key='data')
    daterange = get_date_range_from_directory(directory=directory, periods=len(df))
    df['datetime'] = daterange
    df = df.set_index('datetime')
    df = df.resample('1T').mean()
    df = (df - df.min()) / (df.max() - df.min())
    save_heatmap_with_datetime(df, output_path=output, dformat=format)


if __name__ == '__main__':
    cli()