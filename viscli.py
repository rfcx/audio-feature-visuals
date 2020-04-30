#!/usr/bin/env python3

import sys
import time
import click
import logging
import pandas as pd
from datavis.common import setup_logging
from datavis.features import wav_dir_to_features
from datavis.audio_io import get_date_range_from_directory
from datavis.audio_vis import save_heatmap_with_datetime, SUPPORTED_FORMATS

@click.group()
@click.option('--quiet', default=False, is_flag=True, help='Run in a silent mode')
def cli(quiet):
    """
    viscli is a command line program that extracts audio features
    """
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if quiet:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=format)
    setup_logging()


@cli.command('a2f', help='Audio to HDF5 features')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a directory with audio in WAV format.")
@click.option("--output", "-out", type=click.STRING, help="Output file. If not present, the output will be written to "
                                                          "separate files with the same signature as the original one" )
@click.option("--jobs", "-j", type=click.INT, default=-1, help="Number of jobs to run. Defaults to all cores",
              show_default=True)
@click.option("--config", "-c", type=click.Path(exists=True), default='datavis/config.yaml',
              help="File with configuration parameters for the algorithm.")
@click.option('--resume', default=False, is_flag=True, help='Resume processing')
def audio_to_features(input, output, jobs, config, resume):
    start_time = time.time()
    wav_dir_to_features(directory=input, config=config, n_jobs=jobs, resume=resume)
    logging.info(f'Total time: {time.time() - start_time:.2f}s')


@cli.command('f2i', help='HDF5 features to image')
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to a file with HDF5 features.")
@click.option("--directory", "-d", type=click.Path(exists=True), required=True, help="Path to the directory with "
              "audio files that were used to produce features. Datatime will be inferred from timestamps.")
@click.option("--output", "-out", type=click.STRING, required=True, help="Output file.")
@click.option("--format", "-f", type=click.Choice(SUPPORTED_FORMATS), default="html",
              show_default=True)
def features_to_image(input, directory, output, format):
    df = pd.read_hdf(input, key='data')
    daterange = get_date_range_from_directory(directory=directory, periods=len(df))
    df['datetime'] = daterange
    df = df.set_index('datetime')
    df = df.resample('1T').mean()
    df = (df - df.min()) / (df.max() - df.min())
    save_heatmap_with_datetime(df, output_path=output, dformat=format)


if __name__ == '__main__':
    cli()