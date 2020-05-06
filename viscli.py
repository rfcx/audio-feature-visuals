#!/usr/bin/env python3

import sys
import time
import click
import logging
from datavis.common import setup_logging
from datavis.features import wav_dir_to_features
from datavis.audio_io import read_results
from datavis.audio_vis import save_heatmap_with_datetime, SUPPORTED_FORMATS, save_corr_matrix


@click.group()
@click.option('--quiet', default=False, is_flag=True, help='Run in a silent mode')
def cli(quiet):
    """
    viscli is a command line program that extracts audio features
    """
    setup_logging(quiet)


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
@click.option("--input", "-in", type=click.Path(exists=True), required=True, help="Path to the directory with csv features.")
@click.option("--output", "-out", type=click.STRING, required=True, help="Output file.")
@click.option("--format", "-f", type=click.Choice(SUPPORTED_FORMATS), default="html", show_default=True)
@click.option("--aggregation", "-agg", type=click.INT, help="Aggregation (in minutes) to apply on the data",
              default=10, show_default=True)
@click.option('--corr', type=click.STRING, help="Output path for plotting correlation matrix. ")
def features_to_image(input, output, format, aggregation, corr):
    df = read_results(directory=input)
    df = df.resample(f'{aggregation}T').mean()
    df = (df - df.min()) / (df.max() - df.min())
    save_heatmap_with_datetime(df, output_path=output, dformat=format)
    if corr:
        save_corr_matrix(df, output_path=corr)


if __name__ == '__main__':
    cli()