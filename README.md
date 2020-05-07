# Visualisation of long-duration audio features

- [Development env](#development-env)
- [CLI](#cli)
  * [Audio to Features](#audio-to-features)
  * [Features to Image](#features-to-image)
- [Audio features](#audio-features)
  * [YAAFE set](#yaafe-set)
- [Handy commands](#handy-commands)

## Development env

Use Anaconda for the best experience:

```bash
conda env create --file=environment.yml --name rfcx
```

## CLI

```bash
Usage: viscli.py [OPTIONS] COMMAND [ARGS]...

  viscli is a command line program that extracts audio features

Options:
  --quiet  Run in a silent mode
  --help   Show this message and exit.

Commands:
  a2f  Audio to Features.
  f2i  Features to Image
```

### Audio to Features

```bash
Usage: viscli.py a2f [OPTIONS]

  Audio to Features. The script will calculate features per file and save
  the result next to the input file.

Options:
  -in, --input PATH   Path to a directory with audio in WAV format.
                      [required]
  -j, --jobs INTEGER  Number of jobs to run. Defaults to all cores  [default:
                      -1]
  -c, --config PATH   File with configuration parameters for the algorithm.
  --resume            Resume processing
  --help              Show this message and exit.
```

Example:

```bash
viscli.py a2f --input rfcx/sample_24h_tembe --jobs -2
```

The script will process all WAV files present in `sample_24h_tembe`.

### Features to Image

```bash
Usage: viscli.py f2i [OPTIONS]

  Features to Image

Options:
  -in, --input PATH               Path to a file with HDF5 features.
                                  [required]
  -d, --directory PATH            Path to the directory with audio files that
                                  were used to produce features. Datatime will
                                  be inferred from timestamps.  [required]
  -out, --output TEXT             Output file.  [required]
  -f, --format [html|png|webp|svg|pdf|eps]
                                  [default: html]
  --help                          Show this message and exit.
```

HTML allows interactive use of the plot. 


## Audio features

Initial set of features that will be modified later.

### YAAFE set

Number of basic audio features are computed via [YAAFE](https://github.com/Yaafe/Yaafe) library. Features are explained in [docs](http://yaafe.sourceforge.net/features.html).

* Chroma
* Linear Predictor Coefficients
* Line Spectral Frequency
* Mel-frequency cepstral coefficients
* Octave band signal intensity with triangular filter
* Perceptual sharpness
* Perceptual spread
* Spectral crest factors
* Spectral irregularity
* Spectral decrease
* Spectral flatness
* Spectral flux
* Spectral rolloff
* Spectral variation
* Spectral slope
* Zero-crossing rate

## Handy commands

Merge all WAVE files into one:

```bash
sox $(find . -name '*.wav' | sort -n) combined.wav
```