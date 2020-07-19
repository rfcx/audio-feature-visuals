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

### Docker

First time (and any time the environment.yml changes), build the Docker image:

```bash
docker build -t audio-feature-visuals .
```

Then to run the CLI (from your local files):

```bash
docker run -it --rm -v ${PWD}:/app audio-feature-visuals viscli.py
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

If the process was interrupted before all files were processed, you can resume it with `--resume` option like this:

```
./viscli.py a2f --input rfcx/97519ab33e08 -j 15 --resume
2020-05-07 17:59:20,652 :: root :: INFO :: Resuming processing
2020-05-07 17:59:31,238 :: root :: INFO :: 235944 / 309435 completed. Remaining: 73491
 0%|▏                                                | 255/73491 [00:29<2:23:40,  8.50it/s]
```

The script will process all WAV files present in `sample_24h_tembe`.

### Features to Image

```
Usage: viscli.py f2i [OPTIONS]

  Features to Image

Options:
  -in, --input PATH               Path to the directory with csv features.
                                  [required]
  -out, --output TEXT             Output file.  [required]
  -f, --format [html|png|webp|svg|pdf|eps]
                                  [default: html]
  -agg, --aggregation INTEGER     Aggregation (in minutes) to apply on the
                                  data  [default: 10]
  --corr TEXT                     Output path for plotting correlation matrix.
  --help                          Show this message and exit.
```

HTML is the default one as it allows interaction with the plot. Passing optional `--corr` argument plots [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) matrix. Example of such a plot made on site 6658c4fd3657 can be found [here](https://plotly.com/~tracewsl/390/#/) .


## Audio features

The audio features are defined in the [config file](datavis/config.yaml) and split in two groups:

* Bioacoustic features. These were implemented from respective papers and / or R packages, primarily [seewave](https://cran.r-project.org/package=seewave) and [soundecology](https://cran.r-project.org/package=soundecology) - both very popular in the community. The correctness of the implementation was verified by running two random samples on our and R code and checking that the relative difference is less than 1%. 
* [YAAFE features](http://yaafe.sourceforge.net/). Yaafe is an audio features extraction toolbox implemented in C++. It's fast and smart, its author is a very good audio engineer. They have a separate category so that we can easily not only differentiate them from the bioacoustic features, but also add new features if needed simply by editing the config file. 

To turn on / off calculation of the feature, change the `use` option on the config.

### YAAFE set

Number of basic audio features are computed via [YAAFE](https://github.com/Yaafe/Yaafe) library. Features are explained in [docs](http://yaafe.github.io/Yaafe/features.html).

* **Chroma**. Describes pitch class profiles and nicely captures capture harmonic and melodic characteristics.
* Linear Predictor Coefficients (LPC). Primarily used in speech processing, used to calculate [formants](https://en.wikipedia.org/wiki/Formant). It allows us to capture the main frequencies in the signal. 
* **Line Spectral Frequency** (LSF).  LSFs have a very good quantization ability as well as they possess an efficiency in terms
of representation. Closely related to LPC.
* **Mel-frequency cepstral coefficients** (MFCC). Representation of the short-term power spectrum of a sound on a nonlinear mel scale of frequency.
* **Octave band signal intensity ratio** (OBSIR). Computes log of ratio between consecutive octaves. 
* **Perceptual sharpness**. The sharpness is a sensation value caused by high-frequency components in a signal.
* **Perceptual spread**. Compute spread of loudness coefficients. Loudness is the psychological counterpart to sound pressure level. 
* **Spectral crest factors**. Crest factor is a parameter of a waveform, such as alternating current or sound, showing the ratio of peak values to the effective value. They indicate how extreme the peaks are in a waveform.
* **Spectral irregularity**. Deviation of the amplitude harmonic peaks from a global spectral envelope derived from a running mean of the amplitude of three adjacent harmonics averaged over the sound duration. 
* **Spectral decrease**. Describes the average spectral slope, puts stronger emphasis on the low frequencies.
* **Spectral flatness**. A low spectral flatness (approaching 0.0 for a pure tone) indicates that the spectral power is concentrated in a relatively small number of bands — this would typically sound like a mixture of sine waves, and the spectrum would appear "spiky". A high spectral flatness (approaching 1.0 for white noise) indicates that the spectrum has a similar amount of power in all spectral bands — this would sound similar to white noise, and the graph of the spectrum would appear relatively flat and smooth.
* **Spectral flux**. A measure of how quickly the power spectrum of a signal is changing, calculated by comparing the power spectrum for one frame against the power spectrum from the previous frame.
* **Spectral rolloff**. Spectral roll-off is the frequency so that 99% of the energy is contained below.
* **Spectral variation**. Normalised correlation of spectrum between consecutive frames.
* **Spectral slope**. Approximation of the spectrum shape by a linear regression line. 
* **Zero-crossing rate**. Rate of sign-changes along a signal.

#### How to extend YAAFE features set

There's no need to change the code, just edit the config file. Let's say you would like to add `SpectralFlatnessPerBand`. From YAAFE docs:

> yaafelib.yaafe_extensions.yaafefeatures.SpectralFlatnessPerBand
Compute spectral flatness per log-spaced band of 1/4 octave, as proposed in MPEG7 standard.
>
> Parameters:
>
> FFTLength (default=0): Frame’s length on which perform FFT. Original frame is padded with zeros or truncated to reach this size. If 0 then use original frame length.
>
> FFTWindow (default=Hanning): Weighting window to apply before fft. Hanning|Hamming|None
>
> blockSize (default=1024): output frames size
>
> stepSize (default=512): step between consecutive frames

Add the following to the `config.yaml`:

```
  SpectralFlatnessPerBand:
    use: on
    params:
      blockSize: 2048
```   

Note that there is no need to add parameters that you are not changing.  

