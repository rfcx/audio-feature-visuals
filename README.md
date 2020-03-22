# Visualisation of long-duration audio features

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
  a2f  Audio to HDF5 features
```

### Audio to Features

```bash
Usage: viscli.py a2f [OPTIONS]

  Audio to HDF5 features

Options:
  -in, --input PATH    Path to a directory with audio in WAV format.
                       [required]
  -out, --output TEXT  Output file or directory.
  -j, --jobs INTEGER   Number of jobs to run. Defaults to all cores  [default:
                       -1]
  -c, --config PATH    File with configuration parameters for the algorithm.
  --help               Show this message and exit.

```

Example:

```bash
viscli.py a2f --input rfcx/sample_24h_tembe --output test.h5 --jobs -2
```

The script will use the default config present in the module directory and compute features on all but 1 vCPU.

On Ryzen 1700X and 2.1 GB of input sampled at 12khz it completes the task in less than a minute. Example output:

```bash
INFO:root:Following features were selected: Variation,Chroma,Rolloff,Crest factors,LPC,OBSI,Flatness,MFCC,Flux,LSF
INFO:numexpr.utils:Note: NumExpr detected 16 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO:numexpr.utils:NumExpr defaulting to 8 threads.
INFO:root:Total time: 54.86s, output shape: (501804, 67)
```

The shape on `axis=0` (i.e. `501804`) reflects the block size parameter (in this case `2**12 = 4096`) present in the config file.  By default, it comes with a step size that is half of the block size, i.e.

```bash
step_size = block_size / 2 = 2**11
```



## Handy commands

Merge all WAVE files into one:

```bash
sox $(find . -name '*.wav' | sort -n) combined.wav
```