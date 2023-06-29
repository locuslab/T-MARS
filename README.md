# T-MARS Text Masking Code
We use FAST (https://github.com/czczup/FAST) as the base algorithm for text detection and MMOCR (https://mmocr.readthedocs.io/en/dev-1.x/) for text recognition i.e. reading the text. This repository shares the combined implementation of FAST and MMOCR for running on web-scales (adapted from DataComp https://www.datacomp.ai/).


## Installation

```sh
pip install git+https://github.com/mlfoundations/dataset2metadata
```

## Text detection and recognition
Please refer to [/dataset2metadata/text_detection/text_snake_wrapper.py] for the main implementation of text detection (FAST) and text recognition (MMOCR). 

## RUN
Please see the [examples/] folder for ways in which dataset2metadata is to be used for running T-MARS on webscale. You can specify the tar file paths in [examples/text_template.yml] and create multiple such template files using [examples/prepare_jobs.py]

```
dataset2metadata --yml examples/text_template.yml
```

## Acknowlegements
We thank the authors of FAST, MMOCR and DataComp team for open sourcing their code bases. 