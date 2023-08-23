# T-MARS
Official repository for the paper [T-MARS: Improving Visual Representations by Circumventing Text Feature Learning](https://arxiv.org/abs/2307.03132).  
Webpage: https://tmars-clip.github.io/


## T-MARS Text Masking Code
We use FAST (https://github.com/czczup/FAST) as the base algorithm for text detection and MMOCR (https://mmocr.readthedocs.io/en/dev-1.x/) for text recognition i.e. reading the text. This repository shares the combined implementation of FAST and MMOCR for running on web-scales (adapted from DataComp https://www.datacomp.ai/).


## Installation

```sh
git clone https://github.com/locuslab/T-MARS.git
cd T-MARS
conda create -n tmars python=3.10 -y
conda activate tmars
pip install -e .
```

For MMOCR installation (this is only needed if you want to do OCR detection, and is not needed for T-MARS):

```sh
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
cd dataset2metadata/text_detection
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -e .
```

## Text detection and recognition
Please refer to [text_snake_wrapper.py](https://github.com/locuslab/T-MARS/blob/main/dataset2metadata/text_detection/text_snake_wrapper.py) for the main implementation of text detection (FAST) and text recognition (MMOCR). 
Download the text detection model : 
```sh
cd dataset2metadata/text_detection
wget https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_512_finetune_ic17mlt.pth
```

## RUN

```sh
cd dataset2metadata/text_detection
dataset2metadata --yml ../../examples/slurm/text_template.yml
```

Please see the [examples/](https://github.com/locuslab/T-MARS/tree/main/examples/slurm) folder for ways in which dataset2metadata is to be used for running T-MARS on webscale. You can specify the tar file paths in [text_template.yml](https://github.com/locuslab/T-MARS/blob/main/examples/slurm/text_template.yml) and create multiple such template files using [prepare_jobs.py](https://github.com/locuslab/T-MARS/blob/main/examples/slurm/prepare_jobs.py)

```
cd dataset2metadata
dataset2metadata --yml examples/text_template.yml
```

## Acknowlegements
We thank the authors of FAST, MMOCR and DataComp team for open sourcing their code bases. 
