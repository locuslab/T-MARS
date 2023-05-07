# dataset2metadata

<!-- RUN THE COMMAND -->
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YvsuxKH9M-Gseur9gc-SZJb3pCpTUddi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YvsuxKH9M-Gseur9gc-SZJb3pCpTUddi" -O dataset2metadata/text_detection/textsnake_vgg_180.pth && rm -rf /tmp/cookies.txt
'''

<!--
[![pypi](https://img.shields.io/pypi/v/dataset2metadata.svg)](https://pypi.python.org/pypi/dataset2metadata) -->

Process a downloaded dataset (currently a [webdataset](https://github.com/webdataset/webdataset) generated by [img2dataset](https://github.com/rom1504/img2dataset)). This repo implements the metadata processing for the [DataComp](https://datacomp.ai) project.

Please see the [examples/](https://github.com/mlfoundations/dataset2metadata/tree/main/examples/) folder for ways in which dataset2metadata used, including slurm examples!

## Installation

```sh
pip install git+https://github.com/mlfoundations/dataset2metadata
```
Note: pypi support coming soon!

## basic.yml
Here we provide a basic `yml`, which should run on most consumer hardware.
This includes computing:
* OpenAI [CLIP ViT-B/32](https://github.com/openai/CLIP) image, text features and CLIP scores

```yaml
models: # model directives, specifying the models to instantiate
  - oai-clip-vit-b32
postprocess_columns: # postprocessing directives
  - oai-clip-vit-b32-score
postprocess_features: # saved in an npz format
  - oai-clip-vit-b32-image
  - oai-clip-vit-b32-text
additional_fields: # fields in a webdataset json to carry over into the metadata
  - uid
  - url
  - caption
  - original_width
  - original_height
  - sha256
nworkers: 2
batch_size: 512
device: 0
input_tars: "path/to/my/tars/000057{17..19}.tar" # braceexpand suported, can also be s3 paths
output_metadata_dir: "path/to/my/ouput/metadata" # can be arbitrary path
custom_pypath: null # if model, preprocessors, postprocessors not known, look in this python file for user provided custom implementation
reprocess: True # if true will process from scratch, else will just process tars not already processed
```

Run:

```sh
dataset2metadata --yml basic.yml
```

## datacomp.yml
Here we provide a default `yml`, which is implements DataComp preprocessing.
This includes computing:
* OpenAI [CLIP ViT-B/32](https://github.com/openai/CLIP) image, text features and CLIP scores
* OpenAI [CLIP ViT-L/14](https://github.com/openai/CLIP) image, text features and CLIP scores
* [detoxify](https://github.com/unitaryai/detoxify) text toxicity scores
* NSFW image filtering (custom trained classifier on CLIP ViT-L/14 images featues)
* [ISC Descriptor](https://github.com/lyakaap/ISC21-Descriptor-Track-1st) features and near-duplicate scores against the DataComp evaluation sets.

```yaml
models: # model directives, specifying the models to instantiate
  - oai-clip-vit-b32
  - oai-clip-vit-l14
  - nsfw-detoxify
  - nsfw-image-oai-clip-vit-l-14
  - faces-scrfd10g
  - dedup-isc-ft-v107
postprocess_columns: # postprocessing directives
  - oai-clip-vit-b32-score
  - oai-clip-vit-l14-score
  - nsfw-detoxify-score
  - nsfw-image-score
  - face-boxes
  - dedup-isc-ft-v107-score
postprocess_features: # saved in an npz format
  - oai-clip-vit-b32-image
  - oai-clip-vit-b32-text
  - oai-clip-vit-l14-image
  - oai-clip-vit-l14-text
  - dedup-isc-ft-v107-image
additional_fields: # fields in a webdataset json to carry over into the metadata
  - uid
  - url
  - caption
  - original_width
  - original_height
  - sha256
nworkers: 2
batch_size: 512
device: 0
input_tars: "path/to/my/tars/000057{17..19}.tar" # braceexpand suported, can also be s3 paths
output_metadata_dir: "path/to/my/ouput/metadata" # can be arbitrary path
custom_pypath: null # if model, preprocessors, postprocessors not known, look in this python file for user provided custom implementation
reprocess: True # if true will process from scratch, else will just process tars not already processed
```

Run:

```sh
dataset2metadata --yml datacomp.yml
```
Note, this workload needs ~40GB of GPU VRAM.

## Citation

If you found this repository useful, please consider citing:

```
@article{datacomp,
  title={DataComp: In search of the next generation of multimodal datasets},
  author={Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt},
  journal={arXiv preprint arXiv:2304.14108},
  year={2023}
}
```

## Acknowlegements
Thanks to the whole DataComp team! Specifically, thanks to [Georgios Smyrnis](https://georgiossmyrnis.github.io/) for face blurring, Alex Fang for NSFW processing, and [Ryan Marten](https://www.ryanmarten.com/) for evaluation set near-duplicate removal. Thanks to [Romain Beaumont](https://rom1504.fr/) for providing technical guidence and creating [img2dataset](https://github.com/rom1504/img2dataset), which had a major influence on both this codebase and on the DataComp project.