import os
import pathlib
from importlib.machinery import SourceFileLoader
from typing import List

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import logging
from pathlib import Path

import fsspec
import yaml
from PIL import ImageFile

from dataset2metadata.dataloaders import create_loader
from dataset2metadata.registry import update_registry
from dataset2metadata.utils import topsort
from dataset2metadata.writer import Writer

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.getLogger().setLevel(logging.INFO)

def check_yml(yml):

    # manditory fields in the yml
    yml_fields = [
        'models',
        'nworkers',
        'batch_size',
        'device',
        'input_tars',
        'output_metadata_dir',
    ]

    yml_optional_fields = [
        'postprocess_columns',
        'postprocess_features',
        'additional_fields',
        'custom_pypath',
        'reprocess',
    ]

    for f in yml_fields:
        if f not in yml:
            raise ValueError(f'missing required yml field: {f}')

    for f in yml:
        if f not in yml_fields + yml_optional_fields:
            raise ValueError(f'unknown field: {f}')

def process(
    yml,
):
    if type(yml) is str:
        # parse yml and check resulting dict
        yml = yaml.safe_load(Path(yml).read_text())

    check_yml(yml)

    # if local out dir does not exist make it
    fs, output_path = fsspec.core.url_to_fs(yml['output_metadata_dir'])
    fs.makedirs(output_path, exist_ok=True)

    # assign a name to the group of shards being processed
    name = hashlib.md5(str(yml['input_tars']).encode()).hexdigest()

    # cache if result already there and user does not want to reprocess
    if 'reprocess' not in yml or not yml['reprocess']:
        # cache
        completed = fs.ls(output_path)
        completed_parquets = [p for p in completed if 'parquet' in p]
        if name in set([Path(s).stem for s in completed_parquets]):
            logging.info(f'found cached result: {name}')
            return

    # if the user specifies specific custom implementaion of their own update the registry
    # print(yml['custom_pypath'])
    # if yml['custom_pypath'] is None:
    #     yml['custom_pypath'] = "null"
    # print(yml['custom_pypath'])
    # if 'custom_pypath' in yml and yml['custom_pypath'] is not None:
    #     custom = SourceFileLoader(
    #         pathlib.Path(yml['custom_pypath']).stem,
    #         yml['custom_pypath']
    #     ).load_module()

    #     update_registry(custom)

    # import from registry here after we have updated
    from dataset2metadata.registry import (model_lookup,
                                           postprocess_feature_lookup,
                                           postprocess_parquet_lookup)

    # create dataloader based on user input
    dataloader, input_map = create_loader(
        yml['input_tars'],
        yml['models'],
        yml['additional_fields'],
        yml['nworkers'],
        yml['batch_size'],
    )

    # initializing models
    models = {m_str: model_lookup[m_str](yml['device']) for m_str in yml['models']}

    # deciding order to run them in based on dependencies
    topsort_order = topsort(
        {m_str: model_lookup[m_str].dependencies for m_str in yml['models']}
    )

    logging.info(f'topsort model evaluation order: {topsort_order}')

    # initialize the writer that stores results and dumps them to store
    # TODO: fix the name here
    feature_fields = []
    parquet_fields = []
    if 'postprocess_features' in yml:
        feature_fields = yml['postprocess_features']
    if 'postprocess_columns' in yml:
        parquet_fields.extend(yml['postprocess_columns'])
    if 'additional_fields' in yml:
        parquet_fields.extend(yml['additional_fields'])

    writer = Writer(
        name,
        feature_fields,
        parquet_fields,
    )

    from tqdm import tqdm
    for sample in tqdm(dataloader):
        model_outputs = {}

        # eval all models sequentially in a top sort order
        for m_str in topsort_order:

            model_input = []
            cache = {}

            # fill the model input
            for i in input_map[m_str]:

                if isinstance(i, int):
                    if models[m_str].to_device and i not in cache:
                        if isinstance(sample[i], List):
                            # if list needs to be moved to device transpose and move it
                            sample[i] = list(zip(*sample[i]))
                            for j in range(len(sample[i])):
                                sample[i][j] = torch.cat(sample[i][j]).to(yml['device'])
                            cache[i] = sample[i]
                        else:
                            cache[i] = sample[i].to(yml['device'])
                    else:
                        cache[i] = sample[i]

                    model_input.append(cache[i])
                else:
                    # use previously computed outputs and new inputs
                    # NOTE: assume downstream model consumes on same device as upstream
                    assert i in model_outputs
                    model_input.append(model_outputs[i])

            with torch.no_grad():
                model_outputs[m_str] = models[m_str](*model_input)

            # TODO: make this more general, right now assumes last entry is json fields
            if len(yml['additional_fields']):
                model_outputs['json'] = sample[-1]

        if 'postprocess_features' in yml:
            for k in yml['postprocess_features']:
                writer.update_feature_store(k, postprocess_feature_lookup[k](model_outputs))

        if 'postprocess_columns' in yml:
            for k in yml['postprocess_columns']:
                writer.update_parquet_store(k, postprocess_parquet_lookup[k](model_outputs))

        # if additional fields from json need to be saved, add those to the store
        if 'additional_fields' in yml and len(yml['additional_fields']):
            transposed_additional_fields = postprocess_parquet_lookup['json-transpose'](model_outputs)
            assert len(transposed_additional_fields) == len(yml['additional_fields'])
            for i, v in enumerate(transposed_additional_fields):
                writer.update_parquet_store(yml['additional_fields'][i], v)

    writer.write(yml['output_metadata_dir'])
