from functools import partial

import webdataset as wds
from dataset2metadata.preprocessors import json_decoder


def get_to_tuple_directives(models, additional_fields):

    # import here as registry may have updated
    from dataset2metadata.registry import model_lookup

    wrapper_classes = [model_lookup[m] for m in models]

    input_map = {}

    # get unique preprocessor directive, which is a raw_input, preprocessor pair
    unique_derectives = []

    for i, model_class in enumerate(wrapper_classes):
        assert len(model_class.preprocessors) == len(model_class.raw_inputs)

        preprocess_directives = [
            (model_class.raw_inputs[k], model_class.preprocessors[k]) for k in range(len(model_class.preprocessors))
        ]

        input_map[models[i]] = []

        for j in range(len(preprocess_directives)):
            if preprocess_directives[j] not in unique_derectives:
                input_map[models[i]].append(len(unique_derectives))
                unique_derectives.append(preprocess_directives[j])
            else:
                input_map[models[i]].append(unique_derectives.index(preprocess_directives[j]))

        if len(model_class.dependencies):
            # non-numeric, nameded dependencies, i.e., the outputs of other models
            input_map[models[i]].extend(model_class.dependencies)

    # add directives to include data from the tars into the webdataset
    if additional_fields is not None and len(additional_fields):
        # NOTE: currently no support for these additional fields being taken as inputs to models
        input_map['json'] = [len(unique_derectives), ]
        unique_derectives.append(('json', 'identity'))

    return unique_derectives, input_map

def create_loader(input_shards, models, additional_fields, nworkers, batch_size):

    # import here as registry may have updated
    from dataset2metadata.registry import preprocessor_lookup

    (
        unique_derectives,
        input_map,
    ) = get_to_tuple_directives(models, additional_fields)

    tuple_fields = [e[0] for e in unique_derectives]
    unique_preprocessors = [
        preprocessor_lookup[e[-1]] for e in unique_derectives
    ]

    pipeline = [wds.SimpleShardList(input_shards), ]

    pipeline.extend([
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode(
            'pilrgb',
            partial(json_decoder, json_keys=additional_fields),
            handler=wds.warn_and_continue),
        wds.rename(image='jpg;png;jpeg;webp', text='txt'),
        wds.to_tuple(*tuple_fields),
        wds.map_tuple(*unique_preprocessors),
        wds.batched(batch_size, partial=True),
    ])

    loader = wds.WebLoader(
        wds.DataPipeline(*pipeline),
        batch_size=None,
        shuffle=False,
        num_workers=nworkers,
        persistent_workers=True,
    )

    return loader, input_map