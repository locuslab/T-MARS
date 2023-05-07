
import inspect
from functools import partial

import dataset2metadata.models as models
import dataset2metadata.postprocessors as post
import dataset2metadata.preprocessors as pre

# Models
model_lookup = {
    cls.name: cls for _, cls in models.__dict__.items() if hasattr(cls, 'name') and inspect.isclass(cls)
}

# Preprocessors
preprocessor_lookup = {
    'clip-aug': pre.oai_clip_image,
    'clip-tokens': pre.oai_clip_text,
    'identity': pre.identity,
    'dedup-aug': pre.dedup,
    'faces-aug': pre.faces_scrfd,
    'texts-aug': pre.text_detection,
}

# Postprocessors
postprocess_parquet_lookup = {
    'oai-clip-vit-b32-score': partial(post.batched_dot_product, model='oai-clip-vit-b32'),
    'oai-clip-vit-l14-score': partial(post.batched_dot_product, model='oai-clip-vit-l14'),
    'nsfw-detoxify-score': partial(post.identity, model='nsfw-detoxify'),
    'nsfw-image-score': partial(post.identity, model='nsfw-image-oai-clip-vit-l-14'),
    'dedup-isc-ft-v107-score': partial(post.select, model='dedup-isc-ft-v107', index=1),
    'json-transpose': partial(post.transpose_list, model='json'),
    'face-boxes': partial(post.identity, model='faces-scrfd10g', to_cpu=False),
    'text-boxes': partial(post.return_first, model='texts-mmocr-snake', to_cpu=False),
    'text-ocr': partial(post.return_second, model='texts-mmocr-snake', to_cpu=False),
}

postprocess_feature_lookup = {
    'oai-clip-vit-b32-image': partial(post.select, model='oai-clip-vit-b32', index=0),
    'oai-clip-vit-b32-text': partial(post.select, model='oai-clip-vit-b32', index=1),
    'oai-clip-vit-l14-image': partial(post.select, model='oai-clip-vit-l14', index=0),
    'oai-clip-vit-l14-text': partial(post.select, model='oai-clip-vit-l14', index=1),
    'dedup-isc-ft-v107-image': partial(post.select, model='dedup-isc-ft-v107', index=0),
}

# update functions
def update_registry(module):

    global model_lookup
    global preprocessor_lookup
    global postprocess_parquet_lookup
    global postprocess_feature_lookup

    model_lookup = {
        **model_lookup,
        **module.model_lookup,
    }

    preprocessor_lookup = {
        **preprocessor_lookup,
        **module.preprocessor_lookup
    }

    postprocess_parquet_lookup = {
        **postprocess_parquet_lookup,
        **module.postprocess_parquet_lookup,
    }

    postprocess_feature_lookup = {
        **postprocess_feature_lookup,
        **module.postprocess_feature_lookup,
    }