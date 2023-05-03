import logging
import os
from typing import List

import pandas as pd
import numpy as np
import torch
import fsspec

logging.getLogger().setLevel(logging.INFO)

class Writer(object):

    def __init__(
            self,
            name: str,
            feature_fields: List[str],
            parquet_fields: List[str],
        ) -> None:
        self.name = name

        # store things like CLIP features, ultimately in an npz
        self.feature_store = {e: [] for e in feature_fields}

        # store other metadata like image height, ultimately in a parquet
        self.parquet_store = {e: [] for e in parquet_fields}

    def update_feature_store(self, k, v):
        self.feature_store[k].append(v)

    def update_parquet_store(self, k, v):
        self.parquet_store[k].append(v)

    def write(self, out_dir_path):
        try:
            for k in self.feature_store:
                self.feature_store[k] = self._flatten_helper(self.feature_store[k], to_npy=True)

            for k in self.parquet_store:
                self.parquet_store[k] = self._flatten_helper(self.parquet_store[k])

            if len(self.parquet_store):
                df = pd.DataFrame.from_dict(self.parquet_store)
                df.to_parquet(os.path.join(out_dir_path, f'{self.name}.parquet'), engine='pyarrow')
                logging.info(f'saved metadata: {f"{self.name}.parquet"}')

            if len(self.feature_store):
                fs, output_path = fsspec.core.url_to_fs(os.path.join(out_dir_path, f'{self.name}.npz'))
                with fs.open(output_path, "wb") as f:
                    np.savez_compressed(f, **self.feature_store)
                    logging.info(f'saved features: {f"{self.name}.npz"}')

                return True

        except Exception as e:
            logging.exception(e)
            logging.error(f'failed to write metadata for shard: {self.name}')
            return False

    def _flatten_helper(self, l, to_npy=False):
        if len(l):
            if torch.is_tensor(l[0]):
                if to_npy:
                    return torch.cat(l, dim=0).float().numpy()
                return torch.cat(l, dim=0).float().tolist()
            else:
                l_flat = []
                for e in l:
                    l_flat.extend(e)

                return l_flat
        return l