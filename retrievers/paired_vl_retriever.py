from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import os
import logging

from .base import RetrieverBase


@dataclass
class PairedVLRetriever(RetrieverBase):
    args: object
    database: Dict

    def search(self, query=None) -> Dict:
        image_features = self.database["image_features"]
        text_features = self.database["text_features"]
        index_text = self.database["index_text"]
        index_image = self.database["index_image"]
        keys = self.database["keys"]

        top_k = getattr(self.args, "top_k", None) or 10
        top_k = min(top_k, len(keys))

        image_features = image_features.astype("float32")
        text_features = text_features.astype("float32")

        dist_i2t, idx_i2t = index_text.search(image_features, top_k)
        dist_t2i, idx_t2i = index_image.search(text_features, top_k)

        return {
            "keys": keys,
            "image_features": image_features,
            "text_features": text_features,
            "i2t": (idx_i2t, dist_i2t),
            "t2i": (idx_t2i, dist_t2i),
        }

    def to_dataframe(self, results: Dict) -> Dict[str, pd.DataFrame]:
        keys = results["keys"]
        idx_i2t, dist_i2t = results["i2t"]
        idx_t2i, dist_t2i = results["t2i"]

        i2t_rows = []
        for q_idx, (inds, dists) in enumerate(zip(idx_i2t, dist_i2t)):
            query_key = keys[q_idx]
            for rank, (idx, dist) in enumerate(zip(inds, dists), start=1):
                i2t_rows.append(
                    {
                        "query_key": query_key,
                        "retrieved_key": keys[idx],
                        "rank": rank,
                        "similarity": np.round(float(dist), 6),
                    }
                )

        t2i_rows = []
        for q_idx, (inds, dists) in enumerate(zip(idx_t2i, dist_t2i)):
            query_key = keys[q_idx]
            for rank, (idx, dist) in enumerate(zip(inds, dists), start=1):
                t2i_rows.append(
                    {
                        "query_key": query_key,
                        "retrieved_key": keys[idx],
                        "rank": rank,
                        "similarity": np.round(float(dist), 6),
                    }
                )

        return {
            "image_to_text": pd.DataFrame(i2t_rows),
            "text_to_image": pd.DataFrame(t2i_rows),
            "__eval_input__": {
                "keys": keys,
                "image_features": results.get("image_features"),
                "text_features": results.get("text_features"),
            },
        }

    def save_results(self, output_dir: str, df_results: Dict[str, pd.DataFrame], timestamp: str) -> None:
        if not getattr(self.args, "save_details", False):
            return
        for suffix, df in df_results.items():
            if not isinstance(df, pd.DataFrame):
                continue
            csv_name = f"{timestamp}_{suffix}.csv"
            csv_path = os.path.join(output_dir, csv_name)
            df.to_csv(csv_path, index=False)
            logging.info("Saved retrieval results to %s", csv_path)
