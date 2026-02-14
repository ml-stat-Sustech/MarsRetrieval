import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import os

from .base import RetrieverBase


@dataclass
class GeoLocalizationRetriever(RetrieverBase):
    args: object
    database: Dict

    def search(self, query) -> Dict:
        index = self.database["index"]
        metadata = self.database["metadata"]
        coordinates = self.database["coordinates"]

        if not hasattr(self.args, "top_k") or self.args.top_k <= 0:
            raise ValueError("args.top_k must be a positive integer.")
        top_k = min(self.args.top_k, index.ntotal)

        logging.info("Searching for top-%s neighbors (db size=%s)", top_k, index.ntotal)
        query = query.astype("float32")
        distances, indices = index.search(query, top_k)

        image_names = []
        final_scores = []
        final_coords = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            image_names.append(metadata[idx])
            final_scores.append(float(dist))
            final_coords.append(coordinates[idx])

        results = {
            "image_names": image_names,
            "similarities": final_scores,
            "coordinates": np.array(final_coords) if final_coords else np.empty((0, 2)),
        }
        return results

    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "image_name": results["image_names"],
                "similarity": np.round(results["similarities"], 6),
                "lon": results["coordinates"][:, 0] if len(results["coordinates"]) else [],
                "lat": results["coordinates"][:, 1] if len(results["coordinates"]) else [],
            }
        )
        return df

    def save_results(self, output_dir: str, df_results: pd.DataFrame, timestamp: str) -> None:
        if not getattr(self.args, "save_details", False):
            return
        csv_name = f"{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df_results.to_csv(csv_path, index=False)
        logging.info("Saved retrieval results to %s", csv_path)
