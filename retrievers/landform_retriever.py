from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import os
import logging

from .base import RetrieverBase


@dataclass
class LandformRetriever(RetrieverBase):
    args: object
    database: Dict

    def search(self, query, query_names: Optional[List[str]] = None) -> Dict:
        if isinstance(query, (tuple, list)) and len(query) == 2 and query_names is None:
            query, query_names = query
        index = self.database["index"]
        metadata = self.database["metadata"]
        labels = self.database["labels"]

        if getattr(self.args, "top_k", None) is None or self.args.top_k <= 0:
            top_k = index.ntotal
        else:
            top_k = min(self.args.top_k, index.ntotal)

        query = query.astype("float32")
        distances, indices = index.search(query, top_k)

        results = {
            "query_names": query_names or [f"query_{i}" for i in range(indices.shape[0])],
            "image_names": [],
            "similarities": [],
            "labels": [],
        }

        for row_idx, (row_inds, row_dist) in enumerate(zip(indices, distances)):
            image_names = []
            sims = []
            row_labels = []
            for idx, dist in zip(row_inds, row_dist):
                if idx < 0:
                    continue
                image_names.append(metadata[idx])
                sims.append(float(dist))
                row_labels.append(labels[idx])
            results["image_names"].append(image_names)
            results["similarities"].append(sims)
            results["labels"].append(row_labels)

        return results

    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        rows = []
        for query_name, image_names, sims, labels in zip(
            results["query_names"], results["image_names"], results["similarities"], results["labels"]
        ):
            for image_name, sim, label in zip(image_names, sims, labels):
                rows.append(
                    {
                        "query_name": query_name,
                        "image_name": image_name,
                        "similarity": np.round(sim, 6),
                        "label": label,
                    }
                )
        return pd.DataFrame(rows)

    def save_results(self, output_dir: str, df_results: pd.DataFrame, timestamp: str) -> None:
        if not getattr(self.args, "save_details", False):
            return
        csv_name = f"{timestamp}_all.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df_results.to_csv(csv_path, index=False)
        logging.info("Saved landform retrieval results to %s", csv_path)

        details_dir = os.path.join(output_dir, "landform_details", timestamp)
        os.makedirs(details_dir, exist_ok=True)
        for query_name, df_q in df_results.groupby("query_name"):
            safe_name = "".join(
                ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(query_name)
            )
            class_path = os.path.join(details_dir, f"{safe_name}.csv")
            df_q.sort_values("similarity", ascending=False).to_csv(class_path, index=False)
        logging.info("Saved per-class landform details to %s", details_dir)
