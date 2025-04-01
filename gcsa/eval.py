import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .model import GCSA


def apk(rel: np.ndarray, num_rel: np.ndarray, k: int) -> np.ndarray:
    """! Compute average precision.

    @param rel Array indicating whether a prediction is relevant (n x m, where m ≥ k)
    @param num_rel Number of relevant items (n).
    @param k The integer k to compute AP@k for.
    @return The average precision at k (n).
    """
    precision = np.cumsum(rel[:, :k], axis=1) / np.arange(1, k + 1)
    return np.sum(precision * rel[:, :k], axis=1) / np.minimum(k, num_rel)


def mapk(rel: np.ndarray, num_rel: np.ndarray, k: int) -> float:
    """! Compute mean average precision.

    @param rel Array indicating whether a prediction is relevant (n x m, where m ≥ k)
    @param num_rel Number of relevant items (n).
    @param k The integer k to compute mAP@k for.
    @return The mean average precision at k.
    """
    return np.mean(apk(rel, num_rel, k))


@torch.no_grad()
def retrieve(
    loader: torch.utils.data.DataLoader, model: Optional[GCSA], device: str, k: int, return_idx: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """! Image retrieval (with optional re-ranking).

    @param loader Data loader.
    @param model Optional GCSA model.
    @param device Device.
    @param k Number of database images to retrieve.
    @param return_idx Whether to return the database image indices.
    @return Returns the database image indices if return_idx is set and the labels and number of relevant database
            images otherwise.
    """
    num_queries = len(loader.sampler)
    if return_idx:
        idx = torch.empty((num_queries, k), dtype=torch.int32)
    else:
        labels = torch.empty((num_queries, k), dtype=bool)
        num_rel = torch.empty(num_queries, dtype=torch.int32)
    start = 0

    if model:
        model.eval()

    for desc, label, aff, db_idx, nrel in loader:
        desc, label, aff, db_idx, nrel = (
            desc.to(device),
            label.to(device),
            [a.to(device) for a in aff],
            db_idx.to(device),
            nrel.to(device),
        )

        if model:
            with torch.autocast(device):
                desc = model(desc, aff)
        sim = desc[:, 0].unsqueeze(1) @ desc[:, 1:].transpose(1, 2)
        topk = torch.topk(sim.squeeze(1), k).indices

        end = start + desc.shape[0]
        if return_idx:
            idx[start:end] = torch.gather(db_idx, 1, topk)
        else:
            labels[start:end] = torch.gather(label[:, 1:], 1, topk)
            num_rel[start:end] = nrel
        start = end

    if return_idx:
        return idx
    else:
        return labels, num_rel


def evaluate(
    loader: torch.utils.data.DataLoader, model: Optional[GCSA], device: str, k: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """! Compute image retrieval metrics.

    @param loader Data loader.
    @param model Optional GCSA model.
    @param device Device.
    @param k The thresholds for which to compute mAP@k and Recall@k.
    @return Returns a dictionary with the mAP and recall values.
    """
    rel, num_rel = retrieve(loader, model, device, max(k))

    rel = rel[num_rel > 0].numpy()
    num_rel = num_rel[num_rel > 0].numpy()

    res = dict()

    for i in range(len(k)):
        recall_at_k = np.mean(np.any(rel[:, : k[i]], axis=1))
        logging.info(f"Recall@{k[i]}: {100.0 * recall_at_k:.2f}%")
        res[f"recall@{k[i]}"] = recall_at_k

    for i in range(len(k)):
        map_at_k = mapk(rel, num_rel, k[i])
        logging.info(f"mAP@{k[i]}: {100.0 * map_at_k:.2f}%")
        res[f"map@{k[i]}"] = map_at_k

    return res
