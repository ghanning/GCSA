import logging
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch


def root_dir() -> Path:
    return Path(__file__).parent.parent


def datasets_dir() -> Path:
    return root_dir() / "datasets"


def outputs_dir() -> Path:
    return root_dir() / "outputs"


def get_descriptors(path: Path, names: Iterable, key: str = "global_descriptor") -> np.ndarray:
    """! Read image descriptors from HDF5 file.

    @param path Path to HDF5 file.
    @param names Image names.
    @param key Key.
    @return The stacked image descriptors.
    """
    start = time.time()
    with h5py.File(str(path), "r") as fd:
        desc = [fd[n][key].__array__() for n in names]
    desc = np.stack(desc, 0)
    elapsed = time.time() - start
    logging.debug(f"Read {desc.shape[0]} descriptors from {path} in {elapsed:.3f} s")
    return desc


def chunked_topk(query_desc: torch.Tensor, db_desc: torch.Tensor, k: int, chunk_size: int = 4000) -> torch.Tensor:
    """! Computes the top k database descriptors for each query descriptor.

    @param query_desc Query descriptors.
    @param db_desc Database descriptors.
    @param k Number of top database matches to compute.
    @param chunk_size Number of queries in each chunk.
    @return The indicies to the top k database images.
    """
    start = time.time()
    topk = torch.empty((query_desc.shape[0], k), dtype=torch.int32)
    for i in range(0, query_desc.shape[0], chunk_size):
        sim = query_desc[i : i + chunk_size] @ db_desc.T
        topk[i : i + chunk_size] = torch.topk(sim, k).indices.cpu()
    elapsed = time.time() - start
    logging.debug(f"Computed top {k} database matches for {query_desc.shape[0]} queries in {elapsed:.3f} s")
    return topk
