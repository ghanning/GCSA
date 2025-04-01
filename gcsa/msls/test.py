import argparse
from pathlib import Path
from typing import Dict, List

import torch

from ..checkpoint import load_checkpoint
from ..eval import retrieve
from ..model import GCSA
from ..utils import datasets_dir, outputs_dir
from .dataset import MslsDataset
from .scene import SCENES_TEST, MslsScene


def main(
    model: GCSA, device: str, affinity: List[str], desc_fn: str, batch_size: int, num_workers: int, k: int
) -> Dict[str, List[str]]:
    retrieval = dict()

    for name in SCENES_TEST:
        scene = MslsScene(name, datasets_dir() / "msls", exclude_panos=True, subtask="all")

        desc_path = outputs_dir() / "msls" / name / desc_fn
        N = max(model.K, model.L)
        dataset = MslsDataset(scene, desc_path, N, affinity, None)

        pin_memory = device == "cuda"
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        idx = retrieve(loader, model, device, k, return_idx=True)

        query_list, db_list = scene.query_image_paths(), scene.db_image_paths()
        retrieval.update({query_list[i]: [db_list[j] for j in idx[i]] for i in range(idx.shape[0])})

    return retrieval


def path2key(path: Path) -> str:
    return path.stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained GCSA model on the Mapillary SLS test set")
    parser.add_argument("--checkpoint", "-c", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--desc_fn", "-dfn", default="global-feats-netvlad.h5", help="HDF5 descriptor file name")
    parser.add_argument("--batch_size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", "-nw", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--k", "-k", type=int, default=20, help="The number of database images to retrieve")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to output CSV file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, affinity = load_checkpoint(args.checkpoint, device)

    retrieval = main(model, device, affinity, args.desc_fn, args.batch_size, args.num_workers, args.k)

    with open(args.output, "w") as f:
        for query_image, db_images in retrieval.items():
            f.write(f"{path2key(query_image)} {' '.join([path2key(d) for d in db_images])}\n")
