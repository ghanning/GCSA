import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from ..fov import fov2d_overlap_list, fov2d_overlap_pairs
from ..utils import chunked_topk, get_descriptors
from .scene import MslsScene


class MslsDataset(torch.utils.data.Dataset):
    """! Mapillary Street-level Sequences dataset class."""

    def __init__(
        self,
        scene: MslsScene,
        desc_path: Path,
        N: int,
        affinity: List[str],
        labels: Optional[str],
        fov_thr: float = 1 / 3,
        dist_thr: float = 25.0,
    ) -> None:
        """! Class initializer.

        @param scene Scene.
        @param desc_path Path to HDF5 file with global image descriptors.
        @param N Number of database images to return for each query.
        @param affinity Types of affinity ("positional-db": field-of-view overlap, "heading"/"heading-db": heading
                        angle difference)
        @param labels Type of labels ("fov": field-of-view overlap, "dist": distance, None).
        @param fov_thr Field-of-view overlap threshold.
        @param dist_thr Distance threshold.
        """
        super().__init__()

        if labels not in ("fov", "dist", None):
            raise ValueError(f"Invalid label type {labels}")

        logging.debug(f"MSLS scene {scene.name}")

        db_image_list = list(map(str, scene.db_image_paths()))
        query_image_list = list(map(str, scene.query_image_paths()))

        logging.debug(f"{len(db_image_list)} db images and {len(query_image_list)} query images")

        if affinity or labels == "fov":
            db_meta = scene.db_metadata()
            self.db_heading = np.array([np.radians(m.ca) for m in db_meta])

        if "heading" in affinity or labels == "fov":
            query_meta = scene.query_metadata()
            self.query_heading = np.array([np.radians(m.ca) for m in query_meta])

        if "positional-db" in affinity or labels:
            self.db_coords = scene.db_utm_coords()

        if labels:
            self.query_coords = scene.query_utm_coords()

        self.db_desc = torch.from_numpy(get_descriptors(desc_path, db_image_list))
        self.query_desc = torch.from_numpy(get_descriptors(desc_path, query_image_list))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.topk = chunked_topk(self.query_desc.to(device), self.db_desc.to(device), N)

        if labels == "dist":
            nn = NearestNeighbors()
            nn.fit(self.db_coords)
            nbr = nn.radius_neighbors(self.query_coords, dist_thr, return_distance=False)
            self.num_neighbors = torch.tensor([len(n) for n in nbr])
        else:
            self.num_neighbors = torch.empty(len(self), 0)

        self.desc_path = desc_path
        self.affinity = affinity
        self.labels = labels
        self.fov_thr = fov_thr
        self.dist_thr = dist_thr

    def _labels(self, query_idx: int, db_idx: np.ndarray) -> torch.Tensor:
        if self.labels == "fov":
            overlap = fov2d_overlap_list(
                self.query_coords[query_idx],
                self.query_heading[query_idx],
                self.db_coords[db_idx],
                self.db_heading[db_idx],
            )
            labels = torch.from_numpy(overlap >= self.fov_thr)
            labels = torch.cat((torch.ones(1, dtype=bool), labels))
        elif self.labels == "dist":
            labels = np.linalg.norm(self.query_coords[query_idx] - self.db_coords[db_idx], axis=1) <= self.dist_thr
            labels = torch.cat((torch.ones(1, dtype=bool), torch.from_numpy(labels)))
        else:
            labels = torch.tensor([])
        return labels

    def _affinity(self, query_idx: int, db_idx: np.ndarray) -> List[torch.Tensor]:
        aff = []

        for a in self.affinity:
            if a == "positional-db":
                aff.append(torch.from_numpy(fov2d_overlap_pairs(self.db_coords[db_idx], self.db_heading[db_idx])))
            elif a in ("heading", "heading-db"):
                heading = self.db_heading[db_idx]
                if a == "heading":
                    heading = np.concatenate([[self.query_heading[query_idx]], heading])
                heading_diff = np.abs(heading - heading[None].T)
                heading_diff = np.minimum(heading_diff, 2 * np.pi - heading_diff)
                aff.append(torch.from_numpy((np.pi - 2 * heading_diff) / np.pi).float())
            else:
                raise ValueError(f"Invalid affinity type {a}")

        return aff

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """! Get item.

        @param idx Query image index (into self.query_desc / self.topk).
        @return The image descriptors (1+N, D) where the query is at index 0, the thresholded field-of-view overlap
                between query and the other images (1+N), the affinity between images ((N, N) or (1+N, 1+N)), the
                database image indices (N) and the number of relevant database images.
        """
        db_idx = self.topk[idx]
        query_desc = self.query_desc[idx].unsqueeze(0)
        db_desc = self.db_desc[db_idx]
        desc = torch.cat((query_desc, db_desc), dim=0)

        labels = self._labels(idx, db_idx)
        aff = self._affinity(idx, db_idx)

        return desc, labels, aff, db_idx, self.num_neighbors[idx]

    def __len__(self) -> int:
        """! Get number of items.

        @return The number of items in the dataset.
        """
        return self.query_desc.shape[0]
